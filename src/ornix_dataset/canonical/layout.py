"""Hugging Face output layout for the unified Ornix Dataset (spec §6).

Public tree (nothing else is published)::

    Ornix-Datasets/
        README.md
        MANIFEST.sha256
        RELEASE_READY.json
        train/metadata.jsonl
        train/audio/<first-2-hex>/ornix_<32hex>.wav
        validation/...
        test/...

- Files are sharded by the first two hex chars of the opaque id; no source
  dataset name or speaker folder ever appears.
- Empty splits are not created.
- ``metadata.jsonl`` of a split holds exactly the six canonical fields and the
  audio path is relative to that metadata file.
- Writing is incremental and single-writer: existing rows for other samples are
  preserved (an atomic merge), and a sample is never duplicated.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List

from ..util.io import atomic_write_text, write_json
from ..util.timeutil import utc_now_iso
from .schema import CanonicalRow

SPLITS = ("train", "validation", "test")
MANIFEST_NAME = "MANIFEST.sha256"
READY_NAME = "RELEASE_READY.json"
METADATA_NAME = "metadata.jsonl"


def split_dir(dataset_dir: str, split: str) -> str:
    return os.path.join(dataset_dir, split)


def metadata_path(dataset_dir: str, split: str) -> str:
    return os.path.join(split_dir(dataset_dir, split), METADATA_NAME)


def audio_abs_path(dataset_dir: str, split: str, row: CanonicalRow) -> str:
    return os.path.join(split_dir(dataset_dir, split), *row.audio.split("/"))


def read_split_rows(dataset_dir: str, split: str) -> Dict[str, CanonicalRow]:
    """Existing rows for a split, keyed by ``file_name`` (fail-closed on bad rows)."""
    path = metadata_path(dataset_dir, split)
    rows: Dict[str, CanonicalRow] = {}
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = CanonicalRow.from_dict(json.loads(line))
            if row.file_name in rows:
                raise ValueError(f"duplicate metadata row for {row.file_name}")
            rows[row.file_name] = row
    return rows


def read_all_rows(dataset_dir: str) -> Dict[str, Dict[str, CanonicalRow]]:
    return {s: read_split_rows(dataset_dir, s) for s in SPLITS}


def write_split_rows(dataset_dir: str, split: str,
                     rows: Dict[str, CanonicalRow]) -> None:
    """Atomically rewrite a split's metadata.jsonl (sorted, unicode-preserving)."""
    os.makedirs(split_dir(dataset_dir, split), exist_ok=True)
    body = "".join(json.dumps(rows[k].to_dict(), ensure_ascii=False,
                              sort_keys=True) + "\n" for k in sorted(rows))
    atomic_write_text(metadata_path(dataset_dir, split), body)


def upsert_rows(dataset_dir: str, split: str,
                new_rows: Iterable[CanonicalRow]) -> Dict[str, int]:
    """Merge rows into a split without dropping or duplicating other samples."""
    existing = read_split_rows(dataset_dir, split)
    added = reused = 0
    for row in new_rows:
        row.validate()
        prior = existing.get(row.file_name)
        if prior is not None:
            if prior.to_dict() != row.to_dict():
                raise ValueError(
                    f"conflicting metadata for existing row {row.file_name}")
            reused += 1
            continue
        existing[row.file_name] = row
        added += 1
    write_split_rows(dataset_dir, split, existing)
    return {"added": added, "reused": reused, "total": len(existing)}


def split_stats(dataset_dir: str) -> Dict[str, Any]:
    rows = read_all_rows(dataset_dir)
    by_split = {s: len(rows[s]) for s in SPLITS}
    languages: Dict[str, int] = {}
    speakers = set()
    total_duration = 0.0
    for s in SPLITS:
        for row in rows[s].values():
            languages[row.language] = languages.get(row.language, 0) + 1
            speakers.add(row.speaker)
            total_duration += row.duration
    return {"n_rows": sum(by_split.values()), "by_split": by_split,
            "languages": dict(sorted(languages.items())),
            "n_speakers": len(speakers),
            "total_duration_s": round(total_duration, 6)}


def write_manifest_sha(dataset_dir: str) -> str:
    """Checksum manifest over every published file (except the manifest itself).

    Reused by the approval gate: ``release_digest`` hashes this file, and the
    exact-set remote verification compares the remote tree to it.
    """
    from ..util.hashing import sha256_file

    lines: List[str] = []
    for dp, _dirs, files in os.walk(dataset_dir):
        for fn in sorted(files):
            if fn == MANIFEST_NAME:
                continue
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, dataset_dir)
            lines.append(f"{sha256_file(full)}  {rel}")
    out = os.path.join(dataset_dir, MANIFEST_NAME)
    atomic_write_text(out, "\n".join(sorted(lines)) + "\n")
    return out


def write_ready(dataset_dir: str, stats: Dict[str, Any],
                rights: Dict[str, Any]) -> str:
    payload = {"dataset": "Ornix-Datasets", "ready": True,
               "generated_utc": utc_now_iso(), "n_rows": stats["n_rows"],
               "by_split": stats["by_split"], "languages": stats["languages"],
               "n_speakers": stats["n_speakers"], "rights": rights}
    out = os.path.join(dataset_dir, READY_NAME)
    write_json(out, payload)
    return out
