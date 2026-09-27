"""Parquet container expansion for campaign batches (anti-slow-download).

Standalone ``scripts/extract_hf_audio.py`` already handles whole-dataset
extraction with bounded disk (one shard at a time). This module wires the same
shape into the batch path: a ``.parquet`` row in ``BATCH_MANIFEST.jsonl`` is
expanded into per-clip ``.wav`` rows in the same staging dir, so
``run_qc_batch`` sees only audio it can QC. The original parquet file is kept
on disk until batch cleanup (provenance) but never QCed directly.

Fail-closed: a parquet that cannot be read, has no audio bytes, or yields zero
clips is recorded as a blocker, never silently skipped.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Tuple

_MAGIC_EXTS = (
    (b"RIFF", ".wav"),
    (b"fLaC", ".flac"),
    (b"OggS", ".ogg"),
    (b"ID3", ".mp3"),
)


def _sniff_ext(blob: bytes) -> str:
    for magic, ext in _MAGIC_EXTS:
        if blob.startswith(magic):
            return ext
    if len(blob) >= 2 and blob[0] == 0xFF and (blob[1] & 0xE0) == 0xE0:
        return ".mp3"
    if len(blob) >= 12 and blob[4:8] == b"ftyp":
        return ".m4a"
    return ".wav"


def is_parquet_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".parquet"


def _flat(base: str) -> str:
    return base.replace("/", "__")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expand_parquet_row(staging: str, row: Dict[str, Any],
                       audio_col: str = "audio",
                       text_col: str = "transcription",
                       language: str = "vi") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Expand one parquet manifest row into per-clip wav manifest rows.

    Returns (new_rows, report). New rows carry the same keys as
    ``_manifest_row`` plus ``source_transcript``/``source_language`` so
    ``manifest_to_source_record`` can attach them. The parquet itself is NOT
    removed here — cleanup owns deletion.
    """
    import pyarrow.parquet as pq

    rel = row.get("staged_path", "")
    src_path = os.path.join(staging, rel) if not os.path.isabs(rel) else rel
    if not os.path.exists(src_path):
        return [], {"ok": False, "reason": "parquet-missing",
                    "original_file_id": row.get("original_file_id")}
    out_dir = os.path.join(staging, "extracted",
                           _flat(row.get("original_file_id", "shard")))
    os.makedirs(out_dir, exist_ok=True)
    new_rows: List[Dict[str, Any]] = []
    try:
        pf = pq.ParquetFile(src_path)
    except Exception as e:
        return [], {"ok": False, "reason": f"parquet-unreadable: {e}",
                    "original_file_id": row.get("original_file_id")}
    # The ``audio:{bytes,path}`` struct reads back as one ``audio`` column
    # even when ``schema.names`` shows the flattened ``bytes``/``path``
    # leaves — so try the struct read first, fall back to flat leaves.
    written = 0
    try:
        for rg in range(pf.num_row_groups):
            pairs = None
            try:
                cols = [audio_col] + ([text_col] if text_col else [])
                tbl = pf.read_row_group(rg, columns=cols)
                d = tbl.to_pydict()
                audios = d.get(audio_col, [])
                texts = d.get(text_col, [None] * len(audios))
                pairs = [(a.get("bytes") if isinstance(a, dict) else None,
                          (a.get("path") if isinstance(a, dict) else None), t)
                         for a, t in zip(audios, texts)]
            except Exception:
                tbl = pf.read_row_group(rg)
                d = tbl.to_pydict()
                blobs = d.get("bytes", [])
                if not blobs:
                    raise ValueError("parquet-no-audio-column")
                paths = d.get("path", [None] * len(blobs))
                texts = d.get(text_col, [None] * len(blobs))
                pairs = list(zip(blobs, paths, texts))
            for blob, apath, text in pairs:
                if not blob:
                    continue
                ext = _sniff_ext(blob)
                base = os.path.basename(apath) if apath else ""
                stem, bext = os.path.splitext(base)
                if bext.lower() not in (".wav", ".flac", ".mp3", ".ogg",
                                        ".opus", ".m4a", ".aac", ".wv"):
                    name = (stem or f"clip_{written:06d}") + ext
                else:
                    name = base or f"clip_{written:06d}{ext}"
                # Prefix with shard flat name for uniqueness across shards.
                fname = f"{_flat(row.get('original_file_id', 'shard'))}" \
                        f"__row-{written:06d}__{name}"
                fpath = os.path.join(out_dir, fname)
                if not os.path.exists(fpath):
                    with open(fpath, "wb") as fh:
                        fh.write(blob)
                    os.chmod(fpath, 0o444)
                new_rows.append({
                    "source_uri": row.get("source_uri", "") +
                                  f"#row-{written}",
                    "original_file_id": f"{row.get('original_file_id')}"
                                        f"#row-{written}",
                    "source_revision": row.get("source_revision"),
                    "staged_path": os.path.relpath(fpath, staging),
                    "sha256": _sha256_file(fpath),
                    "size": os.path.getsize(fpath),
                    "verify_method": "extracted-sha256",
                    "from_cache": False,
                    "reused_staged": os.path.exists(fpath),
                    "source_transcript": text,
                    "source_language": language,
                })
                written += 1
    except Exception as e:
        return [], {"ok": False, "reason": f"parquet-extract-failed: {e}",
                    "original_file_id": row.get("original_file_id")}
    if not new_rows:
        return [], {"ok": False, "reason": "parquet-zero-clips",
                    "original_file_id": row.get("original_file_id")}
    return new_rows, {"ok": True, "n_clips": len(new_rows),
                      "original_file_id": row.get("original_file_id")}


def expand_batch_parquet(store: Any, job: Any, batch: Any,
                         staging: str) -> Dict[str, Any]:
    """Expand every parquet in the batch manifest; append clip rows atomically.

    Idempotent: clip rows carry ``original_file_id = parquet#row-i`` so a
    re-run finds them in the manifest and skips re-extraction. Returns a
    summary with per-shard reports.
    """
    from .downloading import read_manifest, _write_manifest

    rows = read_manifest(staging)
    have = {r.get("original_file_id") for r in rows}
    reports: List[Dict[str, Any]] = []
    added = 0
    changed = False
    for row in list(rows):
        fid = row.get("original_file_id", "")
        if not is_parquet_path(fid.split("#")[0]):
            continue
        # Already expanded? clip rows exist with this prefix.
        prefix = fid + "#row-"
        if any(h.startswith(prefix) for h in have):
            reports.append({"ok": True, "original_file_id": fid,
                            "note": "already-expanded"})
            continue
        new_rows, rep = expand_parquet_row(staging, row)
        reports.append(rep)
        ckpt = store.batch_checkpoint(batch)
        if rep.get("ok"):
            for nr in new_rows:
                if nr["original_file_id"] not in have:
                    rows.append(nr)
                    have.add(nr["original_file_id"])
                    added += 1
            ckpt.mark(fid, "EXTRACTED", n_clips=rep.get("n_clips", 0))
            changed = True
        else:
            ckpt.mark(fid, "EXTRACT_BLOCKED",
                      reason=rep.get("reason", "unknown"))
    if changed:
        merged = {r.get("original_file_id"): r for r in rows}
        _write_manifest(staging, [merged[k] for k in sorted(merged)])
        rows = read_manifest(staging)
    return {"n_parquet": sum(1 for r in rows if is_parquet_path(
        r.get("original_file_id", "").split("#")[0])),
            "n_clips_added": added, "reports": reports}
