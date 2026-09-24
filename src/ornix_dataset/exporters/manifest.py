"""Release builder — immutable local snapshot (spec §6, Phase 6).

Assembles an accepted-only release dir with audio, manifests, checksums, reports
and dataset card. ``RELEASE_READY.json`` is written ONLY when every gate passes
(all rows validate, all rights redistributable, no rejected/review rows, no
absolute paths/secrets). Never publishes.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts.release import ReleaseRow
from ..util.hashing import sha256_file
from ..util.io import write_json
from ..util.jsonl import write_jsonl
from ..util.timeutil import utc_now_iso
from ..version import __version__
from .card import render_dataset_card
from .parquet import write_parquet, write_webdataset


@dataclass
class ReleaseArtifacts:
    release_dir: str
    ready: bool
    blockers: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    n_rows: int = 0


def build_release(release_id: str, rows: List[Dict[str, Any]],
                  audio_source_dir: str, out_dir: str,
                  rights_report: Optional[Dict[str, Any]] = None,
                  quality_report: Optional[Dict[str, Any]] = None,
                  export_format: str = "parquet",
                  release_target: str = "train_only") -> ReleaseArtifacts:
    os.makedirs(out_dir, exist_ok=True)
    audio_out = os.path.join(out_dir, "audio")
    os.makedirs(audio_out, exist_ok=True)
    blockers: List[str] = []

    # 1. validate every row + copy audio into repo-relative layout
    validated: List[Dict[str, Any]] = []
    for r in rows:
        row = ReleaseRow.from_dict(r)
        try:
            row.validate(release_target=release_target)
        except ValueError as e:
            blockers.append(f"ROW_INVALID:{row.audio_id}:{e}")
            continue
        src = os.path.join(audio_source_dir, os.path.basename(row.audio))
        if not os.path.exists(src):
            blockers.append(f"AUDIO_MISSING:{row.audio_id}")
            continue
        dst = os.path.join(audio_out, os.path.basename(row.audio))
        shutil.copyfile(src, dst)
        if sha256_file(dst) != row.audio_sha256:
            blockers.append(f"AUDIO_SHA_MISMATCH:{row.audio_id}")
            continue
        row.audio = f"audio/{os.path.basename(row.audio)}"
        validated.append(row.to_dict())

    rights_report = rights_report or {"licenses": [], "note": "no rights report supplied"}
    quality_report = quality_report or {"note": "no quality report supplied"}

    # 2. manifests
    jsonl_path = os.path.join(out_dir, "release_manifest.jsonl")
    write_jsonl(jsonl_path, validated)
    files = [jsonl_path]
    if export_format == "parquet" and validated:
        try:
            files.append(write_parquet(validated, audio_out,
                                       os.path.join(out_dir, "release_manifest.parquet")))
        except Exception as e:
            blockers.append(f"PARQUET_EXPORT_FAILED:{e}")
    elif export_format == "webdataset" and validated:
        files.extend(write_webdataset(validated, audio_out, os.path.join(out_dir, "shards")))

    # 3. reports + card
    write_json(os.path.join(out_dir, "RIGHTS_REPORT.json"), rights_report)
    write_json(os.path.join(out_dir, "QUALITY_REPORT.json"), quality_report)
    card = render_dataset_card(release_id, validated, rights_report, quality_report)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)
    files += [os.path.join(out_dir, x) for x in
              ("RIGHTS_REPORT.json", "QUALITY_REPORT.json", "README.md")]

    # 4. rights gate: no non-redistributable row may exist
    if not validated:
        blockers.append("NO_ACCEPTED_ROWS")
    for row in validated:
        if row.get("quality_gate") != "ACCEPT":
            blockers.append(f"NON_ACCEPT_ROW:{row.get('audio_id')}")
        if release_target == "public":
            if (row.get("rights_status") != "REDISTRIBUTION_APPROVED"
                    or not row.get("redistribution_permitted")):
                blockers.append(f"NON_REDISTRIBUTABLE_ROW:{row.get('audio_id')}")

    ready_payload_extra = {"release_target": release_target}

    # 5. checksums over every emitted file
    _write_manifest_sha(out_dir, audio_out)
    files.append(os.path.join(out_dir, "MANIFEST.sha256"))

    ready = not blockers
    ready_payload = {
        "release_id": release_id, "ready": ready, "blockers": blockers,
        "n_rows": len(validated), "analyzer_version": __version__,
        "built_utc": utc_now_iso(),
        "export_format": export_format,
        **ready_payload_extra,
    }
    if ready:
        write_json(os.path.join(out_dir, "RELEASE_READY.json"), ready_payload)
        files.append(os.path.join(out_dir, "RELEASE_READY.json"))
    else:
        write_json(os.path.join(out_dir, "RELEASE_BLOCKED.json"), ready_payload)
    return ReleaseArtifacts(out_dir, ready, blockers, files, len(validated))


def _write_manifest_sha(out_dir: str, skip_self: str) -> None:
    lines = []
    for dirpath, _dirs, filenames in os.walk(out_dir):
        for fn in sorted(filenames):
            if fn in ("MANIFEST.sha256",):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, out_dir)
            lines.append(f"{sha256_file(full)}  {rel}")
    with open(os.path.join(out_dir, "MANIFEST.sha256"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(lines)) + "\n")
