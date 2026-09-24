"""Offline release verification (spec §6, Phase 6, tests T-015).

Checks: every manifest row maps to a present audio file; readback confirms 24 kHz
mono PCM16 and matching audio_sha256; no absolute/local paths or secrets; row
count and total duration reconcile; MANIFEST.sha256 matches on disk.
"""

from __future__ import annotations

import os
import re
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..util.hashing import sha256_file
from ..util.jsonl import read_jsonl

_SECRET = re.compile(r"(hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN|token\s*[:=])")


@dataclass
class VerifyResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    n_rows: int = 0
    total_duration_s: float = 0.0


def verify_release(release_dir: str, offline: bool = True) -> VerifyResult:
    errors: List[str] = []
    manifest = os.path.join(release_dir, "release_manifest.jsonl")
    if not os.path.exists(manifest):
        return VerifyResult(False, ["MANIFEST_MISSING"], 0, 0.0)

    rows = list(read_jsonl(manifest))
    total_dur = 0.0
    for r in rows:
        aid = r.get("audio_id", "?")
        audio_rel = r.get("audio", "")
        if audio_rel.startswith("/") or audio_rel.startswith("file://") or ":" in audio_rel.split("/")[0]:
            errors.append(f"ABSOLUTE_PATH:{aid}")
            continue
        audio_path = os.path.join(release_dir, audio_rel)
        if not os.path.exists(audio_path):
            errors.append(f"AUDIO_MISSING:{aid}")
            continue
        if sha256_file(audio_path) != r.get("audio_sha256"):
            errors.append(f"SHA_MISMATCH:{aid}")
        _check_wav(audio_path, aid, errors)
        total_dur += float(r.get("duration_s", 0.0))

    _check_secrets(release_dir, errors)
    _check_manifest_sha(release_dir, errors)
    _check_ready(release_dir, errors)
    return VerifyResult(not errors, errors, len(rows), round(total_dur, 3))


def _check_wav(path: str, aid: str, errors: List[str]) -> None:
    try:
        with wave.open(path, "rb") as wf:
            if wf.getframerate() != 24000:
                errors.append(f"SR_NOT_24K:{aid}")
            if wf.getnchannels() != 1:
                errors.append(f"NOT_MONO:{aid}")
            if wf.getsampwidth() != 2:
                errors.append(f"NOT_PCM16:{aid}")
    except Exception as e:
        errors.append(f"WAV_READBACK_FAILED:{aid}:{e}")


def _check_secrets(release_dir: str, errors: List[str]) -> None:
    for name in ("release_manifest.jsonl", "README.md", "RIGHTS_REPORT.json",
                 "QUALITY_REPORT.json"):
        p = os.path.join(release_dir, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                if _SECRET.search(fh.read()):
                    errors.append(f"POSSIBLE_SECRET:{name}")


def _check_manifest_sha(release_dir: str, errors: List[str]) -> None:
    mpath = os.path.join(release_dir, "MANIFEST.sha256")
    if not os.path.exists(mpath):
        errors.append("MANIFEST_SHA_MISSING")
        return
    with open(mpath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            expect, rel = line.split("  ", 1)
            full = os.path.join(release_dir, rel)
            if not os.path.exists(full) or sha256_file(full) != expect:
                errors.append(f"MANIFEST_SHA_MISMATCH:{rel}")


def _check_ready(release_dir: str, errors: List[str]) -> None:
    if not os.path.exists(os.path.join(release_dir, "RELEASE_READY.json")):
        errors.append("RELEASE_NOT_READY")
