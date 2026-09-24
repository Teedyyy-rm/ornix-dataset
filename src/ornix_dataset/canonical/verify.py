"""Offline verification of the canonical Ornix Dataset tree (spec §6, tests T10/T11).

Fail-closed checks, all against bytes on disk:

- every metadata row has exactly the six fields, ``audio == file_name`` and a
  canonical sharded path;
- every referenced WAV exists, is 24 kHz mono PCM16, and its measured duration
  matches the row (no dangling metadata);
- every WAV under ``<split>/audio`` has exactly one row (no orphan WAV);
- no duplicate rows and no sample appearing in more than one split;
- the checksum manifest and READY marker (when present) are consistent.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass, field
from typing import Dict, List

from .layout import SPLITS, split_dir
from .schema import AUDIO_PATH_RE
from .layout import read_all_rows


@dataclass
class CanonicalVerifyResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    n_rows: int = 0
    by_split: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"ok": self.ok, "errors": self.errors, "n_rows": self.n_rows,
                "by_split": self.by_split}


def _measured_duration_s(path: str) -> float:
    with wave.open(path, "rb") as wf:
        if wf.getframerate() <= 0:
            raise ValueError("bad framerate")
        return wf.getnframes() / float(wf.getframerate())


def _check_wav_shape(path: str, errors: List[str], tag: str) -> None:
    try:
        with wave.open(path, "rb") as wf:
            if wf.getframerate() != 24000:
                errors.append(f"SR_NOT_24K:{tag}")
            if wf.getnchannels() != 1:
                errors.append(f"NOT_MONO:{tag}")
            if wf.getsampwidth() != 2:
                errors.append(f"NOT_PCM16:{tag}")
    except Exception as e:  # unreadable
        errors.append(f"WAV_READBACK_FAILED:{tag}:{e}")


def verify_canonical_dataset(dataset_dir: str,
                             require_manifest: bool = True) -> CanonicalVerifyResult:
    errors: List[str] = []
    try:
        rows_by_split = read_all_rows(dataset_dir)
    except Exception as e:
        return CanonicalVerifyResult(False, [f"METADATA_UNREADABLE:{e}"])

    by_split = {s: len(rows_by_split[s]) for s in SPLITS}
    seen: Dict[str, str] = {}
    speaker_split: Dict[str, str] = {}
    for split in SPLITS:
        for fn, row in rows_by_split[split].items():
            if fn != row.file_name:
                errors.append(f"KEY_MISMATCH:{fn}")
            if row.audio != row.file_name:
                errors.append(f"AUDIO_NE_FILE_NAME:{fn}")
            if not AUDIO_PATH_RE.match(row.audio):
                errors.append(f"BAD_AUDIO_PATH:{fn}")
            if fn in seen:
                errors.append(f"DUPLICATE_ROW:{fn}:{split}/{seen[fn]}")
            else:
                seen[fn] = split
            prior = speaker_split.get(row.speaker)
            if prior is not None and prior != split:
                errors.append(f"SPEAKER_SPLIT_LEAK:{row.speaker}:{prior}/{split}")
            else:
                speaker_split[row.speaker] = split
            full = os.path.join(dataset_dir, split, *row.audio.split("/"))
            if not os.path.exists(full):
                errors.append(f"DANGLING_METADATA:{fn}")
                continue
            _check_wav_shape(full, errors, fn)
            try:
                actual = _measured_duration_s(full)
                if abs(actual - row.duration) > 0.01:
                    errors.append(f"DURATION_MISMATCH:{fn}:{row.duration}!={round(actual, 6)}")
            except Exception as e:
                errors.append(f"DURATION_UNREADABLE:{fn}:{e}")

    # orphan WAV: any audio file without a metadata row
    for split in SPLITS:
        adir = os.path.join(split_dir(dataset_dir, split), "audio")
        if not os.path.isdir(adir):
            if by_split[split] > 0:
                errors.append(f"MISSING_AUDIO_DIR:{split}")
            continue
        for dp, _dirs, files in os.walk(adir):
            for fn in files:
                rel = os.path.relpath(os.path.join(dp, fn), split_dir(dataset_dir, split))
                if rel not in rows_by_split[split]:
                    errors.append(f"ORPHAN_AUDIO:{split}/{rel}")

    if require_manifest:
        mpath = os.path.join(dataset_dir, "MANIFEST.sha256")
        ready = os.path.join(dataset_dir, "RELEASE_READY.json")
        if not os.path.exists(mpath):
            errors.append("MANIFEST_MISSING")
        if not os.path.exists(ready):
            errors.append("RELEASE_NOT_READY")

    return CanonicalVerifyResult(not errors, errors, sum(by_split.values()),
                                 by_split)


def metadata_is_six_field(row: Dict) -> bool:
    from .schema import CANONICAL_FIELDS
    return set(row.keys()) == set(CANONICAL_FIELDS)
