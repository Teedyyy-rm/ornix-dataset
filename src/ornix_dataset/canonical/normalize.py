"""Normalization: verified records -> canonical Ornix Dataset (spec §4–§7).

Only *verified* records are accepted. A sample enters the public dataset when
all of the following hold, otherwise it is excluded (never silently accepted):

- the input WAV is a canonical 24 kHz mono PCM16 file that re-verifies;
- the transcript is present **and** flagged verified;
- the language is verified (no blind default);
- the rights are redistributable (for the public destination);
- its content is not a conflicting rewrite of an existing output path.

Identity and speaker normalization are delegated to the locked, persistent
``CanonicalState`` so retry/resume and multi-worker ordering cannot duplicate
ids or rows. State is committed before the caller may publish.
"""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..dsp.render import verify_canonical_wav
from ..util.hashing import sha256_file
from . import card, layout
from .identity import CanonicalState, internal_sample_key, open_state
from .schema import CanonicalRow, SchemaError, audio_path_for, normalize_language

_ALLOWED_RIGHTS = {"REDISTRIBUTION_APPROVED", "TRAIN_ONLY"}
_PUBLIC_ONLY = {"REDISTRIBUTION_APPROVED"}


@dataclass
class SampleInput:
    """One verified candidate for the canonical dataset (already QC-accepted)."""
    wav_path: str
    text: str
    language: Optional[str]
    speaker_ref: Optional[str]
    source_scope: str
    source_id: str
    source_sha256: str
    seg_start: int
    seg_end: int
    split: str
    rights_status: str
    redistribution_permitted: bool
    transcript_verified: bool = False
    language_verified: bool = False
    source_uri: str = ""
    source_revision: str = ""
    original_file_id: str = ""
    audio_sha256: str = ""
    quality_evidence_id: str = ""
    quality_policy_version: str = ""
    source_license: str = "UNKNOWN"


def _blocker(sample: SampleInput, reason: str) -> Dict[str, Any]:
    return {"source_id": sample.source_id,
            "original_file_id": sample.original_file_id or "",
            "reason": reason}


def _validate_sample(sample: SampleInput,
                     require_redistributable: bool) -> Optional[str]:
    """Return a blocker reason, or ``None`` when the sample may be exported."""
    allowed = _PUBLIC_ONLY if require_redistributable else _ALLOWED_RIGHTS
    if sample.rights_status not in allowed:
        return f"RIGHT_NOT_PERMITTED:{sample.rights_status}"
    if require_redistributable and not sample.redistribution_permitted:
        return "REDISTRIBUTION_FLAG_FALSE"
    if not sample.transcript_verified or not (sample.text or "").strip():
        return "TRANSCRIPT_UNVERIFIED"
    if not sample.language_verified:
        return "LANGUAGE_UNVERIFIED"
    if normalize_language(sample.language) is None:
        return "LANGUAGE_UNVERIFIED"
    return None


def _place_audio(sample: SampleInput, dst: str) -> Optional[str]:
    """Copy/verify the output WAV. Returns a blocker reason or ``None``.

    If the destination already exists, content identity is checked first: an
    identical file is reused; a *different* file is a hard conflict (never
    overwrite audio that another record already owns).
    """
    if os.path.exists(dst):
        if sha256_file(dst) == sha256_file(sample.wav_path):
            return None
        return "CONTENT_IDENTITY_CONFLICT"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    shutil.copyfile(sample.wav_path, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, dst)
    return None


def normalize_samples(samples: List[SampleInput], dataset_dir: str,
                      state_dir: str,
                      require_redistributable: bool = True,
                      uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
                      changelog: str = "",
                      ) -> Dict[str, Any]:
    """Normalize verified samples into ``dataset_dir`` (incremental, idempotent).

    Returns a report; excluded samples carry an explicit reason. The public tree
    is finalized (card + checksum manifest + READY marker) only when the run has
    at least one exportable row.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    rows_by_split: Dict[str, List[CanonicalRow]] = {}
    blocked: List[Dict[str, Any]] = []
    n_reused_files = 0
    licenses: set = set()
    statuses: set = set()
    with open_state(state_dir) as state:
        for sample in samples:
            reason = _validate_sample(sample, require_redistributable)
            if reason is None and not os.path.exists(sample.wav_path):
                reason = "WAV_MISSING"
            if reason is None:
                ok, why = verify_canonical_wav(sample.wav_path)
                if not ok:
                    reason = "WAV_NOT_CANONICAL:" + ",".join(why)
            if reason is not None:
                blocked.append(_blocker(sample, reason))
                continue

            actual_sha = sha256_file(sample.wav_path)
            if sample.audio_sha256 and sample.audio_sha256 != actual_sha:
                blocked.append(_blocker(sample, "AUDIO_SHA_MISMATCH"))
                continue

            import wave
            with wave.open(sample.wav_path, "rb") as wf:
                duration = round(wf.getnframes() / float(wf.getframerate()), 6)
            if not duration > 0:
                blocked.append(_blocker(sample, "DURATION_NONPOSITIVE"))
                continue

            key = internal_sample_key(sample.source_id, sample.source_sha256,
                                      sample.seg_start, sample.seg_end,
                                      sample.source_revision)
            ornix_id = state.ornix_id_for(key, actual_sha, uuid_factory)
            speaker = state.speaker_for(sample.source_scope, sample.speaker_ref,
                                        key, uuid_factory)
            row = CanonicalRow(audio=audio_path_for(ornix_id),
                               file_name=audio_path_for(ornix_id),
                               text=sample.text,
                               speaker=speaker, duration=duration,
                               language=normalize_language(sample.language))
            try:
                row.validate()
            except SchemaError as e:
                blocked.append(_blocker(sample, f"ROW_INVALID:{e}"))
                continue

            dst = layout.audio_abs_path(dataset_dir, sample.split, row)
            existed = os.path.exists(dst)
            conflict = _place_audio(sample, dst)
            if conflict:
                blocked.append(_blocker(sample, conflict))
                continue
            if existed:
                n_reused_files += 1

            state.set_provenance(ornix_id, {
                "internal_key": key, "source_id": sample.source_id,
                "source_uri": sample.source_uri, "source_revision": sample.source_revision,
                "original_file_id": sample.original_file_id,
                "source_sha256": sample.source_sha256, "audio_sha256": actual_sha,
                "quality_evidence_id": sample.quality_evidence_id,
                "quality_policy_version": sample.quality_policy_version,
                "rights_status": sample.rights_status,
                "redistribution_permitted": sample.redistribution_permitted,
                "source_license": sample.source_license,
                "transcript": sample.text, "language": row.language,
                "speaker": speaker, "split": sample.split,
                "source_speaker_ref": sample.speaker_ref})
            rows_by_split.setdefault(sample.split, []).append(row)
            licenses.add(sample.source_license)
            statuses.add(sample.rights_status)

        merged_summary = {}
        for split, rows in rows_by_split.items():
            merged_summary[split] = layout.upsert_rows(dataset_dir, split, rows)
        state.commit()

    stats = layout.split_stats(dataset_dir)
    rights = {"licenses": sorted(l for l in licenses if l),
              "rights_status": sorted(statuses),
              "require_redistributable": require_redistributable}
    if stats["n_rows"] > 0:
        card.write_card(dataset_dir, stats, rights, changelog)
        layout.write_manifest_sha(dataset_dir)
        layout.write_ready(dataset_dir, stats, rights)
    return {"ok": True, "dataset_dir": dataset_dir, "state_dir": state_dir,
            "n_rows": stats["n_rows"], "by_split": stats["by_split"],
            "n_blocked": len(blocked), "blocked": blocked,
            "n_reused_files": n_reused_files, "merged": merged_summary,
            "stats": stats}
