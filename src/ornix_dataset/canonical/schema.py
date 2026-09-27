"""Canonical training-metadata schema — EXACTLY six fields (spec §3).

The public training contract is deliberately narrow:

    audio:    str   repo-relative path to the canonical WAV
    text:     str   verified transcript, matched to the canonical audio
    file_name: str  same relative path (Hugging Face AudioFolder link column)
    speaker:  str   Ornix speaker id ``spk_<opaque>`` (never a source name)
    duration: float seconds, measured from the verified output WAV
    language: str   unified language code (e.g. ``vi``), never a blind default

Nothing else is allowed in an exported row: provenance (source id/uri/license,
checksums, quality evidence, original filename) lives in the *internal*
audit state, never in the public training metadata (spec §3, §6.4).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

CANONICAL_FIELDS = ("audio", "text", "file_name", "speaker", "duration", "language")

# Current scheme: Ornix_<zero-padded sequence>.wav (7 digits minimum, grows
# past 7 digits past 9,999,999), sharded under audio/<last-2-digits>/.
# Sequential ids shard evenly on the numeric suffix; prefix-sharding would pile
# the first ~10k files into "00".
ORNIX_FILE_RE = re.compile(r"^Ornix_[0-9]{7,}\.wav$")
ORNIX_ID_RE = re.compile(r"^Ornix_[0-9]{7,}$")
# Legacy scheme (files published before the sequential rename): still accepted
# on every read/verify path so existing trees keep validating. New ids are
# never allocated in this form.
ORNIX_FILE_RE_LEGACY = re.compile(r"^ornix_[0-9a-f]{32}\.wav$")
ORNIX_ID_RE_LEGACY = re.compile(r"^ornix_[0-9a-f]{32}$")
SPEAKER_RE = re.compile(r"^spk_[0-9a-f]{32}$")
AUDIO_PATH_RE = re.compile(
    r"^audio/[0-9a-f]{2}/(?:ornix_[0-9a-f]{32}|Ornix_[0-9]{7,})\.wav$")


class SchemaError(ValueError):
    """Raised when a canonical row would violate the 6-field contract."""


def normalize_language(raw: Optional[str]) -> Optional[str]:
    """Return a compact language code from a tag, or ``None`` when unknown.

    ``vi-VN``/``vi_VN`` -> ``vi``; ``en-US`` -> ``en``; ``vi`` -> ``vi``.
    A missing/blank value returns ``None`` — callers MUST NOT default it,
    because "no evidence" is not "Vietnamese" (spec §3).
    """
    if raw is None:
        return None
    tag = str(raw).strip().replace("_", "-")
    if not tag:
        return None
    code = tag.split("-", 1)[0].strip().lower()
    return code or None


def is_ornix_id(value: str) -> bool:
    """Accepts current sequential ids and legacy hex ids (read paths)."""
    v = value or ""
    return bool(ORNIX_ID_RE.match(v) or ORNIX_ID_RE_LEGACY.match(v))


def is_new_ornix_id(value: str) -> bool:
    """True only for the current sequential ``Ornix_<digits>`` scheme."""
    return bool(ORNIX_ID_RE.match(value or ""))


def audio_path_for(ornix_id: str) -> str:
    """Repo-relative WAV path for either id scheme.

    Sequential ids shard on the LAST TWO digits (even distribution);
    legacy hex ids keep their original first-two-hex shard so already
    published files keep resolving to their existing paths.
    """
    if ORNIX_ID_RE.match(ornix_id or ""):
        digits = ornix_id[len("Ornix_"):]
        return f"audio/{digits[-2:]}/{ornix_id}.wav"
    if ORNIX_ID_RE_LEGACY.match(ornix_id or ""):
        shard = ornix_id[len("ornix_"):][:2]
        return f"audio/{shard}/{ornix_id}.wav"
    raise SchemaError(f"not an ornix id: {ornix_id!r}")


@dataclass
class CanonicalRow:
    audio: str
    text: str
    file_name: str
    speaker: str
    duration: float
    language: str

    def to_dict(self) -> Dict[str, Any]:
        return {"audio": self.audio, "text": self.text,
                "file_name": self.file_name, "speaker": self.speaker,
                "duration": self.duration, "language": self.language}

    def validate(self) -> None:
        """Fail-closed structural checks for one public training row."""
        if set(self.__dict__.keys()) != set(CANONICAL_FIELDS):
            raise SchemaError("canonical row must have exactly 6 fields")
        for name in ("audio", "file_name", "text", "speaker", "language"):
            if not isinstance(getattr(self, name), str):
                raise SchemaError(f"{name} must be str")
        if self.audio != self.file_name:
            raise SchemaError("audio must equal file_name")
        if not AUDIO_PATH_RE.match(self.audio):
            raise SchemaError(f"audio path is not canonical: {self.audio!r}")
        if not self.text.strip():
            raise SchemaError("text must be a non-empty verified transcript")
        if not SPEAKER_RE.match(self.speaker):
            raise SchemaError(f"speaker is not spk_<opaque>: {self.speaker!r}")
        if not isinstance(self.duration, float) or not math.isfinite(self.duration):
            raise SchemaError("duration must be a finite float")
        if self.duration <= 0:
            raise SchemaError("duration must be > 0")
        if not self.language or len(self.language) > 10:
            raise SchemaError(f"language is not a compact code: {self.language!r}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanonicalRow":
        extra = set(d) - set(CANONICAL_FIELDS)
        missing = set(CANONICAL_FIELDS) - set(d)
        if extra or missing:
            raise SchemaError(
                f"canonical row keys mismatch (missing={sorted(missing)}, "
                f"extra={sorted(extra)})")
        row = cls(audio=d["audio"], text=d["text"], file_name=d["file_name"],
                  speaker=d["speaker"], duration=float(d["duration"]),
                  language=d["language"])
        row.validate()
        return row
