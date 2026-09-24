"""Release row — ACCEPT-only, published schema (spec §6.3)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class ReleaseRow:
    audio_id: str
    audio: str  # repo-relative path INSIDE the shard/repo (never a build-machine path)
    language: str
    speaker_id: str
    transcript: str
    sample_rate: int
    channels: int
    encoding: str
    duration_s: float
    source_id: str
    source_sha256: str
    audio_sha256: str
    segment_start_sample_source: int
    segment_end_sample_source: int
    rights_record_id: str
    quality_evidence_id: str
    quality_policy_version: str
    quality_gate: str
    split: str
    release_id: str
    # rights carried onto every published row (spec §0.2.6 — gated != redistributable)
    source_license: str = "UNKNOWN"
    rights_status: str = "UNKNOWN"
    redistribution_permitted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReleaseRow":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def validate(self, release_target: str = "train_only") -> None:
        """Fail-closed structural invariants for a publishable row (spec §6.3).

        ``release_target='public'`` additionally requires the row carry explicit
        redistribution rights; ``TRAIN_ONLY`` (or any non-approved status) can
        never appear in a public release, regardless of operator memory.
        """
        if self.quality_gate != "ACCEPT":
            raise ValueError(f"release row {self.audio_id} is not ACCEPT")
        if self.sample_rate != 24000 or self.channels != 1 or self.encoding != "PCM_S16LE":
            raise ValueError(f"release row {self.audio_id} is not canonical 24k mono PCM16")
        if self.audio.startswith("/") or ":" in self.audio.split("/")[0]:
            raise ValueError(f"release row {self.audio_id} audio path is absolute/local")
        if len(self.audio_sha256) != 64 or len(self.source_sha256) != 64:
            raise ValueError(f"release row {self.audio_id} has malformed sha256")
        if self.duration_s <= 0 or self.duration_s > 12.0:
            raise ValueError(f"release row {self.audio_id} duration {self.duration_s} out of policy")
        if release_target == "public":
            if self.rights_status != "REDISTRIBUTION_APPROVED" or not self.redistribution_permitted:
                raise ValueError(
                    f"release row {self.audio_id} not redistributable for public target "
                    f"(rights_status={self.rights_status}, "
                    f"redistribution_permitted={self.redistribution_permitted})")
