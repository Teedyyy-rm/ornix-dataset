"""Source manifest record — append-only, one row per source file (spec §6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from ..version import SCHEMA_VERSION
from .enums import IngestStatus, RightsStatus


@dataclass
class SourceRecord:
    # identity / provenance (immutable)
    source_id: str
    source_uri: str
    source_revision: str
    original_file_id: str
    source_sha256: str
    source_bytes: int
    # measured / declared audio metadata
    source_mime: Optional[str] = None
    source_container: Optional[str] = None
    source_codec: Optional[str] = None
    source_lossy: Optional[bool] = None
    source_sample_rate: Optional[int] = None
    source_channels: Optional[int] = None
    source_duration_s: Optional[float] = None
    # source-admission provenance (Level 1 — clean-HQ eligibility)
    source_rate_class: Optional[str] = None
    effective_bandwidth_hz: Optional[float] = None
    low_bandwidth_suspected: bool = False
    # rights (spec §0.2.6 — gated != redistributable)
    source_license: str = "UNKNOWN"
    license_evidence_uri: Optional[str] = None
    rights_owner: Optional[str] = None
    consent_reference: Optional[str] = None
    rights_status: RightsStatus = RightsStatus.UNKNOWN
    redistribution_permitted: bool = False
    commercial_training_permitted: str = "UNKNOWN"  # true|false|UNKNOWN
    attribution_required: bool = True
    # content
    source_split: Optional[str] = None
    source_speaker_ref: Optional[str] = None
    source_transcript: Optional[str] = None
    source_language: Optional[str] = None
    # bookkeeping
    ingest_status: IngestStatus = IngestStatus.INGESTED
    ingestion_timestamp_utc: Optional[str] = None
    staged_path: Optional[str] = None
    reason_codes: list = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rights_status"] = self.rights_status.value
        d["ingest_status"] = self.ingest_status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceRecord":
        d = dict(d)
        d.pop("schema_version", None)
        if "rights_status" in d:
            d["rights_status"] = RightsStatus(d["rights_status"])
        if "ingest_status" in d:
            d["ingest_status"] = IngestStatus(d["ingest_status"])
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in allowed})
