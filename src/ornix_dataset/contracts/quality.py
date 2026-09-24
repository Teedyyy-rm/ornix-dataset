"""Quality evidence record — one row per candidate clip (spec §6.2).

Never overwrites prior evidence; each analyzer run appends a new row keyed by
(segment_id, analyzer_version, policy_version).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..version import ANALYZER_VERSION, SCHEMA_VERSION
from .enums import DecisionState, MeasurementStatus, NoiseLabel, Severity


@dataclass
class NoiseEvent:
    label: NoiseLabel
    start_s: float
    end_s: float
    overlaps_speech: bool
    severity: Severity
    confidence: float
    score: float
    detector: str
    model_revision: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label.value
        d["severity"] = self.severity.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NoiseEvent":
        d = dict(d)
        d["label"] = NoiseLabel(d["label"])
        d["severity"] = Severity(d["severity"])
        return cls(**d)


@dataclass
class QualityEvidence:
    segment_id: str
    source_sha256: str
    interval_start_sample: int
    interval_end_sample: int
    analysis_sample_rate: int
    vad_intervals: List[List[float]] = field(default_factory=list)
    noise_events: List[NoiseEvent] = field(default_factory=list)
    estimated_snr_db: Optional[float] = None
    snr_method: str = "none"
    snr_status: MeasurementStatus = MeasurementStatus.UNKNOWN
    clipping_ratio: Optional[float] = None
    max_clipped_run: Optional[int] = None
    speech_ratio: Optional[float] = None
    sig: Optional[float] = None
    bak: Optional[float] = None
    ovrl: Optional[float] = None
    quality_model_id: Optional[str] = None
    quality_status: MeasurementStatus = MeasurementStatus.UNKNOWN
    speaker_overlap_intervals: List[List[float]] = field(default_factory=list)
    speaker_status: MeasurementStatus = MeasurementStatus.NOT_APPLICABLE
    transcript_match_status: MeasurementStatus = MeasurementStatus.UNKNOWN
    calibration_domain: str = "UNKNOWN"
    required_checks: List[str] = field(default_factory=list)
    observed_checks: Dict[str, str] = field(default_factory=dict)
    decision: DecisionState = DecisionState.REVIEW
    reason_codes: List[str] = field(default_factory=list)
    policy_version: str = "unset"
    processing_sha256: Optional[str] = None
    analyzer_version: str = ANALYZER_VERSION
    timestamp_utc: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["noise_events"] = [e.to_dict() for e in self.noise_events]
        d["snr_status"] = self.snr_status.value
        d["quality_status"] = self.quality_status.value
        d["speaker_status"] = self.speaker_status.value
        d["transcript_match_status"] = self.transcript_match_status.value
        d["decision"] = self.decision.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QualityEvidence":
        d = dict(d)
        d.pop("schema_version", None)
        d["noise_events"] = [NoiseEvent.from_dict(e) for e in d.get("noise_events", [])]
        for k in ("snr_status", "quality_status", "speaker_status", "transcript_match_status"):
            if k in d:
                d[k] = MeasurementStatus(d[k])
        if "decision" in d:
            d["decision"] = DecisionState(d["decision"])
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in allowed})
