"""Enumerations for the Ornix data contracts (spec §0.1, §3, §5.3, §8)."""

from __future__ import annotations

from enum import Enum


class MeasurementStatus(str, Enum):
    """Why a numeric field may be absent — never conflate with a real value."""

    OK = "OK"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class RightsStatus(str, Enum):
    """Redistribution eligibility of a source (spec §0.1 Thu thập)."""

    REDISTRIBUTION_APPROVED = "REDISTRIBUTION_APPROVED"
    TRAIN_ONLY = "TRAIN_ONLY"
    LICENSE_REVIEW = "LICENSE_REVIEW"
    FORBIDDEN = "FORBIDDEN"
    UNKNOWN = "UNKNOWN"


class IngestStatus(str, Enum):
    INGESTED = "INGESTED"
    QUARANTINE = "QUARANTINE"
    ERROR = "ERROR"
    SKIPPED_DUPLICATE = "SKIPPED_DUPLICATE"


class DecisionState(str, Enum):
    """Terminal QC decision (spec §0.1 Chất lượng)."""

    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
    REJECT_TECH = "REJECT_TECH"
    ERROR = "ERROR"
    LICENSE_REVIEW = "LICENSE_REVIEW"


class SourceRateClass(str, Enum):
    """Source sample-rate class for clean-HQ admission (Level 1, spec §Repair).

    Assigned from the *measured* sample rate (never the header). Thresholds are
    project-policy defaults held in ``configs/audio_profile.yaml``.
    """

    NATIVE_OR_HIGHER = "NATIVE_OR_HIGHER"        # sr >= canonical (24k)
    NEAR_TARGET_UPSAMPLE = "NEAR_TARGET_UPSAMPLE"  # conditional_min <= sr < canonical
    LOW_BANDWIDTH_SOURCE = "LOW_BANDWIDTH_SOURCE"  # low_bw_min <= sr < conditional_min
    NARROWBAND_SOURCE = "NARROWBAND_SOURCE"        # sr < low_bw_min


class CanonicalizationAction(str, Enum):
    """What the canonicalization stage does (or refuses to do) for a source."""

    IDENTITY = "IDENTITY"
    DOWNSAMPLE = "DOWNSAMPLE"
    UPSAMPLE_NEAR_TARGET = "UPSAMPLE_NEAR_TARGET"
    REJECT_LOW_BANDWIDTH = "REJECT_LOW_BANDWIDTH"
    REJECT_NARROWBAND = "REJECT_NARROWBAND"
    ERROR = "ERROR"


class Severity(str, Enum):
    """Noise impact severity (spec §3)."""

    N0 = "N0"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    NX = "NX"


class NoiseLabel(str, Enum):
    """Multi-label noise taxonomy (spec §3)."""

    MUSIC_BACKGROUND = "MUSIC_BACKGROUND"
    MUSIC_ONLY = "MUSIC_ONLY"
    HISS_STATIC = "HISS_STATIC"
    HUM_BUZZ = "HUM_BUZZ"
    TRANSIENT_CRACKLE = "TRANSIENT_CRACKLE"
    ENV_CONTINUOUS = "ENV_CONTINUOUS"
    ENV_TRANSIENT = "ENV_TRANSIENT"
    WIND_MIC = "WIND_MIC"
    INTERFERING_SPEECH = "INTERFERING_SPEECH"
    REVERB_ECHO = "REVERB_ECHO"
    CLIPPING_DISTORTION = "CLIPPING_DISTORTION"
    CODEC_PROCESSING_ARTIFACT = "CODEC_PROCESSING_ARTIFACT"
    NO_SPEECH = "NO_SPEECH"
    TRUNCATED = "TRUNCATED"
    TRANSCRIPT_MISMATCH = "TRANSCRIPT_MISMATCH"
    UNKNOWN_EVENT = "UNKNOWN_EVENT"


class RunState(str, Enum):
    """Long-running per-record state machine (spec §8)."""

    DISCOVERED = "DISCOVERED"
    RIGHTS_VERIFIED = "RIGHTS_VERIFIED"
    INGESTED = "INGESTED"
    TECH_VERIFIED = "TECH_VERIFIED"
    ANALYZED = "ANALYZED"
    REVIEWED = "REVIEWED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    PACKAGED = "PACKAGED"
    RELEASE_VERIFIED = "RELEASE_VERIFIED"
    PUBLISHED_VERIFIED = "PUBLISHED_VERIFIED"
    BLOCKED = "BLOCKED"
