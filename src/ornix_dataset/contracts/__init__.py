"""Data contracts: typed schemas + tri-state enums shared across all modules.

Spec §6. Enums distinguish ``NOT_APPLICABLE`` / ``UNKNOWN`` / ``ERROR`` from a
real measurement so we never write ``false``/``0`` when a check did not run.
"""

from .enums import (
    DecisionState,
    IngestStatus,
    MeasurementStatus,
    NoiseLabel,
    RightsStatus,
    Severity,
    RunState,
)
from .source import SourceRecord
from .quality import NoiseEvent, QualityEvidence
from .release import ReleaseRow

__all__ = [
    "DecisionState",
    "IngestStatus",
    "MeasurementStatus",
    "NoiseLabel",
    "RightsStatus",
    "Severity",
    "RunState",
    "SourceRecord",
    "NoiseEvent",
    "QualityEvidence",
    "ReleaseRow",
]
