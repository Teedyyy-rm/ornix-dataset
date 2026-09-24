"""Calibration harness + metrics (spec §5, Phase 4)."""

from .goldset import GoldClip, load_goldset, check_separation, stratify_summary
from .metrics import (
    classification_metrics,
    false_accept_rate,
    false_reject_rate,
    coverage,
    calibration_report,
)
from .runner import CalibrationResult, run_calibration

__all__ = [
    "GoldClip",
    "load_goldset",
    "check_separation",
    "stratify_summary",
    "classification_metrics",
    "false_accept_rate",
    "false_reject_rate",
    "coverage",
    "calibration_report",
    "CalibrationResult",
    "run_calibration",
]
