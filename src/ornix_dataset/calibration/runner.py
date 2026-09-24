"""Calibration runner (spec §5.1–5.3).

Runs the *real* QC pipeline over the human-labeled gold set and measures
false-accept / false-reject / coverage against ground truth. It refuses to run
if calibration and heldout share any source/speaker/recording-family (tuning on
test). It reports metrics only — it NEVER writes thresholds. Signing a threshold
from these numbers is an explicit, human, operator-gated step (see docs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..audit.events import AuditLog
from ..contracts.enums import DecisionState
from ..contracts.source import SourceRecord
from ..pipeline import OrnixPipeline, RunPaths
from .goldset import GoldClip, check_separation, load_goldset, stratify_summary
from .metrics import calibration_report


@dataclass
class CalibrationResult:
    ok: bool
    split: str
    report: Dict[str, Any] = field(default_factory=dict)
    stratification: Dict[str, Any] = field(default_factory=dict)
    per_clip: List[Dict[str, Any]] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)


# ranked worst-first so a clip's terminal label is its most severe segment outcome
_SEVERITY = {
    DecisionState.REJECT_TECH.value: 4, DecisionState.REJECT.value: 3,
    DecisionState.ERROR.value: 3, DecisionState.REVIEW.value: 2,
    DecisionState.LICENSE_REVIEW.value: 2, DecisionState.ACCEPT.value: 1,
}


def _terminal_decision(decisions: List[str]) -> str:
    if not decisions:
        return DecisionState.REJECT_TECH.value  # produced nothing analyzable
    return max(decisions, key=lambda d: _SEVERITY.get(d, 0))


def _clip_record(clip: GoldClip) -> SourceRecord:
    from ..util.hashing import sha256_file

    sha = sha256_file(clip.path)
    return SourceRecord(
        source_id="GOLD_" + clip.clip_id, source_uri="file://" + os.path.abspath(clip.path),
        source_revision="goldset", original_file_id=os.path.basename(clip.path),
        source_sha256=sha, source_bytes=os.path.getsize(clip.path),
        source_language="vi", staged_path=clip.path)


def run_calibration(goldset_path: str, pipeline: OrnixPipeline, workdir: str,
                    split: str = "calibration",
                    run_id: str = "calibration") -> CalibrationResult:
    clips = load_goldset(goldset_path)
    overlap = check_separation(clips)
    blockers: List[str] = []
    for key, shared in overlap.items():
        if shared:
            blockers.append(f"GOLDSET_LEAKAGE:{key}:{','.join(shared)}")
    if blockers:  # fail-closed: never calibrate on a leaking gold set
        return CalibrationResult(False, split, blockers=blockers,
                                 stratification=stratify_summary(clips))

    subset = [c for c in clips if c.split == split]
    paths = RunPaths.create(workdir, run_id)
    audit = AuditLog(paths.audit, run_id=run_id)
    gold_is_noisy: List[bool] = []
    decisions: List[str] = []
    per_clip: List[Dict[str, Any]] = []
    for clip in subset:
        rec = _clip_record(clip)
        res = pipeline.analyze_source(rec, paths, audit)
        seg_decisions = [e.decision.value for e in res.evidences if e.decision is not None]
        terminal = _terminal_decision(seg_decisions)
        gold_is_noisy.append(not clip.is_clean)
        decisions.append(terminal)
        per_clip.append({"clip_id": clip.clip_id, "is_clean": clip.is_clean,
                         "terminal_decision": terminal, "segments": seg_decisions})

    report = calibration_report(gold_is_noisy, decisions)
    return CalibrationResult(True, split, report=report,
                             stratification=stratify_summary(clips), per_clip=per_clip)
