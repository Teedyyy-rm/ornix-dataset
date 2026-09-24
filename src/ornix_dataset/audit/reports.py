"""Funnel / coverage report builder (spec §5.2).

Reports REVIEW / UNKNOWN in their own buckets — never hidden from the funnel
denominator.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable

from ..contracts.enums import DecisionState


def build_funnel_report(evidences: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    decisions = Counter()
    reason_codes = Counter()
    total = 0
    accepted_duration_samples = 0
    for ev in evidences:
        total += 1
        decisions[ev.get("decision", "UNKNOWN")] += 1
        for rc in ev.get("reason_codes", []):
            reason_codes[rc] += 1
        if ev.get("decision") == DecisionState.ACCEPT.value:
            accepted_duration_samples += max(
                0, int(ev.get("interval_end_sample", 0)) - int(ev.get("interval_start_sample", 0))
            )
    accept = decisions.get(DecisionState.ACCEPT.value, 0)
    coverage = (accept / total) if total else 0.0
    return {
        "total_candidates": total,
        "by_decision": dict(decisions),
        "accept_coverage": round(coverage, 6),
        "reason_codes": dict(reason_codes.most_common()),
        "note": "REVIEW/UNKNOWN/ERROR reported explicitly; not folded into ACCEPT.",
    }
