"""Retention + takedown / rights-revocation response (spec §8, §11).

Removal never deletes source provenance silently: a takedown quarantines the
affected records and produces a NEW release without them (prior releases are not
rewritten). Rights revocation is handled the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RetentionPolicy:
    keep_raw_days: int = 0          # 0 => keep raw indefinitely under access control
    keep_rejected_days: int = 90
    keep_evidence_days: int = 3650  # audit trail retained long-term
    quarantine_on_revocation: bool = True


def takedown_plan(affected_source_ids: List[str], current_release_id: str,
                  next_release_id: str) -> Dict[str, Any]:
    """Produce a non-destructive takedown plan for a rights revocation / removal."""
    return {
        "action": "QUARANTINE_AND_REISSUE",
        "affected_source_ids": sorted(set(affected_source_ids)),
        "from_release": current_release_id,
        "to_release": next_release_id,
        "steps": [
            "Mark affected sources FORBIDDEN in the rights matrix (append-only).",
            "Exclude affected records from the new accepted manifest.",
            "Build a new release id; do NOT rewrite the previous release.",
            "If legally required, request removal of prior HF revision via a new "
            "commit; document the reason and keep an audit record.",
        ],
        "invariant": "source provenance and audit history are preserved.",
    }
