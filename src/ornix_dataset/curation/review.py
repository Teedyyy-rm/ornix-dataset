"""Review queue builder (spec §5.3, Phase 5).

Collects REVIEW / UNKNOWN / ERROR candidates with evidence pointers for human
adjudication. Reviewer decisions are recorded elsewhere with reviewer id +
timestamp + reason and never overwrite raw metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List

from ..contracts.enums import DecisionState


@dataclass
class ReviewItem:
    segment_id: str
    source_sha256: str
    decision: str
    reason_codes: List[str]
    evidence_ref: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_REVIEWABLE = {DecisionState.REVIEW.value, DecisionState.ERROR.value,
               DecisionState.LICENSE_REVIEW.value}


def build_review_queue(evidences: Iterable[Dict[str, Any]], evidence_path: str) -> List[ReviewItem]:
    queue: List[ReviewItem] = []
    for ev in evidences:
        if ev.get("decision") in _REVIEWABLE:
            queue.append(ReviewItem(
                segment_id=ev.get("segment_id", ""),
                source_sha256=ev.get("source_sha256", ""),
                decision=ev.get("decision", ""),
                reason_codes=list(ev.get("reason_codes", [])),
                evidence_ref=f"{evidence_path}#{ev.get('segment_id','')}",
            ))
    return queue
