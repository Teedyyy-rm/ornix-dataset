"""Curation: deterministic policy engine, segmentation, transcript, dedup, split, review."""

from .policy import PolicyConfig, PolicyDecision, PolicyEngine, load_policy
from .segmentation import SegmentPlan, plan_segments
from .transcript import transcript_match_status
from .dedup import exact_duplicates, near_duplicate_groups
from .split import assign_splits
from .review import ReviewItem, build_review_queue

__all__ = [
    "PolicyConfig",
    "PolicyDecision",
    "PolicyEngine",
    "load_policy",
    "SegmentPlan",
    "plan_segments",
    "transcript_match_status",
    "exact_duplicates",
    "near_duplicate_groups",
    "assign_splits",
    "ReviewItem",
    "build_review_queue",
]
