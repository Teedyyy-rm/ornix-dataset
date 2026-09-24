"""Gold-set schema + separation guard (spec §5.1).

The gold set is split into ``calibration`` and ``heldout``; no source / speaker /
recording-family may appear in both (prevents tuning-on-test). Each clip carries
human labels: noise event types, intervals and impact severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..util.jsonl import read_jsonl


@dataclass
class GoldClip:
    clip_id: str
    path: str
    split: str  # "calibration" | "heldout"
    source_id: str
    speaker_id: str
    recording_family: str
    is_clean: bool
    labels: List[str] = field(default_factory=list)  # NoiseLabel values
    severity: str = "NX"
    domain: str = "UNKNOWN"  # studio | podcast | phone | ...

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GoldClip":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in allowed})


def load_goldset(path: str) -> List[GoldClip]:
    return [GoldClip.from_dict(d) for d in read_jsonl(path)]


def check_separation(clips: List[GoldClip]) -> Dict[str, List[str]]:
    """Return overlapping keys between calibration and heldout (must be empty)."""
    cal = {"source": set(), "speaker": set(), "family": set()}
    hel = {"source": set(), "speaker": set(), "family": set()}
    for c in clips:
        bucket = cal if c.split == "calibration" else hel
        bucket["source"].add(c.source_id)
        bucket["speaker"].add(c.speaker_id)
        bucket["family"].add(c.recording_family)
    return {
        "source": sorted(cal["source"] & hel["source"]),
        "speaker": sorted(cal["speaker"] & hel["speaker"]),
        "family": sorted(cal["family"] & hel["family"]),
    }


def stratify_summary(clips: List[GoldClip]) -> Dict[str, Any]:
    from collections import Counter

    by_split = Counter(c.split for c in clips)
    by_domain = Counter(c.domain for c in clips)
    by_label = Counter(l for c in clips for l in c.labels)
    clean = sum(1 for c in clips if c.is_clean)
    return {
        "n_clips": len(clips),
        "by_split": dict(by_split),
        "by_domain": dict(by_domain),
        "by_label": dict(by_label),
        "clean": clean,
        "noisy": len(clips) - clean,
    }
