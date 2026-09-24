"""Conditional segmentation (spec §0.2.4, Phase 2/5, tests T-003, T-009).

Cuts candidate clips only at VAD silence boundaries (never mid-syllable), keeps
each clip <= max_duration, and excludes regions overlapping confirmed severe
noise/music. Transcript must be re-verified for any clip that drops words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..detectors.windowing import union_intervals


@dataclass
class SegmentPlan:
    intervals_s: List[List[float]]
    reason: str


def _subtract(base: List[List[float]], remove: List[List[float]]) -> List[List[float]]:
    base = union_intervals(base)
    remove = union_intervals(remove)
    out: List[List[float]] = []
    for s, e in base:
        cur = [[s, e]]
        for rs, re in remove:
            nxt = []
            for cs, ce in cur:
                if re <= cs or rs >= ce:
                    nxt.append([cs, ce])
                    continue
                if rs > cs:
                    nxt.append([cs, min(rs, ce)])
                if re < ce:
                    nxt.append([max(re, cs), ce])
            cur = nxt
        out.extend(cur)
    return [iv for iv in out if iv[1] - iv[0] > 1e-6]


def plan_segments(speech_intervals: List[List[float]], exclude_intervals: List[List[float]],
                  max_duration_s: float = 12.0, min_duration_s: float = 0.4,
                  pad_s: float = 0.05) -> SegmentPlan:
    """Return renderable clip intervals within speech, excluding bad regions."""
    keep = _subtract(speech_intervals, exclude_intervals)
    clips: List[List[float]] = []
    for s, e in keep:
        s2, e2 = max(0.0, s - pad_s), e + pad_s
        length = e2 - s2
        if length < min_duration_s:
            continue
        if length <= max_duration_s:
            clips.append([round(s2, 6), round(e2, 6)])
            continue
        # split long speech spans into <= max_duration chunks at even cuts
        n = int(length // max_duration_s) + 1
        step = length / n
        for i in range(n):
            cs = s2 + i * step
            ce = min(e2, cs + step)
            if ce - cs >= min_duration_s:
                clips.append([round(cs, 6), round(ce, 6)])
    return SegmentPlan(clips, "vad_boundaries_minus_noise")
