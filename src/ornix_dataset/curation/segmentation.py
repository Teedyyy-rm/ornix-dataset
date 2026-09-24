"""Conditional segmentation (spec §0.2.4, Phase 2/5, tests T-003, T-009).

Cuts candidate clips only at VAD silence boundaries (never mid-syllable), keeps
each clip <= max_duration, and excludes regions overlapping confirmed severe
noise/music. Transcript must be re-verified for any clip that drops words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

import numpy as np

from ..detectors.windowing import union_intervals


@dataclass
class SegmentPlan:
    intervals_s: List[List[float]]
    reason: str
    uncertain_indices: Set[int] = field(default_factory=set)


def low_energy_points(mono: np.ndarray, sr: int, frame_s: float = 0.03,
                      percentile: float = 20.0) -> List[float]:
    """Timestamps (s) of local RMS-energy minima — natural silence-boundary cuts.

    A long speech span is cut only at these low-energy troughs so we never slice
    through a syllable. Returns an empty list for empty/too-short input; callers
    must treat "no silence point" as *boundary uncertain*, not as a clean cut.
    """
    if mono.size == 0 or sr <= 0:
        return []
    hop = max(1, int(round(frame_s * sr)))
    n = mono.size // hop
    if n < 3:
        return []
    frames = mono[: n * hop].reshape(n, hop).astype(np.float64)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    thresh = np.percentile(rms, percentile)
    points: List[float] = []
    for i in range(1, n - 1):
        if rms[i] <= thresh and rms[i] <= rms[i - 1] and rms[i] <= rms[i + 1]:
            points.append(round((i + 0.5) * hop / sr, 6))
    return points


def _nearest_point(t: float, points: List[float], tol: float) -> float | None:
    if not points:
        return None
    best = min(points, key=lambda p: abs(p - t))
    return best if abs(best - t) <= tol else None



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
                  pad_s: float = 0.05, silence_points: List[float] | None = None,
                  snap_tol_s: float = 0.35) -> SegmentPlan:
    """Return renderable clip intervals within speech, excluding bad regions.

    Long speech spans are split at the nearest low-energy silence trough
    (``silence_points``) instead of arithmetic midpoints. When no trough lies
    within ``snap_tol_s`` of a required cut, the clip is still emitted but its
    index is recorded in ``uncertain_indices`` so the transcript is not trusted.
    """
    points = sorted(silence_points or [])
    keep = _subtract(speech_intervals, exclude_intervals)
    clips: List[List[float]] = []
    uncertain: Set[int] = set()
    for s, e in keep:
        s2, e2 = max(0.0, s - pad_s), e + pad_s
        length = e2 - s2
        if length < min_duration_s:
            continue
        if length <= max_duration_s:
            clips.append([round(s2, 6), round(e2, 6)])
            continue
        # split a long span at silence troughs, never mid-syllable
        n = int(length // max_duration_s) + 1
        step = length / n
        cs = s2
        for i in range(n):
            target = e2 if i == n - 1 else cs + step
            if i < n - 1:
                snapped = _nearest_point(target, points, snap_tol_s)
                if snapped is not None and snapped > cs + min_duration_s:
                    ce = snapped
                else:
                    ce = target
                    uncertain.add(len(clips))  # arithmetic cut => re-verify transcript
            else:
                ce = e2
            ce = min(e2, ce)
            if ce - cs >= min_duration_s:
                clips.append([round(cs, 6), round(ce, 6)])
            cs = ce
    return SegmentPlan(clips, "vad_boundaries_minus_noise", uncertain)
