"""Windowing / interval algebra for detectors (spec §3, Phase 3).

Chunk analysis audio into overlapping windows and map results back to source
coordinates. Union + controlled dilation avoids missing short events, per spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WindowPlan:
    window_s: float
    hop_s: float
    windows: List[Tuple[float, float]]


def plan_windows(duration_s: float, window_s: float = 1.5, overlap: float = 0.5) -> WindowPlan:
    if window_s <= 0 or not (0.0 <= overlap < 1.0):
        raise ValueError("invalid window/overlap")
    hop = window_s * (1.0 - overlap)
    windows: List[Tuple[float, float]] = []
    t = 0.0
    if duration_s <= window_s:
        return WindowPlan(window_s, hop, [(0.0, duration_s)])
    while t < duration_s:
        end = min(t + window_s, duration_s)
        windows.append((round(t, 6), round(end, 6)))
        if end >= duration_s:
            break
        t += hop
    return WindowPlan(window_s, hop, windows)


def union_intervals(intervals: List[List[float]]) -> List[List[float]]:
    if not intervals:
        return []
    ivs = sorted([list(iv) for iv in intervals], key=lambda x: x[0])
    merged = [ivs[0]]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def dilate_intervals(intervals: List[List[float]], pad_s: float, max_s: float) -> List[List[float]]:
    out = [[max(0.0, s - pad_s), min(max_s, e + pad_s)] for s, e in intervals]
    return union_intervals(out)


def intersect_intervals(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    i = j = 0
    a, b = union_intervals(a), union_intervals(b)
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append([lo, hi])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def total_duration(intervals: List[List[float]]) -> float:
    return float(sum(e - s for s, e in union_intervals(intervals)))
