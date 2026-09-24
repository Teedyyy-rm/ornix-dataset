"""Dedup: exact (SHA) + near-duplicate (fingerprint) (spec §6.4, Phase 5, T-011).

Near-duplicate detection uses a cheap, dependency-light audio fingerprint (coarse
log-energy + spectral-centroid trajectory) so the same utterance under two codecs
is grouped and cannot leak across splits.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def exact_duplicates(sha_by_id: Dict[str, str]) -> List[List[str]]:
    groups: Dict[str, List[str]] = {}
    for _id, sha in sha_by_id.items():
        groups.setdefault(sha, []).append(_id)
    return [sorted(ids) for ids in groups.values() if len(ids) > 1]


def fingerprint(mono: np.ndarray, sr: int, n_bins: int = 32) -> np.ndarray:
    """Fixed-length normalized fingerprint from log-energy envelope."""
    if mono.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    hop = max(1, mono.size // n_bins)
    n = mono.size // hop
    frames = mono[: n * hop].reshape(n, hop)
    env = np.log1p(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-9))
    # resample envelope to exactly n_bins
    idx = np.linspace(0, len(env) - 1, n_bins) if len(env) > 1 else np.zeros(n_bins, int)
    fp = env[idx.astype(int)] if len(env) else np.zeros(n_bins)
    norm = np.linalg.norm(fp)
    return (fp / norm).astype(np.float32) if norm > 0 else fp.astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # inputs are unit-normalized


def near_duplicate_groups(fingerprints: Dict[str, np.ndarray],
                          threshold: float = 0.985) -> List[List[str]]:
    ids = list(fingerprints.keys())
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if _cosine(fingerprints[ids[i]], fingerprints[ids[j]]) >= threshold:
                union(ids[i], ids[j])
    clusters: Dict[str, List[str]] = {}
    for i in ids:
        clusters.setdefault(find(i), []).append(i)
    return [sorted(v) for v in clusters.values() if len(v) > 1]
