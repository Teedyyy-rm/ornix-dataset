"""Dedup: exact (SHA) + near-duplicate (fingerprint) (spec §6.4, Phase 5, T-011).

Near-duplicate detection uses a cheap, dependency-light audio fingerprint (coarse
log-energy + spectral-centroid trajectory) so the same utterance under two codecs
is grouped and cannot leak across splits.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

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


def _candidate_pairs_lsh(fingerprints: Dict[str, np.ndarray], ids: List[str],
                         n_planes: int = 16, n_tables: int = 8,
                         seed: int = 1234) -> Set[Tuple[str, str]]:
    """Random-hyperplane LSH: bucket unit vectors by sign bits, return only the
    within-bucket pairs. Cosine-similar vectors collide with high probability, so
    exact comparison happens on O(candidates) pairs instead of O(N^2)."""
    dim = int(next(iter(fingerprints.values())).shape[0]) if fingerprints else 0
    rng = np.random.default_rng(seed)
    mat = np.stack([fingerprints[i] for i in ids]).astype(np.float64)  # (N, dim)
    candidates: Set[Tuple[str, str]] = set()
    for _ in range(n_tables):
        planes = rng.standard_normal((dim, n_planes))
        bits = (mat @ planes) >= 0.0                      # (N, n_planes)
        weights = (1 << np.arange(n_planes))              # pack sign bits -> bucket key
        keys = (bits.astype(np.int64) * weights).sum(axis=1)
        buckets: Dict[int, List[int]] = {}
        for row, key in enumerate(keys.tolist()):
            buckets.setdefault(key, []).append(row)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    candidates.add((ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i]))
    return candidates


def near_duplicate_groups(fingerprints: Dict[str, np.ndarray],
                          threshold: float = 0.985,
                          n_planes: int = 16, n_tables: int = 8) -> List[List[str]]:
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

    if len(ids) >= 2:
        candidates = _candidate_pairs_lsh(fingerprints, ids,
                                          n_planes=n_planes, n_tables=n_tables)
        for a, b in candidates:
            if _cosine(fingerprints[a], fingerprints[b]) >= threshold:
                union(a, b)
    clusters: Dict[str, List[str]] = {}
    for i in ids:
        clusters.setdefault(find(i), []).append(i)
    return [sorted(v) for v in clusters.values() if len(v) > 1]
