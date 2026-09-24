"""Leakage-safe train/val/test split (spec §6.4, Phase 5, T-011).

Split is by *group key* (speaker / recording-family / source-group) so no group
straddles splits. Assignment is deterministic from a saved seed and the group
key hash — never from scan order.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple


def _bucket(group_key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def assign_splits(group_keys: Dict[str, str], ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                  seed: int = 20260924) -> Dict[str, str]:
    """Map item_id -> split. All items sharing a group_key get the same split."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    train_r, val_r, _ = ratios
    group_split: Dict[str, str] = {}
    for group in sorted(set(group_keys.values())):
        b = _bucket(group, seed)
        if b < train_r:
            group_split[group] = "train"
        elif b < train_r + val_r:
            group_split[group] = "validation"
        else:
            group_split[group] = "test"
    return {item: group_split[g] for item, g in group_keys.items()}


def split_leakage(group_keys: Dict[str, str], assignment: Dict[str, str]) -> List[str]:
    """Return group keys that appear in more than one split (must be empty)."""
    seen: Dict[str, str] = {}
    bad: List[str] = []
    for item, group in group_keys.items():
        split = assignment.get(item)
        if group in seen and seen[group] != split:
            bad.append(group)
        else:
            seen[group] = split
    return sorted(set(bad))
