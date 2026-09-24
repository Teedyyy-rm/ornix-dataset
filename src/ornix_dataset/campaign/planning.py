"""Mechanical batch splitter (MD-001).

Deterministic packing of a file list into finite batches. This is NOT the
resource-aware planner — MD-002 replaces the sizing policy while keeping the
stable batch_id contract from models.py.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .models import Batch, batch_id_for, job_id_for


def plan_batches(repo_id: str, pinned_sha: str, job_id: Optional[str],
                 files: Sequence[Tuple[str, Optional[int]]],
                 max_files: int = 500, max_bytes: int = 10 * 1024**3,
                 checkpoint_prefix: str = "checkpoints",
                 start_index: int = 0) -> List[Batch]:
    """Pack ``(path, size_or_None)`` entries into batches.

    Pure + deterministic: identical inputs always yield identical batch_ids.
    A single file larger than max_bytes is never split — it gets its own batch
    with an ``oversize`` flag so MD-002 can route it to streaming/BLOCKED.
    """
    if max_files < 1 or max_bytes < 1:
        raise ValueError("max_files and max_bytes must be >= 1")
    jid = job_id or job_id_for(repo_id, pinned_sha)
    groups: List[List[Tuple[str, Optional[int]]]] = []
    cur: List[Tuple[str, Optional[int]]] = []
    cur_bytes = 0
    for path, size in files:
        known = size if isinstance(size, int) and size >= 0 else None
        if cur and (len(cur) >= max_files
                    or (known is not None and cur_bytes + known > max_bytes)):
            groups.append(cur)
            cur, cur_bytes = [], 0
        cur.append((path, size))
        if known is not None:
            cur_bytes += known
        # a lone oversize file closes its own batch immediately
        if len(cur) == 1 and known is not None and known > max_bytes:
            groups.append(cur)
            cur, cur_bytes = [], 0
    if cur:
        groups.append(cur)
    out: List[Batch] = []
    for i, g in enumerate(groups):
        idx = start_index + i
        paths = [p for p, _ in g]
        total = sum(s for _, s in g if isinstance(s, int) and s >= 0)
        unknown = sum(1 for _, s in g if not (isinstance(s, int) and s >= 0))
        bid = batch_id_for(repo_id, pinned_sha, idx, paths)
        oversize = len(g) == 1 and (g[0][1] or 0) > max_bytes
        out.append(Batch(
            batch_id=bid, job_id=jid, index=idx, files=paths,
            total_bytes=total, n_unknown_bytes=unknown,
            checkpoint_rel=f"{checkpoint_prefix}/{jid}/{bid}.jsonl",
            result={"oversize": bool(oversize)} if oversize else {}))
    return out
