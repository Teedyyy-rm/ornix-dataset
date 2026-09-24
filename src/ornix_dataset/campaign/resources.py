"""Resource-aware batch planner (MD-002).

Contract (plan §3/MD-002, review contract): **the planner holds reservations,
the downloader only enforces.** A batch carries its worst-case footprint in
``Batch.reservation``; no worker may start a batch without acquiring that
reservation from the ledger first.

Footprint model: a batch with B accountable bytes can transiently hold
``B * factor`` bytes in each stage at once (HF cache blob + verified staging
copy + QC/canonical expansion + upload package). Accountable bytes = known
bytes + ``unknown_estimate`` per unknown-size file. MD-007 benchmarks tune
the factors; until then they stay conservative.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import Batch, BatchStatus, batch_id_for, job_id_for

STAGES = ("cache", "staging", "qc", "upload")


@dataclass
class StageProfile:
    """Per-stage byte multipliers + unknown-size policy (all tunable)."""
    cache_factor: float = 1.0
    staging_factor: float = 1.0
    qc_factor: float = 2.0      # canonical WAV expansion, conservative
    upload_factor: float = 1.0
    unknown_estimate_bytes: int = 512 * 1024**2   # per unknown-size file
    max_unknown_per_batch: int = 50

    def footprint(self, accountable_bytes: int) -> Dict[str, int]:
        return {
            "cache": int(accountable_bytes * self.cache_factor),
            "staging": int(accountable_bytes * self.staging_factor),
            "qc": int(accountable_bytes * self.qc_factor),
            "upload": int(accountable_bytes * self.upload_factor),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Budget:
    """Workspace budget. Stage caps bound coexisting bytes per stage."""
    workspace_max_bytes: int
    min_free_bytes: int                      # system headroom, never touched
    stage_caps: Dict[str, int] = field(default_factory=dict)
    max_files_per_batch: int = 500
    max_batch_bytes: int = 10 * 1024**3      # accountable bytes per batch
    high_watermark_bytes: int = 0            # stop admitting below this free
    low_watermark_bytes: int = 0             # resume admitting at/above this

    def __post_init__(self) -> None:
        if self.workspace_max_bytes < 1 or self.min_free_bytes < 0:
            raise ValueError("invalid workspace budget")
        if self.max_files_per_batch < 1 or self.max_batch_bytes < 1:
            raise ValueError("invalid batch limits")
        for stage in STAGES:
            self.stage_caps.setdefault(stage, self.workspace_max_bytes)
        if not self.high_watermark_bytes:
            self.high_watermark_bytes = self.min_free_bytes
        if not self.low_watermark_bytes:
            self.low_watermark_bytes = self.high_watermark_bytes
        if self.low_watermark_bytes < self.high_watermark_bytes:
            raise ValueError("low watermark must be >= high watermark")

    @classmethod
    def from_workspace(cls, workspace: Dict[str, Any], path: str,
                       **overrides: Any) -> "Budget":
        """Build a budget from campaign workspace config + REAL disk state."""
        free = shutil.disk_usage(path).free
        cfg = dict(workspace or {})
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        min_free = int(cfg.get("min_free_bytes", 40 * 1024**3))
        usable = free - min_free
        if usable <= 0:
            raise ValueError(
                f"no usable space at {path}: free={free}, min_free={min_free}")
        max_w = int(cfg.get("max_bytes", usable))
        workspace_max = min(max_w, usable)
        if workspace_max <= 0:
            raise ValueError("workspace_max_bytes must be > 0")
        return cls(
            workspace_max_bytes=workspace_max, min_free_bytes=min_free,
            stage_caps=dict(cfg.get("stage_caps") or {}),
            max_files_per_batch=int(cfg.get("max_files_per_batch", 500)),
            max_batch_bytes=int(cfg.get("max_batch_bytes", 10 * 1024**3)),
            high_watermark_bytes=int(
                cfg.get("high_watermark_bytes", 0) or min_free),
            low_watermark_bytes=int(
                cfg.get("low_watermark_bytes", 0)
                or cfg.get("high_watermark_bytes", 0) or min_free))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchPlan:
    batches: List[Batch]
    plan_hash: str
    total_known_bytes: int
    total_unknown_files: int
    budget: Dict[str, Any]
    profile: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"batches": [b.batch_id for b in self.batches],
                "n_batches": len(self.batches),
                "plan_hash": self.plan_hash,
                "total_known_bytes": self.total_known_bytes,
                "total_unknown_files": self.total_unknown_files,
                "budget": self.budget, "profile": self.profile}


def plan_with_budget(repo_id: str, pinned_sha: str, job_id: Optional[str],
                     files: Sequence[Tuple[str, Optional[int]]],
                     budget: Budget, profile: Optional[StageProfile] = None,
                     checkpoint_prefix: str = "checkpoints",
                     start_index: int = 0) -> BatchPlan:
    """Deterministically pack files into reservation-carrying batches.

    Same inputs (files + budget + profile) always yield the same batch_ids
    and reservations. A single file whose footprint exceeds the workspace is
    never split — it becomes a BLOCKED batch so an operator (or a future
    verified streaming path) handles it explicitly.
    """
    prof = profile or StageProfile()
    jid = job_id or job_id_for(repo_id, pinned_sha)
    ordered = sorted(files, key=lambda e: e[0])
    canon_src = "\0".join(
        [repo_id, pinned_sha,
         *[f"{p}\0{s if isinstance(s, int) and s >= 0 else '?'}"
           for p, s in ordered],
         str(budget.to_dict()), str(prof.to_dict())])
    plan_hash = hashlib.sha256(canon_src.encode()).hexdigest()[:16]

    groups: List[List[Tuple[str, Optional[int]]]] = []
    cur: List[Tuple[str, Optional[int]]] = []
    cur_bytes, cur_unknown = 0, 0

    def flush() -> None:
        nonlocal cur, cur_bytes, cur_unknown
        if cur:
            groups.append(cur)
        cur, cur_bytes, cur_unknown = [], 0, 0

    for path, size in ordered:
        known = size if isinstance(size, int) and size >= 0 else None
        acct = known if known is not None else prof.unknown_estimate_bytes
        would_overflow = (
            cur and (len(cur) >= budget.max_files_per_batch
                     or cur_bytes + acct > budget.max_batch_bytes
                     or (known is None
                         and cur_unknown >= prof.max_unknown_per_batch)))
        if would_overflow:
            flush()
        cur.append((path, size))
        if known is not None:
            cur_bytes += known
        else:
            cur_unknown += 1
    flush()

    batches: List[Batch] = []
    total_known, total_unknown = 0, 0
    for i, g in enumerate(groups):
        idx = start_index + i
        paths = [p for p, _ in g]
        known_sum = sum(s for _, s in g if isinstance(s, int) and s >= 0)
        unknown_n = sum(1 for _, s in g if not (isinstance(s, int) and s >= 0))
        accountable = known_sum + unknown_n * prof.unknown_estimate_bytes
        reservation = prof.footprint(accountable)
        reservation["accountable_bytes"] = accountable
        bid = batch_id_for(repo_id, pinned_sha, idx, paths)
        oversize = (len(g) == 1 and known_sum > 0
                    and sum(reservation[s] for s in STAGES)
                    > budget.workspace_max_bytes)
        # unknown-only accounting can also exceed: same treatment
        if not oversize and len(g) == 1 and unknown_n and \
                sum(reservation[s] for s in STAGES) > budget.workspace_max_bytes:
            oversize = True
        status = BatchStatus.PLANNED.value
        result: Dict[str, Any] = {}
        if oversize:
            status = BatchStatus.BLOCKED.value
            result = {"oversize": True,
                      "blocker": f"single-file footprint exceeds workspace "
                                 f"({accountable} accountable bytes); needs "
                                 f"verified streaming path or bigger budget"}
        batches.append(Batch(
            batch_id=bid, job_id=jid, index=idx, files=paths,
            file_sizes={p: (s if isinstance(s, int) and s >= 0 else None)
                        for p, s in g},
            total_bytes=known_sum, n_unknown_bytes=unknown_n,
            reservation=reservation,
            checkpoint_rel=f"{checkpoint_prefix}/{jid}/{bid}.jsonl",
            status=status, result=result))
        total_known += known_sum
        total_unknown += unknown_n
    return BatchPlan(batches=batches, plan_hash=plan_hash,
                     total_known_bytes=total_known,
                     total_unknown_files=total_unknown,
                     budget=budget.to_dict(), profile=prof.to_dict())


class ReservationLedger:
    """Runtime holder of reservations. Acquire BEFORE a batch starts work.

    Thread-safe: the pump overlap worker acquires in the main thread and
    releases from the worker thread.
    """

    def __init__(self, caps: Dict[str, int]):
        self.caps = {s: int(caps.get(s, 0)) for s in STAGES}
        self.used: Dict[str, int] = {s: 0 for s in STAGES}
        self.held: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()

    def fits(self, reservation: Dict[str, int]) -> bool:
        with self._lock:
            return self._fits_locked(reservation)

    def _fits_locked(self, reservation: Dict[str, int]) -> bool:
        return all(self.used[s] + int(reservation.get(s, 0)) <= self.caps[s]
                   for s in STAGES)

    def acquire(self, batch: Batch) -> bool:
        """Hold a PLANNED batch's reservation. False => must not start."""
        with self._lock:
            if batch.status != BatchStatus.PLANNED.value:
                return False
            if batch.batch_id in self.held:
                return False  # already held: no double-acquire
            need = {s: int(batch.reservation.get(s, 0)) for s in STAGES}
            if not self._fits_locked(batch.reservation):
                return False
            for s in STAGES:
                self.used[s] += need[s]
            self.held[batch.batch_id] = need
            return True

    def release(self, batch_id: str) -> bool:
        with self._lock:
            need = self.held.pop(batch_id, None)
            if need is None:
                return False
            for s in STAGES:
                self.used[s] = max(0, self.used[s] - need[s])
            return True

    def totals(self) -> Dict[str, Any]:
        with self._lock:
            return {"caps": dict(self.caps), "used": dict(self.used),
                    "held_batches": sorted(self.held)}


class WatermarkGate:
    """Hysteresis admission gate over real free bytes.

    Running until free drops below HIGH (stop), then stopped until free
    recovers to LOW (run). Prevents flapping when free hovers at one line.
    """

    def __init__(self, high_watermark_bytes: int, low_watermark_bytes: int):
        if low_watermark_bytes < high_watermark_bytes:
            raise ValueError("low watermark must be >= high watermark")
        self.high = high_watermark_bytes
        self.low = low_watermark_bytes
        self.stopped = False

    def evaluate(self, free_bytes: int) -> str:
        if not self.stopped and free_bytes < self.high:
            self.stopped = True
        elif self.stopped and free_bytes >= self.low:
            self.stopped = False
        return "stop" if self.stopped else "run"


def try_admit(batch: Batch, ledger: ReservationLedger, free_bytes: int,
              min_free_bytes: int) -> bool:
    """Admit a batch iff: PLANNED + reservation fits + free stays >= min_free.

    The single choke point answering Gate 2: no task starts without a held
    reservation for its execution path.
    """
    if batch.status != BatchStatus.PLANNED.value:
        return False
    footprint = sum(int(batch.reservation.get(s, 0)) for s in STAGES)
    if free_bytes - footprint < min_free_bytes:
        return False
    return ledger.acquire(batch)
