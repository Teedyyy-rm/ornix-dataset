"""Verified cleanup & automatic continuation (MD-006).

Cleanup deletes ONLY regenerable intermediates, and only with proof:

- batch download staging + batch QC canonical outputs — after a
  REMOTE_VERIFIED receipt exists for the batch's release, the release dir is
  intact (digest matches the receipt), and the receipt used full-hash
  verification (sample-only receipts can never authorize deletion).
- NEVER deleted automatically: evidence, audit, checkpoints, manifests (the
  batch manifest is archived into releases/ before staging goes away), gate
  receipts, release dirs, or the shared HF cache (untouchable — other jobs
  and users share it).

Every deleted path must resolve inside the batch's own staging/QC run dirs
(ownership); anything else fails closed. Deletion is file-by-file and
re-entrant: a crash mid-cleanup is finished by the next run.

``pump_campaign`` drives the queue automatically: download → QC → gate →
(publish when a publish callback is supplied, else record awaiting-publish)
→ cleanup → next batch / next dataset, within watermark and step budgets.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass, field
from shutil import disk_usage
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from ..publishing.approval import release_digest
from ..util.timeutil import utc_now_iso
from .downloading import batch_staging_dir
from .models import BatchStatus
from .processing import qc_paths
from .publishing import load_receipt, release_file_inventory, receipt_path_for
from .resources import try_admit

ELIGIBLE_STATUSES = (BatchStatus.RELEASE_READY.value, BatchStatus.DONE.value)


@dataclass
class CleanupPolicy:
    require_full_hash: bool = True   # Gate-5 rule, enforced at deletion time
    archive_manifest: bool = True    # copy BATCH_MANIFEST into releases/ first
    # Fixed non-negotiables (documented, not flags): evidence/audit/
    # checkpoints/gate receipts/release dirs/HF cache are never auto-deleted.


def _owned(path: str, *roots: str) -> bool:
    real = os.path.realpath(path)
    return any(real == os.path.realpath(r) or
               real.startswith(os.path.realpath(r) + os.sep) for r in roots)


def manifest_archive_path(store: Any, job_id: str, batch_id: str) -> str:
    return os.path.join(store.root, "releases", f"{job_id}.{batch_id}.MANIFEST.jsonl")


def eligibility(store: Any, job_id: str, batch_id: str,
                ledger: Any = None,
                policy: Optional[CleanupPolicy] = None) -> Dict[str, Any]:
    """Check whether a batch's intermediates may be deleted. Read-only."""
    policy = policy or CleanupPolicy()
    job = store.load_job(job_id)
    batch = store.load_batch(batch_id)
    if batch.job_id != job.job_id:
        raise ValueError(f"batch {batch_id} does not belong to job {job_id}")
    if batch.status not in ELIGIBLE_STATUSES:
        return {"ok": False, "reason": f"batch-status-{batch.status}",
                "note": "only RELEASE_READY/DONE batches can be cleaned"}
    release = (batch.result or {}).get("release") or {}
    release_id = release.get("release_id")
    if not release_id:
        return {"ok": False, "reason": "no-release",
                "note": "batch was never released; nothing may be deleted"}
    try:
        receipt = load_receipt(store, release_id)
    except FileNotFoundError:
        return {"ok": False, "reason": "no-remote-receipt",
                "note": "no REMOTE_VERIFIED receipt for this batch's release"}
    if policy.require_full_hash and not receipt.get("full_hash_verified"):
        return {"ok": False, "reason": "sample-only-receipt",
                "note": "receipt is not full-hash verified; re-verify before cleanup"}
    release_dir = release.get("release_dir", "")
    if not release_dir or not os.path.isdir(release_dir):
        return {"ok": False, "reason": "release-dir-missing"}
    try:
        if release_digest(release_dir) != receipt.get("release_digest"):
            return {"ok": False, "reason": "release-dir-changed",
                    "note": "release dir no longer matches the verified digest"}
    except FileNotFoundError:
        return {"ok": False, "reason": "release-dir-changed"}
    # exact file-set + content check: catches added/stray files the digest
    # (which only covers MANIFEST.sha256) cannot see.
    if receipt.get("files") != release_file_inventory(release_dir):
        return {"ok": False, "reason": "release-dir-changed",
                "note": "release dir file set/content differs from receipt"}
    if ledger is not None and batch.batch_id in getattr(
            ledger, "held", {}):
        return {"ok": False, "reason": "batch-in-flight",
                "note": "reservation still held; workers active"}
    staging = batch_staging_dir(store.root, job.job_id, batch.batch_id)
    canonical = qc_paths(store, job.job_id, batch.batch_id).canonical_dir
    for p in (staging, canonical):
        if not _owned(p, store.root):
            return {"ok": False, "reason": "ownership-violation", "path": p}
    return {"ok": True, "release_id": release_id, "receipt": receipt,
            "staging": staging, "canonical": canonical}


def cleanup_batch(store: Any, job_id: str, batch_id: str,
                  ledger: Any = None,
                  policy: Optional[CleanupPolicy] = None) -> Dict[str, Any]:
    """Delete a batch's regenerable intermediates. Idempotent and re-entrant."""
    policy = policy or CleanupPolicy()
    check = eligibility(store, job_id, batch_id, ledger, policy)
    if not check["ok"]:
        return {"ok": False, **{k: v for k, v in check.items() if k != "ok"}}
    batch = store.load_batch(batch_id)
    removed: List[str] = []
    freed = 0

    def _rmtree(path: str) -> None:
        nonlocal freed
        if not os.path.isdir(path):
            return
        for dp, _ds, fs in os.walk(path):
            for f in fs:
                full = os.path.join(dp, f)
                if full.endswith("BATCH_MANIFEST.jsonl") and policy.archive_manifest:
                    continue  # archived below, removed with the dir after
                try:
                    freed += os.path.getsize(full)
                except OSError:
                    pass
        shutil.rmtree(path, ignore_errors=False)
        removed.append(path)

    if policy.archive_manifest:
        src = os.path.join(check["staging"], "BATCH_MANIFEST.jsonl")
        dst = manifest_archive_path(store, job_id, batch_id)
        if os.path.exists(src) and not os.path.exists(dst):
            with open(src, "rb") as fh_in:
                data = fh_in.read()
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".part"
            with open(tmp, "wb") as fh_out:
                fh_out.write(data)
                fh_out.flush()
                os.fsync(fh_out.fileno())
            os.replace(tmp, dst)
    _rmtree(check["staging"])
    _rmtree(check["canonical"])
    # manifest archive is written before the staging rmtree; if the crash
    # happened between archive and rmtree, the rerun re-archives (exists →
    # skip) and finishes the deletion. Either order converges.
    if os.path.exists(check["staging"]) or os.path.exists(check["canonical"]):
        return {"ok": False, "reason": "cleanup-incomplete",
                "removed": removed, "freed_bytes": freed}
    batch.status = BatchStatus.DONE.value
    batch.result = {**(batch.result or {}),
                    "cleanup": {"freed_bytes": freed, "removed": removed,
                                "manifest_archive": manifest_archive_path(
                                    store, job_id, batch_id),
                                "cleaned_utc": utc_now_iso()}}
    store.save_batch(batch)
    return {"ok": True, "freed_bytes": freed, "removed": removed,
            "manifest_archive": manifest_archive_path(store, job_id, batch_id)}


def _siblings_processed(store: Any, job: Any, exclude: str = "") -> bool:
    """True when every batch of the job reached at least BATCH_PROCESSED.

    The release gate is job-global, so it can only succeed once no batch is
    still PLANNED / IN_PROGRESS. Used to defer gate attempts instead of
    retrying a premature, guaranteed-failing gate.
    """
    ok_states = (BatchStatus.BATCH_PROCESSED.value,
                 BatchStatus.RELEASE_READY.value, BatchStatus.DONE.value)
    for bid in job.batch_ids:
        if bid == exclude:
            continue
        if store.load_batch(bid).status not in ok_states:
            return False
    return True


class _Outcome(NamedTuple):
    """Effect of one batch action on the pump's two loops.

    ``progressed`` re-enters the job's outer while-loop (real work happened).
    ``break_scan`` ends the batch pass early in overlap mode so the next pass
    can pair QC-current with download-next (G1).
    """

    progressed: bool
    break_scan: bool


_NOTHING = _Outcome(False, False)     # no work; may still have logged a status
_BREAK = _Outcome(True, True)        # download / qc / gate / publish / cleanup
_KEEP_SCAN = _Outcome(True, False)   # prepare: real work, but keep scanning


@dataclass
class _Pump:
    """Mutable state of one ``pump_campaign`` run.

    Holding the callbacks, the step budget and the step log here lets every
    phase be a small method instead of one branch inside a long state machine.
    """

    store: Any
    ledger: Any
    gate: Any
    download_one: Callable[..., Dict[str, Any]]
    qc_one: Callable[[str, str], Dict[str, Any]]
    gate_one: Callable[[str, str], Dict[str, Any]]
    prepare_one: Optional[Callable[[str, str], Dict[str, Any]]] = None
    publish_one: Optional[Callable[[str], Dict[str, Any]]] = None
    cleanup_one: Optional[Callable[[str, str], Dict[str, Any]]] = None
    max_steps: int = 100
    min_free_bytes: int = 0
    log: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    stopped: bool = False   # step budget exhausted

    # -- budget + log --------------------------------------------------------

    def record(self, action: str, job_id: str, batch_id: str = "",
               detail: Optional[Dict[str, Any]] = None) -> None:
        """Append a step and latch ``stopped`` once the budget is gone.

        ``stopped`` is a latch rather than an exception so the caller still runs
        its end-of-pass bookkeeping (job completion) before unwinding — a run
        that dies exactly on its last step never loses a DONE marker.
        """
        self.steps += 1
        self.log.append({"step": self.steps, "action": action,
                         "job": job_id, "batch": batch_id, **(detail or {})})
        if self.steps >= self.max_steps:
            self.stopped = True

    def free_bytes(self) -> int:
        return disk_usage(self.store.root).free

    def may_run(self) -> bool:
        return self.gate.evaluate(self.free_bytes()) == "run"

    # -- job level -----------------------------------------------------------

    def mark_job_done(self, job_id: str) -> None:
        """Mark a job DONE once every batch is DONE. Idempotent."""
        job = self.store.load_job(job_id)
        if job.status == BatchStatus.DONE.value or not job.batch_ids:
            return
        if all(self.store.load_batch(b).status == BatchStatus.DONE.value
               for b in job.batch_ids):
            job.status = BatchStatus.DONE.value
            self.store.save_job(job)
            self.log.append({"step": self.steps + 1, "action": "job-done",
                             "job": job_id, "batch": ""})

    def run_job(self, job_id: str, overlap: bool) -> None:
        """Drive one job's queue until it is DONE, stuck, or out of budget."""
        if self.store.load_job(job_id).status == BatchStatus.DONE.value:
            return
        while not self.stopped:
            progressed = False
            job = self.store.load_job(job_id)
            if overlap and not self.stopped and self._overlap_step(job):
                progressed = True
                self.mark_job_done(job_id)
                continue
            for bid in list(job.batch_ids):
                if self.stopped:
                    break
                outcome = self.advance(job, self.store.load_batch(bid))
                progressed = progressed or outcome.progressed
                if overlap and outcome.break_scan:
                    break
            self.mark_job_done(job_id)
            if not progressed:
                break

    # -- batch level ---------------------------------------------------------

    def advance(self, job: Any, batch: Any) -> _Outcome:
        """Advance one batch by at most one action."""
        st = batch.status
        if st == BatchStatus.PLANNED.value:
            return self._download_planned(batch)
        if st == BatchStatus.IN_PROGRESS.value:
            return self._resume_in_progress(batch)
        if st == BatchStatus.BATCH_PROCESSED.value:
            return self._gate_batch(job, batch)
        if st == BatchStatus.RELEASE_READY.value:
            return self._finish_release(batch)
        return _NOTHING  # DONE batches need nothing

    def _download_planned(self, batch: Any) -> _Outcome:
        if not self.may_run():
            self.record("stopped-disk", batch.job_id, batch.batch_id)
            return _NOTHING
        rep = self.download_one(batch.job_id, batch.batch_id)
        self.record("download", batch.job_id, batch.batch_id,
                    {"ok": rep.get("ok")})
        return _BREAK

    def _resume_in_progress(self, batch: Any) -> _Outcome:
        """Finish an interrupted download, or QC a complete one."""
        complete = bool(
            (batch.result or {}).get("download", {}).get("complete"))
        rep = self.qc_one(batch.job_id, batch.batch_id) if complete \
            else self.download_one(batch.job_id, batch.batch_id)
        self.record("qc" if complete else "download",
                    batch.job_id, batch.batch_id, {"ok": rep.get("ok")})
        return _BREAK

    def _gate_batch(self, job: Any, batch: Any) -> _Outcome:
        # The gate is job-global: it refuses until EVERY sibling batch is
        # processed. Attempting it early would burn the step budget on a
        # guaranteed failure and starve the QC of later batches (they never
        # get their turn), so defer instead of retrying.
        if not _siblings_processed(self.store, job, batch.batch_id):
            return _NOTHING
        rep = self.gate_one(batch.job_id, batch.batch_id)
        self.record("gate", batch.job_id, batch.batch_id,
                    {"ok": rep.get("ok")})
        return _BREAK

    def _finish_release(self, batch: Any) -> _Outcome:
        """RELEASE_READY: prepare, publish, then clean up once verified."""
        job_id, bid = batch.job_id, batch.batch_id
        rid = ((batch.result or {}).get("release") or {}).get("release_id", "")
        if not rid:
            return self._prepare_or_wait(job_id, bid)
        if os.path.exists(receipt_path_for(self.store, rid)):
            if self.cleanup_one is None:
                return _NOTHING
            rep = self.cleanup_one(job_id, bid)
            self.record("cleanup", job_id, bid, {"ok": rep.get("ok")})
            return _BREAK
        if self.publish_one is None:
            self.record("awaiting-publish", job_id, bid)
            return _NOTHING
        rep = self.publish_one(rid)
        self.record("publish", job_id, bid, {"ok": rep.get("ok")})
        return _BREAK

    def _prepare_or_wait(self, job_id: str, bid: str) -> _Outcome:
        """Build the missing release, or wait for an operator to do it."""
        if self.prepare_one is None:
            self.record("awaiting-release", job_id, bid)
            return _NOTHING
        rep = self.prepare_one(job_id, bid)
        self.record("prepare", job_id, bid, {"ok": rep.get("ok")})
        return _KEEP_SCAN

    # -- overlap -------------------------------------------------------------

    def _overlap_step(self, job: Any) -> bool:
        """One overlapped pair: QC the current batch while downloading the next.

        Returns True when a pair ran (the caller re-loops). Admission happens
        here in the main thread; the worker only executes the admitted download.
        """
        qc_bid, dl_bid = None, None
        for bid in job.batch_ids:
            b = self.store.load_batch(bid)
            if qc_bid is None and b.status == BatchStatus.IN_PROGRESS.value \
                    and (b.result or {}).get("download", {}).get("complete"):
                qc_bid = bid
            elif dl_bid is None and b.status == BatchStatus.PLANNED.value:
                dl_bid = bid
            if qc_bid and dl_bid:
                break
        if not qc_bid or not dl_bid:
            return False
        free = self.free_bytes()
        if self.gate.evaluate(free) != "run":
            return False
        if not try_admit(self.store.load_batch(dl_bid), self.ledger,
                         free, self.min_free_bytes):
            return False  # reservation unavailable: sequential path handles it

        worker_rep: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                worker_rep.update(self.download_one(job.job_id, dl_bid, True))
            except Exception as e:  # never let a worker thread die silently
                worker_rep.update({"ok": False, "admitted": True,
                                   "reason": f"worker-error: {e}"})

        t = threading.Thread(target=_worker, name=f"pump-dl-{dl_bid}",
                             daemon=True)
        t.start()
        try:
            qc_rep = self.qc_one(job.job_id, qc_bid)
        finally:
            t.join(timeout=3600)
        if t.is_alive():
            return False  # worker hung: leave states as-is, sequential retry later
        self.record("download", job.job_id, dl_bid,
                    {"ok": worker_rep.get("ok"), "overlapped": True})
        self.record("qc", job.job_id, qc_bid,
                    {"ok": qc_rep.get("ok"), "overlapped": True})
        return True


def pump_campaign(store: Any, ledger: Any, gate: Any,
                  download_one: Callable[..., Dict[str, Any]],
                  qc_one: Callable[[str, str], Dict[str, Any]],
                  gate_one: Callable[[str, str], Dict[str, Any]],
                  prepare_one: Optional[Callable[[str, str], Dict[str, Any]]] = None,
                  publish_one: Optional[Callable[[str], Dict[str, Any]]] = None,
                  cleanup_one: Optional[Callable[[str, str], Dict[str, Any]]] = None,
                  max_steps: int = 100, overlap: bool = False,
                  min_free_bytes: int = 0) -> Dict[str, Any]:
    """Drive the job queue automatically within watermark and step budgets.

    One step = one action on one batch (download, QC, gate, prepare, publish,
    cleanup). Jobs run in campaign order; a job whose batches are all DONE is
    marked DONE. Publish without a callback is recorded as awaiting-publish
    (approval is an operator act, never automated here).

    ``overlap`` (G1): while the main thread runs QC on one batch, a single
    background worker downloads the next PLANNED batch — its reservation is
    admitted up-front in the main thread (watermark + ledger), the worker runs
    ``download_one(job, batch, admitted=True)`` and releases on completion.
    Batches are disjoint (own staging/checkpoint/QC dirs), so the two threads
    never share mutable state except the locked ledger.
    """
    campaign = store.load_campaign()
    pump = _Pump(store=store, ledger=ledger, gate=gate,
                 download_one=download_one, qc_one=qc_one, gate_one=gate_one,
                 prepare_one=prepare_one, publish_one=publish_one,
                 cleanup_one=cleanup_one, max_steps=max_steps,
                 min_free_bytes=min_free_bytes)
    for job_id in list(campaign.job_ids):
        pump.run_job(job_id, overlap)
        if pump.stopped:
            break  # budget is global: remaining jobs would be no-ops anyway
    return {"campaign_id": campaign.campaign_id, "steps": pump.steps,
            "log": pump.log}
