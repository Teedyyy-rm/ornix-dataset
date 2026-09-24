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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..publishing.approval import release_digest
from ..util.timeutil import utc_now_iso
from .downloading import batch_staging_dir
from .models import BatchStatus
from .processing import qc_paths
from .publishing import load_receipt, release_file_inventory, receipt_path_for

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


def pump_campaign(store: Any, ledger: Any, gate: Any,
                  download_one: Callable[[str, str], Dict[str, Any]],
                  qc_one: Callable[[str, str], Dict[str, Any]],
                  gate_one: Callable[[str, str], Dict[str, Any]],
                  prepare_one: Optional[Callable[[str, str], Dict[str, Any]]] = None,
                  publish_one: Optional[Callable[[str], Dict[str, Any]]] = None,
                  cleanup_one: Optional[Callable[[str, str], Dict[str, Any]]] = None,
                  max_steps: int = 100) -> Dict[str, Any]:
    """Drive the job queue automatically within watermark and step budgets.

    One step = one action on one batch (download, QC, gate, prepare, publish,
    cleanup). Jobs run in campaign order; a job whose batches are all DONE is
    marked DONE. Publish without a callback is recorded as awaiting-publish
    (approval is an operator act, never automated here).
    """
    from shutil import disk_usage

    campaign = store.load_campaign()
    log: List[Dict[str, Any]] = []
    steps = 0

    def _step(action: str, job_id: str, batch_id: str = "",
              detail: Optional[Dict[str, Any]] = None) -> bool:
        nonlocal steps
        steps += 1
        log.append({"step": steps, "action": action, "job": job_id,
                    "batch": batch_id, **(detail or {})})
        return steps < max_steps

    for job_id in list(campaign.job_ids):
        job = store.load_job(job_id)
        if job.status == BatchStatus.DONE.value or job.status == "DONE":
            continue
        progressed = True
        while progressed and steps < max_steps:
            progressed = False
            job = store.load_job(job_id)
            for bid in list(job.batch_ids):
                if steps >= max_steps:
                    break
                b = store.load_batch(bid)
                st = b.status
                if st == BatchStatus.PLANNED.value:
                    if gate.evaluate(disk_usage(store.root).free) != "run":
                        if not _step("stopped-disk", job_id, bid):
                            return _report(campaign, log, steps)
                        continue
                    rep = download_one(job_id, bid)
                    progressed = True
                    if not _step("download", job_id, bid, {"ok": rep.get("ok")}):
                        return _report(campaign, log, steps)
                elif st == BatchStatus.IN_PROGRESS.value:
                    dl = (b.result or {}).get("download", {})
                    if not dl.get("complete"):
                        rep = download_one(job_id, bid)
                    else:
                        rep = qc_one(job_id, bid)
                    progressed = True
                    if not _step("qc" if dl.get("complete") else "download",
                                 job_id, bid, {"ok": rep.get("ok")}):
                        return _report(campaign, log, steps)
                elif st == BatchStatus.BATCH_PROCESSED.value:
                    rep = gate_one(job_id, bid)
                    progressed = True
                    if not _step("gate", job_id, bid, {"ok": rep.get("ok")}):
                        return _report(campaign, log, steps)
                elif st == BatchStatus.RELEASE_READY.value:
                    rel = (b.result or {}).get("release", {})
                    rid = rel.get("release_id", "")
                    if not rid:
                        if prepare_one is None:
                            if not _step("awaiting-release", job_id, bid):
                                return _report(campaign, log, steps)
                            continue
                        rep = prepare_one(job_id, bid)
                        progressed = True
                        if not _step("prepare", job_id, bid,
                                     {"ok": rep.get("ok")}):
                            return _report(campaign, log, steps)
                        continue
                    receipt_file = os.path.join(
                        store.root, "releases", f"{rid}.REMOTE_VERIFIED.json")
                    if os.path.exists(receipt_file):
                        if cleanup_one is not None:
                            rep = cleanup_one(job_id, bid)
                            progressed = True
                            if not _step("cleanup", job_id, bid,
                                         {"ok": rep.get("ok")}):
                                return _report(campaign, log, steps)
                    elif publish_one is not None:
                        rep = publish_one(rid)
                        progressed = True
                        if not _step("publish", job_id, bid,
                                     {"ok": rep.get("ok")}):
                            return _report(campaign, log, steps)
                    else:
                        if not _step("awaiting-publish", job_id, bid):
                            return _report(campaign, log, steps)
                # DONE batches need nothing
            job = store.load_job(job_id)
            states = [store.load_batch(x).status for x in job.batch_ids]
            if job.batch_ids and all(s == BatchStatus.DONE.value for s in states) \
                    and job.status != "DONE":
                job.status = "DONE"
                store.save_job(job)
                log.append({"step": steps + 1, "action": "job-done",
                            "job": job_id, "batch": ""})
    return _report(campaign, log, steps)


def _report(campaign: Any, log: List[Dict[str, Any]], steps: int) -> Dict[str, Any]:
    return {"campaign_id": campaign.campaign_id, "steps": steps, "log": log}
