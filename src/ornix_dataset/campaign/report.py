"""Campaign reporting + read-only dry-run plan (G4 / plan §2, MVP step 7).

``campaign_report`` aggregates the durable state into ONE artifact: job/batch
status, blockers, release receipts, and cleanup footprint. ``dry_run_plan``
walks the same state machine as ``pump_campaign`` WITHOUT executing anything —
it reports the action each batch would take next, so an operator can preview a
campaign before granting execution.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..util.timeutil import utc_now_iso
from .models import BatchStatus
from .publishing import list_releases, receipt_path_for


def _batch_action(store: Any, batch: Any, min_free_bytes: int) -> str:
    st = batch.status
    if st == BatchStatus.DONE.value:
        return "none"
    if st == BatchStatus.PLANNED.value:
        return "download"
    if st == BatchStatus.IN_PROGRESS.value:
        complete = (batch.result or {}).get("download", {}).get("complete")
        return "qc" if complete else "download"
    if st == BatchStatus.BATCH_PROCESSED.value:
        return "gate"
    if st == BatchStatus.RELEASE_READY.value:
        rel = (batch.result or {}).get("release") or {}
        rid = rel.get("release_id")
        if not rid:
            return "prepare-release"
        if os.path.exists(receipt_path_for(store, rid)):
            return "cleanup"
        return "publish (awaiting approval)"
    if st == BatchStatus.BLOCKED.value:
        return "blocked"
    return f"unknown:{st}"


def dry_run_plan(store: Any) -> Dict[str, Any]:
    """Read-only preview: what each batch would do next. Mutates nothing."""
    campaign = store.load_campaign()
    actions: List[Dict[str, Any]] = []
    for job_id in campaign.job_ids:
        job = store.load_job(job_id)
        for bid in job.batch_ids:
            batch = store.load_batch(bid)
            actions.append({"job_id": job_id, "batch_id": bid,
                            "repo_id": job.repo_id,
                            "pinned_sha": job.pinned_sha,
                            "status": batch.status,
                            "next_action": _batch_action(store, batch, 0),
                            "n_files": len(batch.files)})
    return {"campaign_id": campaign.campaign_id, "dry_run": True,
            "n_actions": len(actions), "actions": actions,
            "generated_utc": utc_now_iso()}


def campaign_report(store: Any) -> Dict[str, Any]:
    """Aggregate campaign state + blockers + releases into one JSON report."""
    campaign = store.load_campaign()
    jobs_out: List[Dict[str, Any]] = []
    by_status: Dict[str, int] = {}
    by_batch_status: Dict[str, int] = {}
    blockers: List[str] = []
    freed_total = 0
    n_cleaned = 0
    for job_id in campaign.job_ids:
        job = store.load_job(job_id)
        by_status[job.status] = by_status.get(job.status, 0) + 1
        for b in job.blockers:
            blockers.append(f"{job_id}: {b}")
        batches_out = []
        for bid in job.batch_ids:
            batch = store.load_batch(bid)
            by_batch_status[batch.status] = \
                by_batch_status.get(batch.status, 0) + 1
            cleanup = (batch.result or {}).get("cleanup") or {}
            freed_total += int(cleanup.get("freed_bytes", 0) or 0)
            if cleanup:
                n_cleaned += 1
            batches_out.append({"batch_id": bid, "status": batch.status,
                                "n_files": len(batch.files),
                                "next_action": _batch_action(store, batch, 0),
                                "freed_bytes": cleanup.get("freed_bytes", 0)})
        jobs_out.append({"job_id": job_id, "repo_id": job.repo_id,
                         "pinned_sha": job.pinned_sha, "status": job.status,
                         "blockers": job.blockers, "batches": batches_out})
    releases = []
    for rec in list_releases(store):
        rid = rec.get("release_id")
        published = bool(rid) and os.path.exists(receipt_path_for(store, rid))
        releases.append({"release_id": rid, "batch_id": rec.get("batch_id"),
                         "ready": rec.get("ready"),
                         "published_verified": published,
                         "blockers": rec.get("blockers", []),
                         "digest": rec.get("digest")})
        for blk in rec.get("blockers", []):
            blockers.append(f"{rid}: {blk}")
    return {"campaign_id": campaign.campaign_id, "name": campaign.name,
            "destination": campaign.destination,
            "publish_approved": campaign.publish_approved,
            "n_jobs": len(jobs_out),
            "n_batches": sum(len(j["batches"]) for j in jobs_out),
            "by_job_status": by_status, "by_batch_status": by_batch_status,
            "cleanup": {"freed_bytes": freed_total, "n_cleaned": n_cleaned},
            "n_releases": len(releases), "releases": releases,
            "blockers": blockers, "jobs": jobs_out,
            "generated_utc": utc_now_iso()}
