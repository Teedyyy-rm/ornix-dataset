"""Durable campaign state store (MD-001).

Layout under ``root`` (every file written atomically via util.io.write_json,
which is tmp + fsync + os.replace + dir fsync)::

    campaign.json            # Campaign meta + ordered job_ids
    jobs/<job_id>.json       # one DatasetJob per file
    batches/<batch_id>.json  # one Batch per file
    checkpoints/<job_id>/<batch_id>.jsonl   # per-batch Checkpoint log
                                            # (written by workers, read via
                                            #  ops.checkpoint.Checkpoint)

Resume is idempotent: re-creating from the same input returns the existing
job (``duplicate: True``) and pinned SHAs are never mutated by resume.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..ops.checkpoint import Checkpoint
from ..util.io import read_json, write_json
from ..util.timeutil import utc_now_iso
from .models import (
    Batch,
    Campaign,
    DatasetJob,
    JobStatus,
    job_id_for,
    slug_repo_id,
)
from .planning import plan_batches
from .refs import normalize_repo_ref, pin_revision
from .resources import Budget, StageProfile, plan_with_budget

_CAMPAIGN_FILE = "campaign.json"
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_name(name: str) -> str:
    return _SAFE.sub("_", name.strip())[:64] or "campaign"


class CampaignStore:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.jobs_dir = os.path.join(self.root, "jobs")
        self.batches_dir = os.path.join(self.root, "batches")
        self.checkpoints_dir = os.path.join(self.root, "checkpoints")
        self.inventory_dir = os.path.join(self.root, "inventory")
        for d in (self.jobs_dir, self.batches_dir, self.checkpoints_dir,
                  self.inventory_dir):
            os.makedirs(d, exist_ok=True)

    # -- paths -----------------------------------------------------------
    @property
    def campaign_file(self) -> str:
        return os.path.join(self.root, _CAMPAIGN_FILE)

    def job_file(self, job_id: str) -> str:
        if "/" in job_id or job_id.startswith("."):
            raise ValueError(f"unsafe job_id: {job_id!r}")
        return os.path.join(self.jobs_dir, f"{job_id}.json")

    def batch_file(self, batch_id: str) -> str:
        if "/" in batch_id or batch_id.startswith("."):
            raise ValueError(f"unsafe batch_id: {batch_id!r}")
        return os.path.join(self.batches_dir, f"{batch_id}.json")

    def checkpoint_file(self, job_id: str, batch_id: str) -> str:
        """Absolute batch checkpoint path. Convention: checkpoints/<job>/<batch>.jsonl."""
        if "/" in job_id or "/" in batch_id:
            raise ValueError("unsafe checkpoint components")
        return os.path.join(self.checkpoints_dir, job_id, f"{batch_id}.jsonl")

    def inventory_file(self, job_id: str) -> str:
        """Default home of `campaign inventory` output for a job."""
        if "/" in job_id or job_id.startswith("."):
            raise ValueError(f"unsafe job_id: {job_id!r}")
        return os.path.join(self.inventory_dir, f"{job_id}.jsonl")

    def batch_checkpoint(self, batch: Batch) -> Checkpoint:
        """A ready-to-use Checkpoint bound to this batch's log path."""
        return Checkpoint(self.checkpoint_file(batch.job_id, batch.batch_id),
                          run_id=batch.batch_id)

    # -- campaign --------------------------------------------------------
    def load_campaign(self) -> Campaign:
        return Campaign.from_dict(read_json(self.campaign_file))

    def save_campaign(self, campaign: Campaign) -> None:
        write_json(self.campaign_file, campaign.to_dict())

    # -- jobs ------------------------------------------------------------
    def load_job(self, job_id: str) -> DatasetJob:
        return DatasetJob.from_dict(read_json(self.job_file(job_id)))

    def save_job(self, job: DatasetJob) -> None:
        write_json(self.job_file(job.job_id), job.to_dict())

    def load_batch(self, batch_id: str) -> Batch:
        return Batch.from_dict(read_json(self.batch_file(batch_id)))

    def save_batch(self, batch: Batch) -> None:
        write_json(self.batch_file(batch.batch_id), batch.to_dict())

    def find_duplicate(self, repo_id: str, pinned_sha: Optional[str],
                       requested_revision: Optional[str]) -> Optional[DatasetJob]:
        """An existing job for the same repo + same pin (or same requested ref)."""
        key = repo_id.lower()
        for _jid, job in self._iter_jobs():
            if job.repo_id.lower() != key:
                continue
            if pinned_sha and job.pinned_sha == pinned_sha:
                return job
            if (requested_revision and job.requested_revision == requested_revision
                    and not job.pinned_sha and not pinned_sha):
                return job
        return None

    def _iter_jobs(self):
        for fn in sorted(os.listdir(self.jobs_dir)):
            if fn.endswith(".json"):
                yield fn[:-5], self.load_job(fn[:-5])

    # -- create ----------------------------------------------------------
    def create_campaign(self, name: str, datasets: List[Dict[str, Any]],
                        destination: Optional[Dict[str, Any]] = None,
                        publish_approved: bool = False,
                        workspace: Optional[Dict[str, Any]] = None,
                        resolver: Optional[Callable[[str, str], str]] = None,
                        ) -> Tuple[Campaign, List[Dict[str, Any]]]:
        """Create (or attach to) the campaign in this root.

        Idempotent per dataset: an input that matches an existing job returns
        that job with ``duplicate: True`` instead of creating a new one.
        Unresolvable refs become BLOCKED jobs — never a crash.
        """
        if not datasets:
            raise ValueError("campaign needs at least one dataset entry")
        campaign_id = _safe_name(name)
        try:
            campaign = self.load_campaign()
            if campaign.campaign_id != campaign_id:
                raise ValueError(
                    f"root holds campaign {campaign.campaign_id!r}, "
                    f"not {campaign_id!r}")
        except FileNotFoundError:
            campaign = Campaign(campaign_id=campaign_id, name=name.strip(),
                                destination=destination or {},
                                publish_approved=bool(publish_approved),
                                workspace=workspace or {},
                                created_utc=utc_now_iso())
            self.save_campaign(campaign)
        outcomes: List[Dict[str, Any]] = []
        for entry in datasets:
            outcomes.append(self._ensure_job(
                campaign, entry, resolver=resolver))
            self.save_campaign(campaign)
        return campaign, outcomes

    def _ensure_job(self, campaign: Campaign, entry: Dict[str, Any],
                    resolver: Optional[Callable[[str, str], str]]
                    ) -> Dict[str, Any]:
        raw = entry.get("repo") or entry.get("url") or entry.get("repo_id")
        if not raw:
            return {"ok": False, "error": "dataset entry needs 'repo'",
                    "entry": entry}
        try:
            repo_id, url_rev = normalize_repo_ref(str(raw))
        except ValueError as e:
            return {"ok": False, "error": str(e), "entry": entry}
        requested = entry.get("revision") or url_rev
        pinned: Optional[str] = None
        blocker: Optional[str] = None
        if resolver is not None:
            try:
                pinned = pin_revision(repo_id, requested, resolver)
            except Exception as e:
                blocker = f"unresolvable {repo_id}@{requested or 'main'}: {e}"
        dup = self.find_duplicate(repo_id, pinned, requested)
        if dup is not None:
            if dup.job_id not in campaign.job_ids:
                campaign.job_ids.append(dup.job_id)
            return {"ok": True, "job_id": dup.job_id, "duplicate": True,
                    "repo_id": dup.repo_id, "pinned_sha": dup.pinned_sha,
                    "status": dup.status}
        if pinned:
            job_id = job_id_for(repo_id, pinned)
        else:
            job_id = f"{slug_repo_id(repo_id)}-unpinned"
        job = DatasetJob(
            job_id=job_id, repo_id=repo_id, requested_revision=requested,
            pinned_sha=pinned,
            splits=list(entry.get("splits") or entry.get("allow_splits") or []),
            rights=dict(entry.get("rights") or {}),
            status=(JobStatus.CREATED.value if pinned else JobStatus.BLOCKED.value),
            blockers=[blocker] if blocker else [],
            created_utc=utc_now_iso())
        self.save_job(job)
        if job.job_id not in campaign.job_ids:
            campaign.job_ids.append(job.job_id)
        return {"ok": True, "job_id": job.job_id, "duplicate": False,
                "repo_id": repo_id, "pinned_sha": pinned, "status": job.status,
                **({"blocker": blocker} if blocker else {})}

    # -- batch planning --------------------------------------------------
    def plan_job_batches(self, job_id: str,
                         files: List[Tuple[str, Optional[int]]],
                         max_files: int = 500,
                         max_bytes: int = 10 * 1024**3,
                         budget: Optional["Budget"] = None,
                         profile: Optional["StageProfile"] = None) -> List[Batch]:
        """Deterministically (re-)plan a pinned job's batches from a file list.

        Re-planning with the same file list yields the same batch_ids; the job
        record is updated atomically afterwards. Fail-closed on unpinned jobs.

        With ``budget`` (MD-002), batches carry per-stage reservations and a
        lone over-budget file becomes BLOCKED instead of silently passing.
        Without it, the mechanical MD-001 split is used (empty reservations).
        """
        job = self.load_job(job_id)
        if not job.pinned_sha:
            raise ValueError(f"job {job_id} has no pinned revision; resolve it first")
        if budget is not None:
            plan = plan_with_budget(job.repo_id, job.pinned_sha, job.job_id,
                                    files, budget, profile)
            batches = plan.batches
        else:
            batches = plan_batches(job.repo_id, job.pinned_sha, job.job_id,
                                   files, max_files=max_files,
                                   max_bytes=max_bytes)
        for b in batches:
            if b.checkpoint_rel != os.path.join(
                    "checkpoints", b.job_id, f"{b.batch_id}.jsonl"):
                raise AssertionError("checkpoint path convention violated")
            self.save_batch(b)
        job.batch_ids = [b.batch_id for b in batches]
        job.status = JobStatus.BATCHED.value
        self.save_job(job)
        campaign = self.load_campaign()
        if job_id not in campaign.job_ids:
            campaign.job_ids.append(job_id)
            self.save_campaign(campaign)
        return batches

    # -- resume ----------------------------------------------------------
    def resume(self, resolver: Optional[Callable[[str, str], str]] = None,
               check_remote: bool = False) -> Dict[str, Any]:
        """Reconcile stored state without ever mutating a pinned SHA.

        - re-reads every record (integrity: corrupt JSON is reported, not skipped)
        - retries pinning for BLOCKED/unpinned jobs when a resolver is available
        - with check_remote=True, re-resolves each job's requested revision and
          REPORTS drift (upstream moved) — the stored pin is never changed
        Running resume twice with no changes yields identical output.
        """
        campaign = self.load_campaign()  # raises if missing/corrupt: fail-closed
        report: Dict[str, Any] = {"campaign_id": campaign.campaign_id,
                                  "jobs": [], "drift": []}
        for jid in list(campaign.job_ids):
            job = self.load_job(jid)  # fail-closed on corrupt record
            entry: Dict[str, Any] = {"job_id": jid, "status": job.status,
                                     "pinned_sha": job.pinned_sha,
                                     "n_batches": len(job.batch_ids),
                                     "action": "none"}
            if not job.pinned_sha and resolver is not None:
                try:
                    job.pinned_sha = pin_revision(job.repo_id,
                                                  job.requested_revision, resolver)
                    job.status = JobStatus.CREATED.value
                    job.blockers = [b for b in job.blockers
                                    if not b.startswith("unresolvable")]
                    # Unpinned jobs can never own batches (planning is
                    # fail-closed without a pin), so adopting the canonical
                    # pinned job_id here is safe and keeps resume idempotent.
                    new_id = job_id_for(job.repo_id, job.pinned_sha)
                    if new_id != jid:
                        os.remove(self.job_file(jid))
                        campaign.job_ids = [
                            new_id if x == jid else x for x in campaign.job_ids]
                        jid = job.job_id = new_id
                        self.save_campaign(campaign)
                    self.save_job(job)
                    entry.update(job_id=jid, pinned_sha=job.pinned_sha,
                                 status=job.status, action="pinned")
                except Exception as e:
                    entry.update(action="still-blocked", error=str(e))
            if check_remote and resolver is not None and job.requested_revision \
                    and not is_unpinned_placeholder(job):
                try:
                    current = pin_revision(job.repo_id, job.requested_revision,
                                           resolver)
                    if current != job.pinned_sha:
                        report["drift"].append(
                            {"job_id": jid, "pinned_sha": job.pinned_sha,
                             "upstream_now": current,
                             "note": "upstream moved; stored pin UNCHANGED"})
                        entry["action"] = "drift-reported"
                except Exception as e:
                    entry.update(action="remote-check-failed", error=str(e))
            report["jobs"].append(entry)
        return report

    # -- status ----------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        campaign = self.load_campaign()
        jobs = []
        n_batches = 0
        by_status: Dict[str, int] = {}
        for jid in campaign.job_ids:
            job = self.load_job(jid)
            n_batches += len(job.batch_ids)
            by_status[job.status] = by_status.get(job.status, 0) + 1
            jobs.append({"job_id": jid, "repo_id": job.repo_id,
                         "requested_revision": job.requested_revision,
                         "pinned_sha": job.pinned_sha, "status": job.status,
                         "n_batches": len(job.batch_ids),
                         "blockers": job.blockers})
        return {"campaign_id": campaign.campaign_id, "name": campaign.name,
                "destination": campaign.destination,
                "publish_approved": campaign.publish_approved,
                "n_jobs": len(jobs), "n_batches": n_batches,
                "by_status": by_status, "jobs": jobs}


def is_unpinned_placeholder(job: DatasetJob) -> bool:
    return not job.pinned_sha or job.job_id.endswith("-unpinned")
