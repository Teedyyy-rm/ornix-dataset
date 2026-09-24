"""Batch download adapter: planner contract ↔ HfBatchDownloader (MD-003).

The downloader itself is reused untouched. This adapter owns the MD-003
deliverables around it:

- ``fetch_job_inventory``: list (path, size, lfs-sha) at the job's PINNED
  commit — the planner input. Never lists a branch.
- ``run_batch``: the single choke point that downloads one batch:
  1. fail-closed pre-checks (pinned job, PLANNED-or-incomplete batch),
  2. ``try_admit`` against the MD-002 ledger (no reservation => no start),
  3. resume: files already marked DOWNLOADED in the batch checkpoint are
     skipped; previously staged files are offered back to the downloader,
     which re-verifies size+sha before reuse (never trusts existence),
  4. per-file checkpoint marks + atomic ``BATCH_MANIFEST.jsonl`` rewrite —
     only verified-staged files are handed to QC (MD-004),
  5. ledger release in ``finally``, batch record saved atomically.

Batches stay PLANNED across attempts until every file is verified-staged;
only then do they become IN_PROGRESS (ready for MD-004 QC).
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..ingestion.hf_downloader import (
    DownloadConfig,
    HfBatchDownloader,
    InventoryItem,
    matches_patterns,
)
from ..util.io import atomic_write_text
from ..util.jsonl import load_jsonl
from .models import BatchStatus, DatasetJob
from .resources import ReservationLedger, try_admit

# Checkpoint states that count a batch file as downloaded (kept local: the
# shared Checkpoint.done set has pipeline-level meanings we must not widen).
DOWNLOADED_STATES = ("DOWNLOADED", "DONE")

MANIFEST_NAME = "BATCH_MANIFEST.jsonl"


def _lfs_sha(entry: Any) -> Optional[str]:
    lfs = getattr(entry, "lfs", None) or {}
    oid = (lfs.get("oid") if isinstance(lfs, dict)
           else getattr(lfs, "oid", None))
    if isinstance(oid, str) and len(oid) == 64:
        try:
            int(oid, 16)
            return oid.lower()
        except ValueError:
            return None
    return None


def fetch_job_inventory(api: Any, job: DatasetJob,
                        allow: Optional[List[str]] = None,
                        ignore: Optional[List[str]] = None
                        ) -> List[Tuple[str, Optional[int], Optional[str]]]:
    """List NON-DIRECTORY files at the pinned commit: (path, size|None, sha|None).

    Fail-closed on unpinned jobs. Filtering happens BEFORE any byte is fetched.
    """
    if not job.pinned_sha:
        raise ValueError(f"job {job.job_id} has no pinned revision")
    out: List[Tuple[str, Optional[int], Optional[str]]] = []
    for entry in api.list_repo_tree(job.repo_id, revision=job.pinned_sha,
                                    repo_type="dataset", recursive=True):
        if getattr(entry, "type", "file") != "file":
            continue
        path = getattr(entry, "path", "")
        if not path or not matches_patterns(path, allow, ignore):
            continue
        size = getattr(entry, "size", None)
        out.append((path, int(size) if isinstance(size, int) else None,
                    _lfs_sha(entry)))
    return sorted(out, key=lambda e: e[0])


def batch_staging_dir(root: str, job_id: str, batch_id: str) -> str:
    for comp in (job_id, batch_id):
        if "/" in comp or comp.startswith("."):
            raise ValueError(f"unsafe staging component: {comp!r}")
    return os.path.join(os.path.abspath(root), "staging", job_id, batch_id)


def manifest_path(staging_dir: str) -> str:
    return os.path.join(staging_dir, MANIFEST_NAME)


def downloaded_files(store: Any, batch: Any) -> List[str]:
    """Files with a DOWNLOADED/DONE mark in the batch checkpoint (ordered)."""
    prog = store.batch_checkpoint(batch).load()
    return sorted(i for i, s in prog.states.items() if s in DOWNLOADED_STATES)


def read_manifest(staging_dir: str) -> List[Dict[str, Any]]:
    return load_jsonl(manifest_path(staging_dir))


def _write_manifest(staging_dir: str, rows: List[Dict[str, Any]]) -> None:
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                   for r in rows)
    atomic_write_text(manifest_path(staging_dir), body)


def _flat_staged_name(item: InventoryItem) -> str:
    # Flat staging namespace: unique per path, no directory traversal ever.
    return item.path_in_repo.replace("/", "__")


def run_batch(store: Any, job_id: str, batch_id: str,
              caps: Dict[str, int], min_free_bytes: int,
              ledger: Optional[ReservationLedger] = None,
              cfg: Optional[DownloadConfig] = None,
              tree: Optional[Dict[str, Tuple[Optional[int], Optional[str]]]] = None,
              download_fn: Optional[Callable[..., str]] = None,
              cache_check_fn: Optional[Callable[..., Optional[str]]] = None,
              ) -> Dict[str, Any]:
    """Download one batch's files into verified staging. Error boundary included.

    Returns a report; raises nothing except unexpected (non-Exception) faults.
    A batch whose manifest already covers every file is a no-op success.
    """
    job = store.load_job(job_id)
    if not job.pinned_sha:
        return {"ok": False, "admitted": False, "reason": "job-unpinned"}
    batch = store.load_batch(batch_id)  # unknown id => FileNotFoundError (fail-closed)
    if batch.job_id != job.job_id:
        raise ValueError(f"batch {batch_id} does not belong to job {job_id}")
    if batch.status == "BLOCKED":
        return {"ok": False, "admitted": False, "reason": "batch-blocked"}
    if batch.status not in ("PLANNED", "IN_PROGRESS"):
        return {"ok": False, "admitted": False,
                "reason": f"batch-status-{batch.status}"}
    if not batch.files:
        return {"ok": False, "admitted": False, "reason": "empty-batch"}

    staging = batch_staging_dir(store.root, job.job_id, batch.batch_id)

    # idempotent re-run: manifest already covers everything => no-op success
    # (reads only; safe before admission since it moves zero bytes)
    have = {r.get("original_file_id") for r in read_manifest(staging)}
    if have.issuperset(batch.files):
        return {"ok": True, "admitted": False, "reason": "already-complete",
                "n_files": len(batch.files), "downloaded": 0,
                "reused": len(batch.files)}
    if batch.status == "IN_PROGRESS":
        # _execute promotes to IN_PROGRESS only on full completion, so this
        # means external interference (e.g. manifest deleted): fail closed
        # instead of guessing.
        return {"ok": False, "admitted": False,
                "reason": "inconsistent-state"}

    ledger = ledger if ledger is not None else ReservationLedger(caps)
    free = shutil.disk_usage(store.root).free
    # Incomplete batches stay PLANNED (see _execute), so try_admit is the only
    # admission path: no reservation => the batch does not start. Period.
    if not try_admit(batch, ledger, free, min_free_bytes):
        return {"ok": False, "admitted": False,
                "reason": "no-reservation-or-disk"}
    os.makedirs(staging, exist_ok=True)
    try:
        return _execute(store, job, batch, staging, tree, cfg,
                        download_fn, cache_check_fn)
    finally:
        ledger.release(batch.batch_id)


def _execute(store: Any, job: DatasetJob, batch: Any, staging: str,
             tree: Optional[Dict[str, Tuple[Optional[int], Optional[str]]]],
             cfg: Optional[DownloadConfig],
             download_fn: Optional[Callable[..., str]],
             cache_check_fn: Optional[Callable[..., Optional[str]]]
             ) -> Dict[str, Any]:
    ckpt = store.batch_checkpoint(batch)
    done = set(downloaded_files(store, batch))

    sizes: Dict[str, Optional[int]] = dict(batch.file_sizes or {})
    shas: Dict[str, Optional[str]] = {}
    if tree is not None:
        for path, (size, sha) in tree.items():
            sizes[path] = size
            shas[path] = sha

    # skip map from the previous manifest: the downloader re-verifies
    # size+sha before reuse, so stale/corrupt staged files can never slip in.
    skip_staged = {
        f"{job.repo_id}@{job.pinned_sha}/{r['original_file_id']}": r["staged_path"]
        for r in read_manifest(staging) if r.get("staged_path")}

    items = [InventoryItem(path_in_repo=p,
                           size=int(sizes.get(p) or 0),
                           expected_sha256=shas.get(p),
                           index=i)
             for i, p in enumerate(batch.files) if p not in done]
    dl = HfBatchDownloader(job.repo_id, job.pinned_sha, staging,
                           cfg=cfg or DownloadConfig(),
                           download_fn=download_fn,
                           cache_check_fn=cache_check_fn)
    try:
        results = dl.run(items, staged_name=_flat_staged_name,
                         skip_staged=skip_staged)
    except Exception as e:
        batch.result = {**(batch.result or {}),
                        "download": {"complete": False, "error": str(e)}}
        store.save_batch(batch)
        return {"ok": False, "admitted": True, "reason": "download-error",
                "error": str(e)}

    rows = [r for r in read_manifest(staging)]
    known = {r.get("original_file_id") for r in rows}
    for res in results:
        ckpt.mark(res.path_in_repo, "DOWNLOADED", sha256=res.sha256,
                  staged_path=res.staged_path, size=res.size)
        if res.path_in_repo not in known:
            rows.append({
                "source_uri": f"hf://datasets/{job.repo_id}"
                              f"@{job.pinned_sha}/{res.path_in_repo}",
                "original_file_id": res.path_in_repo,
                "source_revision": job.pinned_sha,
                "staged_path": os.path.relpath(res.staged_path, staging),
                "sha256": res.sha256, "size": res.size,
                "verify_method": res.verify_method,
                "from_cache": bool(res.from_cache),
                "reused_staged": bool(res.reused_staged)})
            known.add(res.path_in_repo)
    # atomic rewrite: no duplicate rows across retries, ever
    rows.sort(key=lambda r: r.get("original_file_id", ""))
    _write_manifest(staging, rows)

    complete = known.issuperset(batch.files)
    summary = {"complete": complete,
               "n_files": len(batch.files),
               "n_staged": len(known),
               "pending": sorted(set(batch.files) - known),
               "network_bytes": dl.metrics.network_bytes,
               "n_reused_staged": dl.metrics.n_reused_staged,
               "n_cache_hit": dl.metrics.n_cache_hit,
               "retries": dl.metrics.retries}
    if complete:
        batch.status = BatchStatus.IN_PROGRESS.value  # ready for MD-004 QC
    batch.result = {**(batch.result or {}), "download": summary}
    store.save_batch(batch)
    return {"ok": complete, "admitted": True,
            "reason": "complete" if complete else "incomplete",
            **summary}
