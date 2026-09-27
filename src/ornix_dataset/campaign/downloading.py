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
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..ingestion.hf_downloader import (
    DownloadConfig,
    DownloadResult,
    HfBatchDownloader,
    InventoryItem,
    matches_patterns,
)
from ..util.io import atomic_write_text
from ..util.jsonl import append_jsonl, load_jsonl
from .models import BatchStatus, DatasetJob
from .parquet import expand_parquet_row, is_parquet_path
from .resources import ReservationLedger, try_admit

# Checkpoint states that count a batch file as downloaded (kept local: the
# shared Checkpoint.done set has pipeline-level meanings we must not widen).
# EXTRACTED is the terminal state of a downloaded parquet shard whose clips
# were expanded and whose parent bytes were dropped.
DOWNLOADED_STATES = ("DOWNLOADED", "EXTRACTED", "DONE")

MANIFEST_NAME = "BATCH_MANIFEST.jsonl"


def _lfs_sha(entry: Any) -> Optional[str]:
    """LFS/Xet content hash, across huggingface_hub layouts.

    hub 1.x exposes ``lfs.oid``; hub 2.x exposes ``lfs.sha256`` (and leaves
    ``oid`` unset). Missing this means downloads are only size-checked, so
    both spellings are accepted and anything else degrades to None.
    """
    lfs = getattr(entry, "lfs", None) or {}
    candidates = []
    if isinstance(lfs, dict):
        candidates += [lfs.get("oid"), lfs.get("sha256")]
    else:
        candidates += [getattr(lfs, "oid", None),
                       getattr(lfs, "sha256", None)]
    for oid in candidates:
        if isinstance(oid, str) and len(oid) == 64:
            try:
                int(oid, 16)
                return oid.lower()
            except ValueError:
                continue
    return None


def _is_repo_folder(entry: Any) -> bool:
    """True for directory entries across hub versions.

    hub 1.x set ``entry.type``; hub 2.x dropped it and returns ``RepoFolder``
    (no ``size``/``lfs``). Defaulting a missing ``type`` to "file" let folder
    rows into the inventory and broke the downloader on a non-file path.
    """
    kind = getattr(entry, "type", None)
    if kind in ("directory", "folder"):
        return True
    if kind == "file":
        return False
    if type(entry).__name__.endswith("Folder"):
        return True
    # No type info at all: a real listing always gives files a size.
    return getattr(entry, "size", None) is None and not getattr(entry, "lfs", None)


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
        if _is_repo_folder(entry):
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


def _manifest_row(job: DatasetJob, res: Any, staging: str) -> Dict[str, Any]:
    return {
        "source_uri": f"hf://datasets/{job.repo_id}"
                      f"@{job.pinned_sha}/{res.path_in_repo}",
        "original_file_id": res.path_in_repo,
        "source_revision": job.pinned_sha,
        "staged_path": os.path.relpath(res.staged_path, staging),
        "sha256": res.sha256, "size": res.size,
        "verify_method": res.verify_method,
        "from_cache": bool(res.from_cache),
        "reused_staged": bool(res.reused_staged)}


def _append_rows(path: str, rows: List[Dict[str, Any]],
                 seen: set) -> int:
    """Append manifest rows that are not present yet; fsync once at the end."""
    added = 0
    if not rows:
        return 0
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fid = r.get("original_file_id")
            if fid in seen:
                continue
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            seen.add(fid)
            added += 1
        fh.flush()
        os.fsync(fh.fileno())
    return added


def expand_and_drop_parquet(job: DatasetJob, staging: str, res: Any,
                            ckpt: Any) -> Optional[Dict[str, Any]]:
    """Expand a verified parquet shard into clips, then drop the parent bytes.

    Bounded disk (anti-slow-download): the container is removed as soon as its
    clips exist and are hashed, so staging never holds a downloaded shard
    longer than the moment it takes to unpack it. Provenance (repo, pinned
    revision, verified sha256, size) survives in the manifest parent row.

    Returns None for non-parquet files, else a report dict. Fail-closed: a
    shard that cannot be unpacked is marked EXTRACT_BLOCKED and its bytes are
    KEPT (never dropped without a verified clip set).
    """
    if not is_parquet_path(res.path_in_repo):
        return None
    row = _manifest_row(job, res, staging)
    new_rows, rep = expand_parquet_row(staging, row)
    if not rep.get("ok"):
        ckpt.mark(res.path_in_repo, "EXTRACT_BLOCKED",
                  reason=rep.get("reason", "unknown"))
        return rep
    seen = {r.get("original_file_id") for r in read_manifest(staging)}
    added = _append_rows(manifest_path(staging), new_rows, seen)
    dropped, drop_error = False, ""
    try:
        if os.path.exists(res.staged_path):
            os.remove(res.staged_path)
            dropped = True
    except OSError as e:  # parent stays on disk; QC never needs it
        drop_error = str(e)
    ckpt.mark(res.path_in_repo, "EXTRACTED", n_clips=len(new_rows),
              clips_added=added, parent_dropped=dropped,
              drop_error=drop_error or None)
    return {"ok": True, "original_file_id": res.path_in_repo,
            "n_clips": len(new_rows), "clips_added": added,
            "parent_dropped": dropped}


def reap_expired_parents(job: DatasetJob, staging: str,
                         ckpt: Any) -> List[str]:
    """Drop parquet parents whose clips are already in the manifest.

    Crash-recovery twin of ``expand_and_drop_parquet``: if a run died between
    appending clips and unlinking the parent, the next run finishes the job.
    Returns the repo paths whose parent bytes were removed.
    """
    rows = read_manifest(staging)
    parents = [r for r in rows
               if is_parquet_path(r.get("original_file_id", ""))]
    if not parents:
        return []
    all_ids = {r.get("original_file_id") for r in rows}
    reaped: List[str] = []
    for parent in parents:
        fid = parent["original_file_id"]
        prefix = fid + "#row-"
        if not any(i.startswith(prefix) for i in all_ids):
            continue  # clips absent: never drop unproven bytes
        path = os.path.join(staging, parent.get("staged_path", ""))
        if not os.path.exists(path):
            continue
        try:
            os.remove(path)
        except OSError:
            continue
        reaped.append(fid)
        ckpt.mark(fid, "EXTRACTED", parent_dropped=True, reaped=True)
    return reaped


def run_batch(store: Any, job_id: str, batch_id: str,
              caps: Dict[str, int], min_free_bytes: int,
              ledger: Optional[ReservationLedger] = None,
              cfg: Optional[DownloadConfig] = None,
              tree: Optional[Dict[str, Tuple[Optional[int], Optional[str]]]] = None,
              download_fn: Optional[Callable[..., str]] = None,
              cache_check_fn: Optional[Callable[..., Optional[str]]] = None,
              _admitted: bool = False,
              ) -> Dict[str, Any]:
    """Download one batch's files into verified staging. Error boundary included.

    Returns a report; raises nothing except unexpected (non-Exception) faults.
    A batch whose manifest already covers every file is a no-op success.

    ``_admitted``: the caller already holds this batch's ledger reservation
    (pump overlap path) — skip admission but still release in ``finally``.
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
        # Finish any interrupted "expand then drop" before reporting done.
        reaped = reap_expired_parents(job, staging,
                                      store.batch_checkpoint(batch))
        return {"ok": True, "admitted": False, "reason": "already-complete",
                "n_files": len(batch.files), "downloaded": 0,
                "reused": len(batch.files), "parents_reaped": reaped}
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
    # (_admitted skips this: the overlap worker's reservation was taken by the
    # pump thread before spawning, and is released in the finally below.)
    if not _admitted and not try_admit(batch, ledger, free, min_free_bytes):
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

    extracted: List[Dict[str, Any]] = []
    write_lock = threading.Lock()

    def _on_staged(res: DownloadResult) -> None:
        # Earliest handoff (in the io/staging thread, see HfBatchDownloader).
        # Doing it here — not on the drain — is what bounds disk: a parquet
        # shard is unpacked and unlinked in the same window in which it is
        # the only file being staged, instead of queueing up as resident bytes.
        with write_lock:
            ckpt.mark(res.path_in_repo, "DOWNLOADED", sha256=res.sha256,
                      staged_path=res.staged_path, size=res.size)
            append_jsonl(manifest_path(staging),
                         _manifest_row(job, res, staging))
            rep = expand_and_drop_parquet(job, staging, res, ckpt)
            if rep is not None:
                extracted.append(rep)

    try:
        results = dl.run(items, staged_name=_flat_staged_name,
                         skip_staged=skip_staged, on_staged=_on_staged)
    except Exception as e:
        batch.result = {**(batch.result or {}),
                        "download": {"complete": False, "error": str(e),
                                     "n_extracted": len(extracted)}}
        store.save_batch(batch)
        return {"ok": False, "admitted": True, "reason": "download-error",
                "error": str(e), "n_extracted": len(extracted)}

    # Reconcile: streamed rows + any pre-existing rows, deduped and ordered.
    # The atomic rewrite is what guarantees no duplicate rows across retries.
    merged: Dict[str, Dict[str, Any]] = {}
    for r in read_manifest(staging):
        if r.get("original_file_id"):
            merged[r["original_file_id"]] = r
    for res in results:
        merged[res.path_in_repo] = _manifest_row(job, res, staging)
    rows = [merged[k] for k in sorted(merged)]
    _write_manifest(staging, rows)
    for res in results:
        if is_parquet_path(res.path_in_repo):
            # Do not resurrect a DOWNLOADED state over an EXTRACTED one: the
            # Checkpoint replays the LAST mark, and that must stay EXTRACTED.
            if not any(e.get("original_file_id") == res.path_in_repo
                       for e in extracted):
                ckpt.mark(res.path_in_repo, "DOWNLOADED", sha256=res.sha256,
                          staged_path=res.staged_path, size=res.size)
            continue
        ckpt.mark(res.path_in_repo, "DOWNLOADED", sha256=res.sha256,
                  staged_path=res.staged_path, size=res.size)
    known = set(merged)

    complete = known.issuperset(batch.files)
    n_parents = sum(1 for k in known
                    if is_parquet_path(k) and "#row-" not in k)
    n_clips = sum(1 for k in known if "#row-" in k)
    summary = {"complete": complete,
               "n_files": len(batch.files),
               "n_staged": len(known),
               "n_parents": n_parents,
               "n_clips": n_clips,
               "n_extracted": sum(1 for e in extracted if e.get("ok")),
               "n_extract_blocked": sum(1 for e in extracted
                                        if not e.get("ok")),
               "parents_dropped": sum(1 for e in extracted
                                      if e.get("parent_dropped")),
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
