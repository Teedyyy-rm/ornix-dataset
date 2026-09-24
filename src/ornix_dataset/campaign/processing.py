"""Processing orchestrator: verified staging → audio QC → release gate (MD-004).

The audio engine is reused untouched through an injected ``analyzer`` callable
(default: ``OrnixPipeline.analyze_source``). This module owns the MD-004
deliverables around it:

- ``run_qc_batch``: per-file SourceRecords from the BATCH_MANIFEST, direct
  audio into the analyzer, container shards (parquet/tar/…) recorded as
  blockers (no wired extractor exists yet — ``scripts/extract_hf_audio.py``
  is a standalone tool, not a batch extractor), per-file checkpoint marks,
  atomic BATCH_ACCEPTED.
- ``gate_batch_release``: the BATCH_PROCESSED vs RELEASE_READY split.
  A batch becomes RELEASE_READY only when the JOB-global evidence is whole:
  every batch's accepted rows present, split assignment leak-free
  (``curation.split``), exact content duplicates computed and reported
  (``curation.dedup`` — reported, never silently dropped; the quality policy
  itself is unchanged).
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from ..audit.events import AuditLog
from ..contracts.source import SourceRecord
from ..curation.dedup import exact_duplicates
from ..curation.split import assign_splits, split_leakage
from ..ingestion.local import AUDIO_EXTS
from ..pipeline import RunPaths
from ..util.hashing import short_id
from ..util.io import atomic_write_text, write_json
from ..util.jsonl import append_jsonl, load_jsonl
from ..util.timeutil import utc_now_iso
from .downloading import batch_staging_dir, read_manifest
from .models import BatchStatus

# Checkpoint states with a terminal QC outcome (kept local, same reason as MD-003).
QC_DONE_STATES = ("QC_DONE", "SHARD_BLOCKED", "DONE")


def qc_run_id(job_id: str, batch_id: str) -> str:
    return f"{job_id}__{batch_id}"


def qc_paths(store: Any, job_id: str, batch_id: str,
             workdir: Optional[str] = None) -> RunPaths:
    base = workdir or os.path.join(store.root, "qc")
    return RunPaths.create(base, qc_run_id(job_id, batch_id))


def manifest_to_source_record(row: Dict[str, Any], job: Any,
                              staging_dir: str) -> SourceRecord:
    """Deterministic SourceRecord from a verified manifest row.

    Identity rule (campaign-local, documented): source_id derives from the
    MEASURED staged content sha — the same rule HfSourceAdapter uses when a
    remote blob sha exists, but stable even when the Hub gives none.
    """
    sha = row["sha256"]
    uri = row["source_uri"]
    return SourceRecord(
        source_id="SRC_" + short_id(sha, job.pinned_sha),
        source_uri=uri, source_revision=job.pinned_sha,
        original_file_id=row["original_file_id"],
        source_sha256=sha, source_bytes=int(row.get("size", 0)),
        staged_path=os.path.join(staging_dir, row.get("staged_path", "")),
        ingestion_timestamp_utc=utc_now_iso())


def qc_done_files(store: Any, batch: Any) -> List[str]:
    prog = store.batch_checkpoint(batch).load()
    return sorted(i for i, s in prog.states.items() if s in QC_DONE_STATES)


def run_qc_batch(store: Any, job_id: str, batch_id: str,
                 analyzer: Callable[..., Any],
                 workdir: Optional[str] = None) -> Dict[str, Any]:
    """Run audio QC over one downloaded batch. Per-file isolated; retryable.

    Analyzer contract (same as ``OrnixPipeline.analyze_source``): append its
    own evidence rows to ``paths.evidence`` and return an object with
    ``evidences``/``accepted`` (accepted items exposing ``to_dict()``).
    The orchestrator snapshots accepted rows atomically; it never re-appends
    evidence.
    """
    job = store.load_job(job_id)
    batch = store.load_batch(batch_id)
    if batch.job_id != job.job_id:
        raise ValueError(f"batch {batch_id} does not belong to job {job_id}")
    if batch.status != BatchStatus.IN_PROGRESS.value:
        return {"ok": False, "reason": f"batch-status-{batch.status}",
                "note": "qc needs a fully downloaded (IN_PROGRESS) batch"}
    staging = batch_staging_dir(store.root, job.job_id, batch.batch_id)
    rows = read_manifest(staging)
    if not rows:
        return {"ok": False, "reason": "no-manifest"}
    paths = qc_paths(store, job.job_id, batch.batch_id, workdir)
    audit = AuditLog(paths.audit, run_id=qc_run_id(job.job_id, batch.batch_id))
    ckpt = store.batch_checkpoint(batch)
    done = set(qc_done_files(store, batch))

    accepted: List[Dict[str, Any]] = []
    n_shard_blocked = 0
    pending: List[str] = []
    for row in rows:
        fid = row["original_file_id"]
        if fid in done:
            continue
        ext = os.path.splitext(fid)[1].lower()
        if ext not in AUDIO_EXTS:
            # No wired extractor for container shards: blocker, never raw QC.
            append_jsonl(paths.evidence, {
                "source_uri": row["source_uri"], "decision": "BLOCKED",
                "reason": f"unsupported shard container {ext or '(no ext)'}; "
                          f"needs a verified extractor (see plan R1/MD-004)"})
            ckpt.mark(fid, "SHARD_BLOCKED", reason="unsupported-container")
            audit.emit("qc_shard_blocked", fid, ext=ext)
            n_shard_blocked += 1
            continue
        src = manifest_to_source_record(row, job, staging)
        try:
            res = analyzer(src, paths, audit)
        except Exception as e:  # per-file isolation: one bad file ≠ dead batch
            ckpt.mark(fid, "QC_ERROR", error=str(e))
            audit.emit("qc_error", fid, error=str(e))
            pending.append(fid)
            continue
        for acc in (getattr(res, "accepted", None) or []):
            accepted.append(acc.to_dict() if hasattr(acc, "to_dict") else dict(acc))
        ckpt.mark(fid, "QC_DONE")
    # atomic accepted snapshot for this batch (the gate reads these files)
    prev = load_jsonl(os.path.join(paths.root, "BATCH_ACCEPTED.jsonl"))
    merged = {a.get("audio_id", f"row-{i}"): a
              for i, a in enumerate(prev + accepted)}
    atomic_write_text(
        os.path.join(paths.root, "BATCH_ACCEPTED.jsonl"),
        "".join(json.dumps(a, ensure_ascii=False, sort_keys=True) + "\n"
                for a in merged.values()))

    finished = set(qc_done_files(store, batch))
    # still_pending covers QC_ERROR files too (they carry no terminal mark),
    # so it is the single source of truth for what remains.
    still_pending = sorted(set(r["original_file_id"] for r in rows) - finished)
    n_evidence = len(load_jsonl(paths.evidence))
    summary = {"n_files": len(rows), "n_evidence": n_evidence,
               "n_accepted": len(merged), "n_shard_blocked": n_shard_blocked,
               "pending": still_pending}
    complete = not summary["pending"]
    if complete:
        batch.status = BatchStatus.BATCH_PROCESSED.value
    batch.result = {**(batch.result or {}), "qc": summary}
    store.save_batch(batch)
    audit.emit("qc_batch", batch.batch_id, **{k: v for k, v in summary.items()
                                              if k != "pending"})
    return {"ok": complete, "reason": "complete" if complete else "incomplete",
            **summary}


def gate_receipt_path(store: Any, job_id: str) -> str:
    return os.path.join(store.root, "releases", f"{job_id}.GATE.json")


def gate_batch_release(store: Any, job_id: str, batch_id: str,
                       ratios: tuple = (0.8, 0.1, 0.1)) -> Dict[str, Any]:
    """Promote BATCH_PROCESSED → RELEASE_READY iff job-global evidence is whole.

    Refuses (fail-closed) when any sibling batch lacks accepted rows, when the
    split assignment leaks across speaker groups, and reports exact content
    duplicates without dropping them.
    """
    job = store.load_job(job_id)
    batch = store.load_batch(batch_id)
    if batch.job_id != job.job_id:
        raise ValueError(f"batch {batch_id} does not belong to job {job_id}")
    if batch.status != BatchStatus.BATCH_PROCESSED.value:
        return {"ok": False, "reason": f"batch-status-{batch.status}"}

    per_batch: Dict[str, List[Dict[str, Any]]] = {}
    missing: List[str] = []
    for bid in job.batch_ids:
        b = store.load_batch(bid)
        acc_file = os.path.join(
            qc_paths(store, job.job_id, b.batch_id).root, "BATCH_ACCEPTED.jsonl")
        rows = load_jsonl(acc_file)
        if b.status not in (BatchStatus.BATCH_PROCESSED.value,
                            BatchStatus.RELEASE_READY.value,
                            BatchStatus.DONE.value) or not os.path.exists(acc_file):
            missing.append(bid)
        per_batch[bid] = rows
    if missing:
        return {"ok": False, "reason": "global-evidence-incomplete",
                "missing_batches": missing,
                "note": "release-ready needs every batch processed first"}

    group_keys: Dict[str, str] = {}
    sha_by_id: Dict[str, str] = {}
    for rows in per_batch.values():
        for r in rows:
            aid = r.get("audio_id") or r.get("source_id")
            group_keys[aid] = r.get("speaker_id") or r.get("source_id") or aid
            sha_by_id[aid] = (r.get("source_sha256")
                              or r.get("audio_sha256") or "")
    assignment = assign_splits(group_keys, ratios=ratios)
    leak = split_leakage(group_keys, assignment)
    if leak:
        return {"ok": False, "reason": "split-leakage", "groups": leak}
    duplicates = exact_duplicates(sha_by_id) if sha_by_id else []

    receipt = {"job_id": job.job_id, "repo_id": job.repo_id,
               "pinned_sha": job.pinned_sha,
               "n_accepted": len(group_keys),
               "assignment": assignment, "duplicates": duplicates,
               "gated_batch": batch.batch_id, "gated_utc": utc_now_iso()}
    os.makedirs(os.path.dirname(gate_receipt_path(store, job.job_id)),
                exist_ok=True)
    write_json(gate_receipt_path(store, job.job_id), receipt)
    batch.status = BatchStatus.RELEASE_READY.value
    batch.result = {**(batch.result or {}),
                    "release_gate": {"ok": True,
                                     "n_accepted_job": len(group_keys),
                                     "n_duplicate_groups": len(duplicates),
                                     "receipt": gate_receipt_path(
                                         store, job.job_id)}}
    store.save_batch(batch)
    return {"ok": True, "batch": batch.batch_id, "n_accepted": len(group_keys),
            "duplicates": duplicates,
            "receipt": gate_receipt_path(store, job.job_id)}
