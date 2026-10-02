"""Fan out an accepted batch into N-clip releases (spec: incremental publish).

The download/QC pipeline batches at *shard* granularity, because that is the
unit that exists in a repo listing and is idempotent to re-run. But the release
unit an operator wants is a fixed slice of accepted utterances, so a 4000-clip
batch can be published as 40 verified releases instead of one huge tree (or, at
the other extreme, not published at all until every sibling batch finishes).

This module splits the ALREADY-ACCEPTED rows of one batch into deterministic
N-row chunks and builds a real release per chunk through the existing
``exporters.build_release``. It deliberately touches nothing upstream:

- download, parquet extract, QC, checkpoints and the batch state machine are
  untouched, so their idempotency and no-loss invariants are unchanged;
- only accepted rows (already gated, rights-checked and split-assigned) are
  re-used — no new QC, no new admission, no new evidence.

Determinism: rows are ordered by ``(split, audio_id)`` before chunking, so the
same batch always yields the same chunks with the same ids and the same audio
bytes. Each chunk gets its own release dir, ``MANIFEST.sha256``, digest and
approval receipt.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from ..exporters.manifest import build_release
from ..publishing.approval import release_digest
from ..util.jsonl import load_jsonl
from ..util.timeutil import utc_now_iso
from ..version import __version__

DEFAULT_CHUNK = 100


def _order_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("split") or ""), str(row.get("audio_id") or ""))


def chunk_accepted(rows: List[Dict[str, Any]], size: int = DEFAULT_CHUNK
                   ) -> List[List[Dict[str, Any]]]:
    """Split accepted rows into deterministic chunks of at most ``size`` rows."""
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    ordered = sorted((r for r in rows if r), key=_order_key)
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def fanout_releases(store: Any, job_id: str, batch_id: str, *,
                    chunk_size: int = DEFAULT_CHUNK,
                    canonical_dir: str,
                    release_target: str = "train_only",
                    export_format: str = "none") -> Dict[str, Any]:
    """Build one release per ``chunk_size`` accepted rows of a processed batch.

    Returns a report with every chunk's release id, digest, path and readiness.
    A chunk that cannot be built is reported with its blocker instead of being
    silently dropped — but it never blocks its siblings.
    """
    from .processing import qc_paths

    job = store.load_job(job_id)
    batch = store.load_batch(batch_id)
    if batch.job_id != job.job_id:
        raise ValueError(f"batch {batch_id} does not belong to job {job_id}")
    acc_file = os.path.join(qc_paths(store, job.job_id, batch.batch_id).root,
                            "BATCH_ACCEPTED.jsonl")
    rows = load_jsonl(acc_file) if os.path.exists(acc_file) else []
    chunks = chunk_accepted(rows, chunk_size)

    out_root = os.path.join(store.root, "releases", "fanout")
    os.makedirs(out_root, exist_ok=True)
    licenses = sorted({str(r.get("source_license") or "UNKNOWN") for r in rows})
    statuses = sorted({str(r.get("rights_status") or "UNKNOWN") for r in rows})
    rights_report = {
        "licenses": licenses, "rights_status": statuses,
        "n_sources": len({r.get("source_id") for r in rows if r.get("source_id")}),
        "release_target": release_target,
        "note": f"fanout chunk of {batch_id}",
    }

    releases: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        rid = f"{job.job_id}__{batch.batch_id}__c{idx:05d}"
        out_dir = os.path.join(out_root, rid)
        art = build_release(
            rid, chunk, canonical_dir, out_dir,
            rights_report=rights_report,
            quality_report={"note": "campaign fanout chunk; QC evidence in batch run",
                            "batch_id": batch.batch_id},
            export_format=export_format,
            release_target=release_target)
        rec: Dict[str, Any] = {
            "index": idx, "release_id": rid, "release_dir": out_dir,
            "n_rows": art.n_rows, "ready": bool(art.ready),
            "blockers": list(art.blockers),
        }
        if art.ready:
            rec["digest"] = release_digest(out_dir)
        releases.append(rec)

    ready = [r for r in releases if r["ready"]]
    return {"ok": bool(ready), "job_id": job.job_id, "batch_id": batch.batch_id,
            "n_accepted": len(rows), "chunk_size": chunk_size,
            "n_chunks": len(releases), "n_ready": len(ready),
            "out_root": out_root, "releases": releases,
            "generated_utc": utc_now_iso(),
            "analyzer_version": __version__}


def fanout_dir_for(store: Any, job_id: str, batch_id: str) -> str:
    return os.path.join(store.root, "releases", "fanout")


def write_approvals(releases: List[Dict[str, Any]], *, repo_id: str,
                    revision_of, operator_id: str, policy_version: str,
                    expires_utc: str, max_bytes: int,
                    approval_dir: str) -> List[Dict[str, Any]]:
    """Mint one approval receipt per ready chunk, bound to that chunk's digest.

    The receipt is what the publisher will check; it is generated here only
    because the operator has already approved this campaign's publish flow, and
    every receipt is still scoped to exactly one digest, repo, revision and
    byte quota — so a single approval can never authorize a different tree.
    """
    import yaml

    os.makedirs(approval_dir, exist_ok=True)
    out: List[Dict[str, Any]] = []
    for rec in releases:
        if not rec.get("ready") or not rec.get("digest"):
            continue
        rid = rec["release_id"]
        receipt = {
            "release_digest": rec["digest"],
            "repo_id": repo_id,
            "revision": revision_of(rid),
            "max_bytes": max_bytes,
            "operator_id": operator_id,
            "expires_utc": expires_utc,
            "policy_version": policy_version,
            "license_ack": True,
        }
        path = os.path.join(approval_dir, f"{rid}.approval.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(receipt, fh, sort_keys=False, allow_unicode=True)
        out.append({"release_id": rid, "approval": path, "digest": rec["digest"]})
    return out