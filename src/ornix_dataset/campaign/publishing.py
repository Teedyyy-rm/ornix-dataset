"""Incremental publisher: RELEASE_READY batch → namespaced release (MD-005).

Naming (mandatory, plan REVIEW-EDIT 3): every release owns a namespace —
``releases/<release_id>/`` inside the destination repo and a staging branch
``ornix/<release_id>`` — so many releases coexist without overwriting each
other. ``StagedPublisher`` itself is reused; this module supplies the release
content, the namespace, and the remote-verification receipt.

``prepare_release`` builds the release dir from the batch's accepted rows
(split column applied from the MD-004 gate receipt, canonical audio copied
from the batch QC run). ``publish_release`` uploads with ``full_hash`` on
request (mandatory before any cleanup deletes local bytes — Gate 5 rule) and
writes the REMOTE_VERIFIED receipt BESIDE the release dir.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..exporters import build_release
from ..publishing.approval import release_digest
from ..publishing.hf import StagedPublisher, marker_path_for
from ..util.hashing import sha256_file
from ..util.io import read_json, write_json
from ..util.jsonl import load_jsonl
from ..util.timeutil import utc_now_iso
from .models import BatchStatus
from .preflight import oversized_files
from .processing import gate_receipt_path, qc_paths

RELEASES_DIR = "releases"


def _dir_bytes(path: str) -> int:
    total = 0
    for dp, _ds, fs in os.walk(path):
        for f in fs:
            total += os.path.getsize(os.path.join(dp, f))
    return total


def release_id_for(campaign_id: str, job_id: str, batch_id: str) -> str:
    for comp in (campaign_id, job_id, batch_id):
        if not comp or "/" in comp or comp.startswith("."):
            raise ValueError(f"unsafe release component: {comp!r}")
    return f"{campaign_id}--{job_id}--{batch_id}"


def staging_revision_for(release_id: str) -> str:
    return f"ornix/{release_id}"


def path_prefix_for(release_id: str) -> str:
    return f"{RELEASES_DIR}/{release_id}"


def release_dir_for(store: Any, release_id: str) -> str:
    return os.path.join(store.root, RELEASES_DIR, release_id)


def receipt_path_for(store: Any, release_id: str) -> str:
    return os.path.join(store.root, RELEASES_DIR,
                        f"{release_id}.REMOTE_VERIFIED.json")


def record_path_for(store: Any, release_id: str) -> str:
    return os.path.join(store.root, RELEASES_DIR,
                        f"{release_id}.RELEASE.json")


def prepare_release(store: Any, job_id: str, batch_id: str,
                    release_target: str = "train_only",
                    export_format: str = "parquet") -> Dict[str, Any]:
    """Build a namespaced release dir from one RELEASE_READY batch."""
    campaign = store.load_campaign()
    job = store.load_job(job_id)
    batch = store.load_batch(batch_id)
    if batch.job_id != job.job_id:
        raise ValueError(f"batch {batch_id} does not belong to job {job_id}")
    if batch.status != BatchStatus.RELEASE_READY.value:
        return {"ok": False, "reason": f"batch-status-{batch.status}",
                "note": "only RELEASE_READY batches can be released"}
    acc_file = os.path.join(
        qc_paths(store, job.job_id, batch.batch_id).root, "BATCH_ACCEPTED.jsonl")
    rows = load_jsonl(acc_file)
    if not rows:
        return {"ok": False, "reason": "no-accepted-rows"}
    gate = read_json(gate_receipt_path(store, job.job_id))
    assignment = gate.get("assignment", {})
    for r in rows:
        aid = r.get("audio_id") or r.get("source_id")
        if aid in assignment:
            r["split"] = assignment[aid]

    release_id = release_id_for(campaign.campaign_id, job.job_id,
                                batch.batch_id)
    out = release_dir_for(store, release_id)
    canonical = qc_paths(store, job.job_id, batch.batch_id).canonical_dir
    rights_report = {
        "licenses": sorted({r.get("source_license", "UNKNOWN") for r in rows}),
        "rights_status": sorted({r.get("rights_status", "UNKNOWN") for r in rows}),
        "n_sources": len({r.get("source_id") for r in rows}),
        "release_target": release_target,
    }
    art = build_release(release_id, rows, canonical, out,
                        rights_report=rights_report,
                        quality_report={"note": "campaign batch release; "
                                                "see BATCH evidence in QC run"},
                        export_format=export_format,
                        release_target=release_target)
    oversized = oversized_files(out)
    if oversized:
        art.ready = False
        art.blockers = list(art.blockers) + [
            f"HUB_FILE_HARD_LIMIT:{o['path']}:{o['bytes']}" for o in oversized]
    record = {"release_id": release_id, "campaign_id": campaign.campaign_id,
              "job_id": job.job_id, "batch_id": batch.batch_id,
              "repo_target": (campaign.destination or {}).get("repo_id"),
              "revision": staging_revision_for(release_id),
              "path_prefix": path_prefix_for(release_id),
              "ready": bool(art.ready), "blockers": list(art.blockers),
              "n_rows": art.n_rows, "release_dir": out,
              "digest": (release_digest(out) if art.ready else None),
              "prepared_utc": utc_now_iso()}
    write_json(record_path_for(store, release_id), record)
    batch.result = {**(batch.result or {}), "release": record}
    store.save_batch(batch)
    return {"ok": bool(art.ready), **record}


def release_file_inventory(release_dir: str) -> Dict[str, str]:
    """Exact {relpath: sha256} inventory of a release dir (for intactness)."""
    inv: Dict[str, str] = {}
    for dp, _ds, fs in os.walk(release_dir):
        for f in fs:
            full = os.path.join(dp, f)
            inv[os.path.relpath(full, release_dir)] = sha256_file(full)
    return inv


def publish_release(store: Any, release_id: str, repo_id: str,
                    approval_path: str, full_hash: bool = False,
                    upload_retries: int = 3,
                    destination: Optional[Dict[str, Any]] = None,
                    api: Any = None) -> Dict[str, Any]:
    """Upload a prepared release; write the REMOTE_VERIFIED receipt on success.

    If ``destination`` declares ``max_bytes``, the destination's real size is
    measured first and the upload is refused when it would exceed the quota
    (G3) — a second, per-release enforcement of the campaign-level preflight.
    """
    record = read_json(record_path_for(store, release_id))
    release_dir = record["release_dir"]
    if destination and destination.get("max_bytes") is not None:
        from .preflight import check_destination, repo_current_bytes

        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ.get("HF_TOKEN"))
        add = _dir_bytes(release_dir)
        try:
            pf = check_destination(api, destination, projected_bytes=add,
                                   repo_id=repo_id)
        except Exception as e:
            return {"ok": False, "status": "BLOCKED",
                    "reasons": [f"DESTINATION_PREFLIGHT_FAILED:{e}"]}
        if not pf.ok:
            return {"ok": False, "status": "BLOCKED", "reasons": pf.reasons,
                    "preflight": pf.to_dict()}
    pub = StagedPublisher()
    res = pub.publish(release_dir, repo_id, dry_run=False,
                      approval_path=approval_path,
                      staging_revision=record["revision"],
                      path_in_repo=record["path_prefix"],
                      upload_retries=upload_retries,
                      full_hash=full_hash)
    report = {"ok": res.status.value == "PUBLISHED_VERIFIED",
              "status": res.status.value, "reasons": res.reasons,
              "remote_commit_sha": res.remote_commit_sha,
              "report": res.report}
    if res.status.value == "PUBLISHED_VERIFIED":
        receipt = {"release_id": release_id,
                   "job_id": record["job_id"], "batch_id": record["batch_id"],
                   "repo_id": repo_id, "revision": record["revision"],
                   "path_prefix": record["path_prefix"],
                   "remote_commit_sha": res.remote_commit_sha,
                   "release_digest": record["digest"],
                   "files": release_file_inventory(release_dir),
                   "n_files": res.plan.get("n_files", 0),
                   "full_hash_verified": bool(full_hash),
                   "marker": marker_path_for(release_dir),
                   "verified_utc": utc_now_iso()}
        write_json(receipt_path_for(store, release_id), receipt)
        report["receipt"] = receipt_path_for(store, release_id)
        batch = store.load_batch(record["batch_id"])
        batch.result = {**(batch.result or {}), "publish": report}
        store.save_batch(batch)
    return report


def load_receipt(store: Any, release_id: str) -> Dict[str, Any]:
    return read_json(receipt_path_for(store, release_id))


def list_releases(store: Any) -> List[Dict[str, Any]]:
    out = []
    d = os.path.join(store.root, RELEASES_DIR)
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".RELEASE.json"):
            out.append(read_json(os.path.join(d, fn)))
    return out
