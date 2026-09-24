"""Publish the canonical Ornix Dataset, reusing the existing gated publisher.

The canonical tree is finalized (card + checksum manifest + READY marker) so the
existing approval receipt (bound to the release digest + destination revision),
the exact-set remote verification and the ``PUBLISHED_VERIFIED`` marker all work
unchanged. A marker is written only after a full remote readback succeeds.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..publishing.approval import release_digest
from ..publishing.hf import StagedPublisher, marker_path_for
from ..util.hashing import sha256_file
from ..util.io import write_json
from ..util.timeutil import utc_now_iso
from . import card, layout


def finalize_dataset(dataset_dir: str, rights: Optional[Dict[str, Any]] = None,
                     changelog: str = "") -> Dict[str, Any]:
    """(Re)write card + checksum manifest + READY marker from current rows."""
    stats = layout.split_stats(dataset_dir)
    if rights is None:
        # preserve the rights summary recorded at export time rather than wiping
        # the card's license/attribution section on a bare re-finalize.
        ready = os.path.join(dataset_dir, "RELEASE_READY.json")
        rights = {}
        if os.path.exists(ready):
            try:
                from ..util.io import read_json
                rights = (read_json(ready) or {}).get("rights", {}) or {}
            except Exception:
                rights = {}
    if stats["n_rows"] == 0:
        return {"ok": False, "reason": "no-rows"}
    card.write_card(dataset_dir, stats, rights, changelog)
    layout.write_manifest_sha(dataset_dir)
    layout.write_ready(dataset_dir, stats, rights)
    return {"ok": True, **stats}


def dataset_inventory(dataset_dir: str) -> Dict[str, str]:
    inv: Dict[str, str] = {}
    for dp, _dirs, files in os.walk(dataset_dir):
        for fn in files:
            full = os.path.join(dp, fn)
            inv[os.path.relpath(full, dataset_dir)] = sha256_file(full)
    return inv


def receipt_path_for(dataset_dir: str) -> str:
    parent = os.path.dirname(os.path.abspath(dataset_dir.rstrip("/")))
    base = os.path.basename(os.path.abspath(dataset_dir.rstrip("/")))
    return os.path.join(parent, f"{base}.CANONICAL_PUBLISHED.json")


def publish_dataset(dataset_dir: str, repo_id: str, approval_path: str,
                    dry_run: bool = False, full_hash: bool = False,
                    upload_retries: int = 3, staging_revision: str = "main",
                    token_env: str = "HF_TOKEN") -> Dict[str, Any]:
    """Upload the canonical dataset through the staged publisher (gated + verified)."""
    if not os.path.exists(os.path.join(dataset_dir, "MANIFEST.sha256")) \
            or not os.path.exists(os.path.join(dataset_dir, "RELEASE_READY.json")):
        fin = finalize_dataset(dataset_dir)
        if not fin.get("ok"):
            return {"ok": False, "status": "BLOCKED", "reasons": ["NOT_FINALIZED"]}
    pub = StagedPublisher(token_env=token_env)
    res = pub.publish(dataset_dir, repo_id, dry_run=dry_run,
                      approval_path=approval_path,
                      staging_revision=staging_revision,
                      path_in_repo=None, upload_retries=upload_retries,
                      full_hash=full_hash)
    report: Dict[str, Any] = {"ok": res.status.value == "PUBLISHED_VERIFIED",
                              "status": res.status.value, "reasons": res.reasons,
                              "remote_commit_sha": res.remote_commit_sha,
                              "report": res.report}
    if res.status.value == "PUBLISHED_VERIFIED":
        receipt = {"repo_id": repo_id, "revision": staging_revision,
                   "remote_commit_sha": res.remote_commit_sha,
                   "release_digest": release_digest(dataset_dir),
                   "files": dataset_inventory(dataset_dir),
                   "n_files": res.plan.get("n_files", 0),
                   "full_hash_verified": bool(full_hash),
                   "marker": marker_path_for(dataset_dir),
                   "verified_utc": utc_now_iso()}
        write_json(receipt_path_for(dataset_dir), receipt)
        report["receipt"] = receipt_path_for(dataset_dir)
    return report
