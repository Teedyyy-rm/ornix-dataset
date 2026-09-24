"""Publish approval receipt (spec §7, Phase 7).

A real publish requires an approval file that binds: the exact release digest,
the destination repo, allowed write scope, a byte quota, the acknowledged
license/policy, and an operator id + expiry. Any mismatch or expiry => STOP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from ..util.hashing import sha256_file


@dataclass
class ApprovalReceipt:
    release_digest: str
    repo_id: str
    revision: str
    max_bytes: int
    operator_id: str
    expires_utc: str
    policy_version: str
    license_ack: bool
    allow_create_repo: bool = False
    raw: Dict[str, Any] = None


def release_digest(release_dir: str) -> str:
    """Digest that identifies a built release (hash of its MANIFEST.sha256)."""
    manifest = os.path.join(release_dir, "MANIFEST.sha256")
    if not os.path.exists(manifest):
        raise FileNotFoundError("release has no MANIFEST.sha256; build it first")
    return sha256_file(manifest)


def load_approval(path: str) -> ApprovalReceipt:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    required = ["release_digest", "repo_id", "revision", "max_bytes", "operator_id",
                "expires_utc", "policy_version", "license_ack"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"approval file missing fields: {missing}")
    return ApprovalReceipt(
        release_digest=raw["release_digest"], repo_id=raw["repo_id"],
        revision=raw["revision"], max_bytes=int(raw["max_bytes"]),
        operator_id=raw["operator_id"], expires_utc=raw["expires_utc"],
        policy_version=raw["policy_version"], license_ack=bool(raw["license_ack"]),
        allow_create_repo=bool(raw.get("allow_create_repo", False)), raw=raw)


def _dir_bytes(path: str) -> int:
    total = 0
    for dp, _d, fs in os.walk(path):
        for f in fs:
            total += os.path.getsize(os.path.join(dp, f))
    return total


def validate_approval(receipt: ApprovalReceipt, release_dir: str,
                      repo_id: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    try:
        digest = release_digest(release_dir)
    except FileNotFoundError as e:
        return False, [str(e)]
    if digest != receipt.release_digest:
        reasons.append("RELEASE_DIGEST_MISMATCH")
    if receipt.repo_id != repo_id:
        reasons.append("REPO_ID_MISMATCH")
    if not receipt.license_ack:
        reasons.append("LICENSE_NOT_ACKNOWLEDGED")
    try:
        exp = datetime.strptime(receipt.expires_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > exp:
            reasons.append("APPROVAL_EXPIRED")
    except ValueError:
        reasons.append("APPROVAL_EXPIRY_UNPARSEABLE")
    if _dir_bytes(release_dir) > receipt.max_bytes:
        reasons.append("RELEASE_EXCEEDS_BYTE_QUOTA")
    if not os.path.exists(os.path.join(release_dir, "RELEASE_READY.json")):
        reasons.append("RELEASE_NOT_READY")
    return (not reasons), reasons
