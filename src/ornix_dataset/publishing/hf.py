"""Staged HF publisher (spec §7, Phase 7).

Dry-run by default: enumerates the upload plan and never touches the network. A
real publish requires a valid approval receipt AND a token from the environment
(never scraped from logs). Uploads to a staging revision, verifies remotely, and
only then writes the PUBLISHED_VERIFIED marker. Never deletes prior history.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..util.io import write_json
from ..util.timeutil import utc_now_iso
from .approval import load_approval, validate_approval, release_digest


class PublishStatus(str, Enum):
    DRY_RUN = "DRY_RUN"
    BLOCKED = "BLOCKED"
    UPLOAD_IN_PROGRESS = "UPLOAD_IN_PROGRESS"
    REMOTE_VERIFY_FAILED = "REMOTE_VERIFY_FAILED"
    PUBLISHED_VERIFIED = "PUBLISHED_VERIFIED"


@dataclass
class PublishResult:
    status: PublishStatus
    repo_id: str
    reasons: List[str] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    remote_commit_sha: Optional[str] = None
    report: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status.value, "repo_id": self.repo_id,
                "reasons": self.reasons, "plan": self.plan,
                "remote_commit_sha": self.remote_commit_sha, "report": self.report}


class StagedPublisher:
    def __init__(self, token_env: str = "HF_TOKEN"):
        self.token_env = token_env

    def _plan(self, release_dir: str) -> Dict[str, Any]:
        files, total = [], 0
        for dp, _d, fs in os.walk(release_dir):
            for f in fs:
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, release_dir)
                size = os.path.getsize(full)
                files.append({"path": rel, "bytes": size})
                total += size
        return {"n_files": len(files), "total_bytes": total,
                "files": sorted(files, key=lambda x: x["path"])}

    def publish(self, release_dir: str, repo_id: str, dry_run: bool = True,
                approval_path: Optional[str] = None,
                staging_revision: str = "refs/pr/ornix-staging") -> PublishResult:
        plan = self._plan(release_dir)
        if dry_run or approval_path is None:
            return PublishResult(PublishStatus.DRY_RUN, repo_id,
                                 reasons=["dry-run default; no remote mutation"], plan=plan)

        receipt = load_approval(approval_path)
        ok, reasons = validate_approval(receipt, release_dir, repo_id)
        if not ok:
            return PublishResult(PublishStatus.BLOCKED, repo_id, reasons=reasons, plan=plan)

        token = os.environ.get(self.token_env)
        if not token:
            return PublishResult(PublishStatus.BLOCKED, repo_id,
                                 reasons=[f"no token in ${self.token_env}"], plan=plan)
        try:
            return self._do_publish(release_dir, repo_id, receipt, token, staging_revision, plan)
        except Exception as e:  # pragma: no cover - network dependent
            return PublishResult(PublishStatus.REMOTE_VERIFY_FAILED, repo_id,
                                 reasons=[f"publish error: {e}"], plan=plan)

    def _do_publish(self, release_dir, repo_id, receipt, token, staging_revision, plan
                    ) -> PublishResult:  # pragma: no cover - network dependent
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.repo_info(repo_id, repo_type="dataset")  # preflight; raises if missing/unauthorized
        try:
            api.create_branch(repo_id, branch=staging_revision, repo_type="dataset",
                              exist_ok=True)
        except Exception:
            pass
        commit = api.upload_folder(
            folder_path=release_dir, repo_id=repo_id, repo_type="dataset",
            revision=staging_revision,
            commit_message=f"ornix staged release {receipt.release_digest[:12]}",
        )
        sha = getattr(commit, "oid", None) or getattr(commit, "commit_sha", None)
        from .verification import remote_verify

        rv = remote_verify(api, repo_id, sha or staging_revision, release_dir)
        if not rv.ok:
            return PublishResult(PublishStatus.REMOTE_VERIFY_FAILED, repo_id,
                                 reasons=rv.errors, plan=plan, remote_commit_sha=sha,
                                 report=rv.to_dict())
        marker = {"status": "PUBLISHED_VERIFIED", "repo_id": repo_id,
                  "remote_commit_sha": sha, "release_digest": receipt.release_digest,
                  "operator_id": receipt.operator_id, "verified_utc": utc_now_iso()}
        write_json(os.path.join(release_dir, "PUBLISHED_VERIFIED.json"), marker)
        return PublishResult(PublishStatus.PUBLISHED_VERIFIED, repo_id, plan=plan,
                             remote_commit_sha=sha, report=rv.to_dict())
