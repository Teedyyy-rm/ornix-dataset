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
    UPLOAD_FAILED = "UPLOAD_FAILED"  # transport/publish error (retryable)
    REMOTE_VERIFY_FAILED = "REMOTE_VERIFY_FAILED"  # content mismatch (inspect)
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


def marker_path_for(release_dir: str) -> str:
    """PUBLISHED_VERIFIED marker lives BESIDE the release dir, never inside it:
    writing into the dir after verification would invalidate the exact-set
    remote comparison on the next verify."""
    parent = os.path.dirname(os.path.abspath(release_dir.rstrip("/")))
    base = os.path.basename(os.path.abspath(release_dir.rstrip("/")))
    return os.path.join(parent, f"{base}.PUBLISHED_VERIFIED.json")


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
                staging_revision: str = "refs/pr/ornix-staging",
                path_in_repo: Optional[str] = None,
                upload_retries: int = 3,
                full_hash: bool = False) -> PublishResult:
        """Publish a release dir. ``path_in_repo`` namespaces the upload inside
        the destination repo so many releases coexist (MD-005); ``None`` keeps
        the legacy repo-root layout. ``full_hash`` re-downloads every file on
        verify (required before any cleanup deletes local bytes)."""
        plan = self._plan(release_dir)
        if dry_run or approval_path is None:
            return PublishResult(PublishStatus.DRY_RUN, repo_id,
                                 reasons=["dry-run default; no remote mutation"], plan=plan)

        receipt = load_approval(approval_path)
        ok, reasons = validate_approval(receipt, release_dir, repo_id,
                                        revision=staging_revision)
        if not ok:
            return PublishResult(PublishStatus.BLOCKED, repo_id, reasons=reasons, plan=plan)

        token = os.environ.get(self.token_env)
        if not token:
            return PublishResult(PublishStatus.BLOCKED, repo_id,
                                 reasons=[f"no token in ${self.token_env}"], plan=plan)
        try:
            return self._do_publish(release_dir, repo_id, receipt, token,
                                    staging_revision, plan,
                                    path_in_repo=path_in_repo,
                                    upload_retries=max(1, upload_retries),
                                    full_hash=full_hash)
        except Exception as e:  # pragma: no cover - network dependent
            return PublishResult(PublishStatus.UPLOAD_FAILED, repo_id,
                                 reasons=[f"publish error: {e}"], plan=plan)

    def _do_publish(self, release_dir, repo_id, receipt, token, staging_revision, plan,
                    path_in_repo=None, upload_retries=3, full_hash=False,
                    ) -> PublishResult:  # pragma: no cover - network dependent
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.repo_info(repo_id, repo_type="dataset")  # preflight; raises if missing/unauthorized
        try:
            api.create_branch(repo_id, branch=staging_revision, repo_type="dataset",
                              exist_ok=True)
        except Exception:
            pass
        attempts, last_error = 0, None
        commit = None
        # Re-running the same upload is the resume mechanism: committed data
        # and duplicate chunks are reused server-side, so a retry continues
        # rather than restarting. Only transport errors retry; a successful
        # commit with bad content must surface as VERIFY_FAILED, never retry.
        for attempts in range(1, upload_retries + 1):
            try:
                commit = api.upload_folder(
                    folder_path=release_dir, repo_id=repo_id, repo_type="dataset",
                    revision=staging_revision,
                    path_in_repo=path_in_repo,
                    commit_message=f"ornix staged release {receipt.release_digest[:12]}",
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
        if last_error is not None:
            return PublishResult(PublishStatus.UPLOAD_FAILED, repo_id,
                                 reasons=[f"upload failed after {attempts} "
                                          f"attempt(s): {last_error}"],
                                 plan=plan,
                                 report={"upload_attempts": attempts})
        sha = getattr(commit, "oid", None) or getattr(commit, "commit_sha", None)
        from .verification import remote_verify

        rv = remote_verify(api, repo_id, sha or staging_revision, release_dir,
                           path_prefix=path_in_repo, full_hash=full_hash)
        if not rv.ok:
            return PublishResult(PublishStatus.REMOTE_VERIFY_FAILED, repo_id,
                                 reasons=rv.errors, plan=plan, remote_commit_sha=sha,
                                 report={**rv.to_dict(),
                                         "upload_attempts": attempts})
        marker = {"status": "PUBLISHED_VERIFIED", "repo_id": repo_id,
                  "remote_commit_sha": sha, "release_digest": receipt.release_digest,
                  "operator_id": receipt.operator_id, "verified_utc": utc_now_iso(),
                  "path_in_repo": path_in_repo, "upload_attempts": attempts}
        write_json(marker_path_for(release_dir), marker)
        return PublishResult(PublishStatus.PUBLISHED_VERIFIED, repo_id, plan=plan,
                             remote_commit_sha=sha,
                             report={**rv.to_dict(), "upload_attempts": attempts})
