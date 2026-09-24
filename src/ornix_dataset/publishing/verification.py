"""Remote readback verification after upload (spec §7, Phase 7, T-014).

Compares the remote file tree at a pinned commit against the local MANIFEST and
downloads a sample to confirm byte integrity. A publish is only PUBLISHED_VERIFIED
if this passes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RemoteVerifyResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    n_remote_files: int = 0
    checked_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors,
                "n_remote_files": self.n_remote_files, "checked_samples": self.checked_samples}


# HF creates/keeps these repo-internal files; they are not part of our MANIFEST.
_REMOTE_IGNORE = {".gitattributes"}


def remote_verify(api: Any, repo_id: str, revision: str, release_dir: str,
                  sample: int = 3, full_hash: bool = False,
                  ignore: set | None = None,
                  path_prefix: Optional[str] = None) -> RemoteVerifyResult:
    """Verify the remote tree at ``revision`` byte-for-byte against the local release.

    Uses an EXACT set comparison (not subset): a remote file that is not in the
    local release — a stale shard from a previous push, a stray upload — is a
    failure, because a consumer would receive bytes we never validated. With
    ``full_hash`` every file is re-downloaded and hashed; otherwise a deterministic
    sample is checked.

    ``path_prefix`` scopes the comparison to one namespaced subtree
    (``releases/<id>/…``): remote files outside the prefix belong to sibling
    releases and are ignored instead of flagged. ``None`` keeps the legacy
    whole-repo comparison.
    """
    ignore = _REMOTE_IGNORE if ignore is None else ignore
    prefix = (path_prefix or "").strip().strip("/")
    errors: List[str] = []
    try:
        remote_files = set(api.list_repo_files(repo_id, repo_type="dataset", revision=revision))
    except Exception as e:
        return RemoteVerifyResult(False, [f"list_repo_files failed: {e}"])
    if prefix:
        remote_files = {f[len(prefix) + 1:] for f in remote_files
                        if f == prefix or f.startswith(prefix + "/")}
    remote_files = {f for f in remote_files if f not in ignore}

    local_files = []
    for dp, _d, fs in os.walk(release_dir):
        for f in fs:
            local_files.append(os.path.relpath(os.path.join(dp, f), release_dir))
    local_set = set(local_files)
    for rel in sorted(local_set - remote_files):
        errors.append(f"REMOTE_MISSING:{rel}")
    for rel in sorted(remote_files - local_set):
        errors.append(f"EXTRA_REMOTE_FILE:{rel}")

    from huggingface_hub import hf_hub_download
    from ..util.hashing import sha256_file

    def remote_name(rel: str) -> str:
        return f"{prefix}/{rel}" if prefix else rel

    to_check = sorted(local_files) if full_hash else sorted(local_files)[:sample]
    checked = 0
    for rel in to_check:
        try:
            got = hf_hub_download(repo_id, remote_name(rel), repo_type="dataset",
                                  revision=revision)
            if sha256_file(got) != sha256_file(os.path.join(release_dir, rel)):
                errors.append(f"REMOTE_SHA_MISMATCH:{rel}")
            checked += 1
        except Exception as e:
            errors.append(f"REMOTE_READBACK_FAILED:{rel}:{e}")
    return RemoteVerifyResult(not errors, errors, len(remote_files), checked)
