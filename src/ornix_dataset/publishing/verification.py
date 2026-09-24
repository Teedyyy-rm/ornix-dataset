"""Remote readback verification after upload (spec §7, Phase 7, T-014).

Compares the remote file tree at a pinned commit against the local MANIFEST and
downloads a sample to confirm byte integrity. A publish is only PUBLISHED_VERIFIED
if this passes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RemoteVerifyResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    n_remote_files: int = 0
    checked_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors,
                "n_remote_files": self.n_remote_files, "checked_samples": self.checked_samples}


def remote_verify(api: Any, repo_id: str, revision: str, release_dir: str,
                  sample: int = 3) -> RemoteVerifyResult:  # pragma: no cover - network
    errors: List[str] = []
    try:
        remote_files = set(api.list_repo_files(repo_id, repo_type="dataset", revision=revision))
    except Exception as e:
        return RemoteVerifyResult(False, [f"list_repo_files failed: {e}"])

    local_files = []
    for dp, _d, fs in os.walk(release_dir):
        for f in fs:
            local_files.append(os.path.relpath(os.path.join(dp, f), release_dir))
    for rel in local_files:
        if rel not in remote_files:
            errors.append(f"REMOTE_MISSING:{rel}")

    from huggingface_hub import hf_hub_download
    from ..util.hashing import sha256_file

    checked = 0
    for rel in sorted(local_files)[:sample]:
        try:
            got = hf_hub_download(repo_id, rel, repo_type="dataset", revision=revision)
            if sha256_file(got) != sha256_file(os.path.join(release_dir, rel)):
                errors.append(f"REMOTE_SHA_MISMATCH:{rel}")
            checked += 1
        except Exception as e:
            errors.append(f"REMOTE_READBACK_FAILED:{rel}:{e}")
    return RemoteVerifyResult(not errors, errors, len(remote_files), checked)
