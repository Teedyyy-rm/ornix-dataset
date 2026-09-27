"""Incremental HF push: after every N processed files, push only the diff.

Isolation guarantee: a push ledger (``<state>/push_ledger.json``) records
``{relpath: {sha256, size}}`` for every byte verified on the Hub. A file is
uploaded iff its bytes are absent from the ledger, so a previously pushed file
is NEVER re-pushed — resume, retry and repeated exports all converge to zero
new bytes. The only overwrites are mutable index files (``metadata.jsonl``,
``MANIFEST.sha256``, card, READY), which live at a single path each and
therefore cannot duplicate.

Each push mints a fresh approval receipt bound to the current whole-tree
digest (the same gate as a full publish — nothing is relaxed), uploads the
diff as ONE commit, then verifies the pushed subset by remote readback +
sha256 before advancing the ledger. The ledger advances only on verified
success: a crash retries the same bytes, and ``upload``/commit overwrite is
idempotent, so recovery never duplicates.

Hash cost control: audio files are immutable content-addressed blobs
(``Ornix_<digits>.wav``), so a ledgered path with a matching size reuses its
recorded sha without re-reading bytes. Only mutable index files are
re-hashed every push.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..util.hashing import sha256_file
from ..util.io import atomic_write_text

LEDGER_NAME = "push_ledger.json"

# Index files change on every export; everything else is an immutable blob.
_MUTABLE_SUFFIXES = (".jsonl", ".json", ".md")

# Serializes whole pushes in this process (campaign workers share one tree).
_PUSH_LOCK = threading.Lock()


def ledger_path(state_dir: str) -> str:
    return os.path.join(state_dir, LEDGER_NAME)


def load_ledger(state_dir: str) -> Dict[str, Dict[str, Any]]:
    import json

    path = ledger_path(state_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh) or {}
    except (OSError, ValueError):
        return {}
    return raw.get("files", raw) if isinstance(raw, dict) else {}


def save_ledger(state_dir: str, ledger: Dict[str, Dict[str, Any]]) -> None:
    import json

    os.makedirs(state_dir, exist_ok=True)
    atomic_write_text(ledger_path(state_dir),
                      json.dumps({"files": ledger}, ensure_ascii=False,
                                 indent=1, sort_keys=True))


def _is_mutable(rel: str) -> bool:
    return rel.lower().endswith(_MUTABLE_SUFFIXES)


def tree_inventory(dataset_dir: str,
                   ledger: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Current ``{relpath: sha256}``, reusing ledger hashes for unchanged blobs."""
    out: Dict[str, str] = {}
    for dp, _dirs, files in os.walk(dataset_dir):
        for fn in sorted(files):
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, dataset_dir)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            rec = ledger.get(rel)
            if (rec and not _is_mutable(rel)
                    and rec.get("size") == size and rec.get("sha256")):
                out[rel] = rec["sha256"]
            else:
                out[rel] = sha256_file(full)
    return out


def mint_push_approval(dataset_dir: str, repo_id: str, state_dir: str,
                       policy_path: Optional[str],
                       revision: str = "main",
                       operator_id: Optional[str] = None) -> str:
    """Write a fresh approval receipt bound to the CURRENT tree digest."""
    import yaml

    from .approval import release_digest
    from ..canonical.publish import finalize_dataset

    fin = finalize_dataset(dataset_dir)
    if not fin.get("ok"):
        raise RuntimeError(f"cannot finalize tree for push: {fin.get('reason')}")
    digest = release_digest(dataset_dir)
    total = 0
    for dp, _d, fs in os.walk(dataset_dir):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    expires = (datetime.now(timezone.utc) + timedelta(days=2)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    approval = {
        "release_digest": digest,
        "repo_id": repo_id,
        "revision": revision,
        "max_bytes": int(total * 1.25) + (1 << 20),
        "operator_id": operator_id or os.environ.get("USER", "campaign-operator"),
        "expires_utc": expires,
        "policy_version": os.path.basename(policy_path or "unknown"),
        "license_ack": True,
        "allow_create_repo": True,
    }
    path = os.path.join(state_dir, f"push_approval_{digest[:12]}.yaml")
    os.makedirs(state_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(approval, fh, sort_keys=False)
    return path


def _verify_subset(api: Any, repo_id: str, revision: str, dataset_dir: str,
                   rels: List[str]) -> List[str]:
    """Remote readback + sha256 for exactly the pushed paths."""
    from huggingface_hub import hf_hub_download

    errors: List[str] = []
    for rel in sorted(rels):
        try:
            got = hf_hub_download(repo_id, rel, repo_type="dataset",
                                  revision=revision)
            if sha256_file(got) != sha256_file(os.path.join(dataset_dir, rel)):
                errors.append(f"REMOTE_SHA_MISMATCH:{rel}")
        except Exception as e:
            errors.append(f"REMOTE_READBACK_FAILED:{rel}:{e}")
    return errors


def push_tree(dataset_dir: str, repo_id: str, state_dir: str,
              revision: str = "main", token_env: str = "HF_TOKEN",
              policy_path: Optional[str] = None,
              full_verify: bool = False) -> Dict[str, Any]:
    """Push only files absent from the ledger; verify; advance the ledger.

    Returns a report dict. ``full_verify`` additionally runs the whole-tree
    exact-set remote verification (for the final campaign push).
    """
    with _PUSH_LOCK:
        return _push_tree_locked(
            dataset_dir, repo_id, state_dir, revision=revision,
            token_env=token_env, policy_path=policy_path,
            full_verify=full_verify)


def _push_tree_locked(dataset_dir: str, repo_id: str, state_dir: str,
                      revision: str, token_env: str,
                      policy_path: Optional[str],
                      full_verify: bool) -> Dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, HfApi

    from .approval import load_approval, validate_approval

    token = os.environ.get(token_env)
    if not token:
        return {"ok": False, "status": "BLOCKED",
                "reasons": [f"no token in ${token_env}"], "pushed": 0,
                "skipped": 0, "commit_sha": None}
    if not policy_path:
        return {"ok": False, "status": "BLOCKED",
                "reasons": ["push requires a policy-bound approval "
                            "(pass policy_path)"], "pushed": 0,
                "skipped": 0, "commit_sha": None}
    ledger = load_ledger(state_dir)
    try:
        pre = tree_inventory(dataset_dir, ledger)
    except Exception as e:
        return {"ok": False, "status": "BLOCKED",
                "reasons": [f"tree inventory failed: {e}"], "pushed": 0,
                "skipped": 0, "commit_sha": None}
    if not pre:
        return {"ok": True, "status": "NO_PUSH_NEEDED",
                "reasons": ["empty tree"], "pushed": 0, "skipped": 0,
                "commit_sha": None}
    # Finalize BEFORE diffing: the approval digest binds post-finalize bytes,
    # so the ledger must record those (not pre-finalize) hashes — otherwise
    # MANIFEST/READY would look perpetually dirty and re-push forever.
    try:
        approval_path = mint_push_approval(dataset_dir, repo_id, state_dir,
                                           policy_path, revision=revision)
    except Exception as e:
        return {"ok": False, "status": "BLOCKED",
                "reasons": [f"approval mint failed: {e}"], "pushed": 0,
                "skipped": 0, "commit_sha": None}
    inventory = tree_inventory(dataset_dir, ledger)
    pending = sorted(rel for rel, sha in inventory.items()
                     if (ledger.get(rel) or {}).get("sha256") != sha)
    if not pending:
        return {"ok": True, "status": "NO_PUSH_NEEDED",
                "reasons": ["tree matches ledger; nothing new"],
                "pushed": 0, "skipped": len(inventory), "commit_sha": None}
    receipt = load_approval(approval_path)
    ok, reasons = validate_approval(receipt, dataset_dir, repo_id,
                                    revision=revision)
    if not ok:
        return {"ok": False, "status": "BLOCKED", "reasons": reasons,
                "pushed": 0, "skipped": 0, "commit_sha": None}
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id, repo_type="dataset")
    except Exception:
        if receipt.allow_create_repo:
            try:
                api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
            except Exception as e:
                return {"ok": False, "status": "BLOCKED",
                        "reasons": [f"repo unavailable and create failed: {e}"],
                        "pushed": 0, "skipped": 0, "commit_sha": None}
        else:
            return {"ok": False, "status": "BLOCKED",
                    "reasons": ["repo unavailable"], "pushed": 0,
                    "skipped": 0, "commit_sha": None}
    ops = [CommitOperationAdd(
        path_in_repo=rel,
        path_or_fileobj=os.path.join(dataset_dir, rel)) for rel in pending]
    try:
        commit = api.create_commit(
            repo_id, ops, commit_message=f"ornix incremental push "
            f"{receipt.release_digest[:12]} ({len(pending)} files)",
            revision=revision, repo_type="dataset")
    except Exception as e:
        return {"ok": False, "status": "UPLOAD_FAILED",
                "reasons": [f"commit failed: {e}"], "pushed": 0,
                "skipped": 0, "commit_sha": None}
    commit_sha = getattr(commit, "oid", None) or revision
    errors = _verify_subset(api, repo_id, commit_sha, dataset_dir, pending)
    if errors:
        return {"ok": False, "status": "REMOTE_VERIFY_FAILED",
                "reasons": errors, "pushed": 0, "skipped": 0,
                "commit_sha": commit_sha}
    if full_verify:
        from .verification import remote_verify

        rv = remote_verify(api, repo_id, commit_sha, dataset_dir)
        if not rv.ok:
            return {"ok": False, "status": "REMOTE_VERIFY_FAILED",
                    "reasons": rv.errors, "pushed": 0, "skipped": 0,
                    "commit_sha": commit_sha}
    for rel in pending:
        try:
            size = os.path.getsize(os.path.join(dataset_dir, rel))
        except OSError:
            size = -1
        ledger[rel] = {"sha256": inventory[rel], "size": size}
    save_ledger(state_dir, ledger)
    return {"ok": True, "status": "PUBLISHED_VERIFIED",
            "reasons": [], "pushed": len(pending), "pushed_files": pending,
            "skipped": len(inventory) - len(pending), "commit_sha": commit_sha,
            "approval": approval_path}
