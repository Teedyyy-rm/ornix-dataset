"""Repository references: normalize user input to (repo_id, revision).

Accepts a bare ``org/name``, an optional ``datasets/`` prefix, or a full
Hugging Face URL (…/datasets/org/name[/tree|blob/<rev>[/<path>]]). Raw URLs
are NEVER used to build filesystem paths — callers must use the normalized
repo_id (slugged) instead.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Optional, Tuple
from urllib.parse import urlparse

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def is_full_sha(s: str) -> bool:
    return bool(_FULL_SHA.match((s or "").strip().lower()))


def canonical_repo_id(repo_id: str) -> str:
    """Stripped ``org/name``; raises on anything that is not exactly that."""
    rid = (repo_id or "").strip().strip("/")
    if rid.startswith("datasets/"):
        rid = rid[len("datasets/"):]
    parts = rid.split("/")
    if len(parts) != 2 or not all(p.strip() and ".." not in p for p in parts):
        raise ValueError(f"not a dataset repo_id 'org/name': {repo_id!r}")
    return f"{parts[0].strip()}/{parts[1].strip()}"


def normalize_repo_ref(raw: str) -> Tuple[str, Optional[str]]:
    """Return ``(repo_id, requested_revision_or_None)`` for bare ids or URLs."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty dataset reference")
    if "://" in raw:
        u = urlparse(raw)
        if u.scheme not in ("http", "https") or "huggingface.co" not in (u.hostname or ""):
            raise ValueError(f"unsupported dataset URL host: {raw!r}")
        segs = [s for s in u.path.split("/") if s]
        if not segs or segs[0] != "datasets" or len(segs) < 3:
            raise ValueError(f"not a dataset URL: {raw!r}")
        repo_id = canonical_repo_id(f"{segs[1]}/{segs[2]}")
        revision: Optional[str] = None
        if len(segs) > 4 and segs[3] in ("tree", "blob", "resolve"):
            revision = segs[4]
        return repo_id, revision
    # bare id, optionally with @revision suffix
    if "@" in raw:
        rid, _, rev = raw.partition("@")
        return canonical_repo_id(rid), (rev.strip() or None)
    return canonical_repo_id(raw), None


def default_resolver(repo_id: str, revision: str) -> str:
    """Resolve a branch/tag to an exact commit SHA via the Hub API."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.repo_info(repo_id, revision=revision, repo_type="dataset")
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not is_full_sha(sha):
        raise RuntimeError(f"could not resolve {repo_id}@{revision} to a commit SHA")
    return sha.lower()


def pin_revision(repo_id: str, requested: Optional[str],
                 resolver: Callable[[str, str], str] = default_resolver) -> str:
    """Pin to an exact 40-hex commit SHA. Full SHAs need no network."""
    if requested and is_full_sha(requested):
        return requested.strip().lower()
    return resolver(repo_id, (requested or "main").strip())
