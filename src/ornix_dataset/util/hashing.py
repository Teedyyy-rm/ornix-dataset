"""Content hashing + stable id derivation (spec §0.2.1, §6.4).

IDs are derived from *source identity + content hash + segment position + recipe
version* so they never depend on scan order or source file name.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_CHUNK = 1 << 20  # 1 MiB


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Deterministic hash of a JSON-serialisable object (sorted keys)."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def short_id(*parts: Any, length: int = 16) -> str:
    """Stable short hex id from arbitrary parts (order-sensitive by design)."""
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]
