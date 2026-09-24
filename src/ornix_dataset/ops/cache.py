"""Content-addressed analysis cache (spec §8).

Cache key binds every input that can change a result: source content hash, the
detector weight shas, the config hash and the policy version. A model upgrade
changes the weight sha and therefore invalidates the cache (no stale results).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..util.hashing import sha256_json, short_id
from ..util.io import read_json, write_json


def cache_key(source_sha256: str, detector_weight_shas: Dict[str, str],
              config_sha: str, policy_version: str) -> str:
    payload = {
        "source": source_sha256,
        "weights": {k: detector_weight_shas[k] for k in sorted(detector_weight_shas)},
        "config": config_sha,
        "policy": policy_version,
    }
    return short_id(sha256_json(payload), length=32)


class ContentCache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.json")

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if os.path.exists(p):
            try:
                return read_json(p)
            except Exception:
                return None
        return None

    def put(self, key: str, value: Any) -> None:
        write_json(self._path(key), value)

    def invalidate(self, key: str) -> None:
        p = self._path(key)
        if os.path.exists(p):
            os.remove(p)
