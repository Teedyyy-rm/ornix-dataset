"""Persistent in-process model server (detector cache).

The campaign used to spawn a fresh ``python -m ornix_dataset.cli qc``
subprocess per chunk, reloading Silero/DNSMOS ONNX sessions (plus a full
weights-sha256 re-read) every time. This module keeps ONE DetectorSet per
models-lock alive for the whole worker process: chunks reuse the already-warm
sessions instead of paying load + sha-verify per chunk.

Thread-safe: ONNX InferenceSession.run is safe for concurrent calls, and the
per-call Silero state stays on the stack. Cache key is (lock abspath, mtime)
so editing the lock transparently rebuilds.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional, Tuple

_lock = threading.Lock()
_cache: Dict[Tuple[str, float], Any] = {}


def _key(models_lock: Optional[str]) -> Tuple[str, float]:
    if not models_lock:
        return ("", 0.0)
    try:
        ap = os.path.abspath(models_lock)
        return (ap, os.path.getmtime(ap))
    except OSError:
        return (os.path.abspath(models_lock), 0.0)


def get_detectors(models_lock: Optional[str] = None):
    """Return the cached DetectorSet for ``models_lock``, building it once."""
    from ..config import build_detectors

    k = _key(models_lock)
    with _lock:
        hit = _cache.get(k)
    if hit is not None:
        return hit
    with _lock:
        # double-checked: another thread may have built it while we waited.
        hit = _cache.get(k)
        if hit is not None:
            return hit
        ds = build_detectors(models_lock)
        # keep the cache bounded: one entry per distinct lock (normally 1).
        _cache.clear()
        _cache[k] = ds
        return ds


def clear_detectors_cache() -> None:
    """Drop all cached sessions (tests / explicit model rotation)."""
    with _lock:
        _cache.clear()
