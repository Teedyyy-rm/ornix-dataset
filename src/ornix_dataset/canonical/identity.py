"""Persistent, stable identity for the unified Ornix Dataset (spec §4, §5).

Two mappings are persisted **outside** the public tree (they are provenance and
MUST NOT be published):

- ``stable_internal_sample_key -> ornix_id``  (opaque ``ornix_<32 hex>``)
- ``source speaker identity -> spk_<32 hex>``

and an internal provenance record per ``ornix_id`` for audit/takedown.

Identity rules:
- The internal sample key distinguishes source revision, file/shard and segment
  identity — it never uses filename order, batch index, timestamp or a
  per-dataset counter.
- An existing mapping is reused on retry/resume; a new opaque id is allocated
  (uuid4) with a collision check and persisted **before** any publish.
- Two datasets that both name a speaker ``speaker_01`` do NOT merge: the
  speaker key is scoped by the source identity. An unverified speaker (no
  trusted ref) gets a unique per-sample id — never merged.

A single logical writer is enforced with an advisory ``flock`` so concurrent
workers cannot clobber the maps or the metadata (spec §7).
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

from ..util.io import read_json, write_json
from ..util.timeutil import utc_now_iso

IDENTITY_SCHEMA = "ornix-canonical-state-v1"


def internal_sample_key(source_id: str, source_sha256: str,
                        seg_start: int, seg_end: int,
                        revision: Optional[str] = None) -> str:
    """Stable key for one sample/segment; independent of names and ordering."""
    parts = [IDENTITY_SCHEMA, revision or "", source_id or "",
             source_sha256 or "", str(int(seg_start)), str(int(seg_end))]
    return "\x1f".join(parts)


def speaker_scope_key(source_scope: str, speaker_ref: Optional[str]
                      ) -> Optional[str]:
    """Trusted speaker identity, or ``None`` when the speaker is unverified."""
    ref = (speaker_ref or "").strip()
    if not ref:
        return None
    return f"{IDENTITY_SCHEMA}\x1fSCOPE:{source_scope}\x1fREF:{ref}"


def _new_opaque(prefix: str, taken: Callable[[str], bool],
                uuid_factory: Callable[[], uuid.UUID]) -> str:
    while True:
        candidate = f"{prefix}{uuid_factory().hex}"
        if not taken(candidate):
            return candidate


class CanonicalState:
    """Locked, durably-persisted canonical identity + provenance store."""

    def __init__(self, state_dir: str):
        self.state_dir = os.path.abspath(state_dir)
        os.makedirs(self.state_dir, exist_ok=True)
        self.path = os.path.join(self.state_dir, "ornix_state.json")
        self._lock_fh = None
        self.data: Dict[str, Any] = {}

    # -- lifecycle -------------------------------------------------------
    def __enter__(self) -> "CanonicalState":
        self._lock_fh = open(os.path.join(self.state_dir, ".lock"), "a+")
        try:
            import fcntl
            fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-posix
            pass
        self.data = self._load()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._lock_fh is not None:
                import fcntl
                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):  # pragma: no cover
            pass
        if self._lock_fh is not None:
            self._lock_fh.close()
            self._lock_fh = None

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"schema": IDENTITY_SCHEMA, "ids": {}, "speakers": {},
                    "provenance": {}}
        data = read_json(self.path)
        if data.get("schema") != IDENTITY_SCHEMA:
            raise ValueError(f"unknown canonical state schema: {data.get('schema')}")
        for k in ("ids", "speakers", "provenance"):
            data.setdefault(k, {})
        return data

    def commit(self) -> None:
        write_json(self.path, self.data)

    # -- ids -------------------------------------------------------------
    def _taken_ids(self) -> set:
        return {v["ornix_id"] for v in self.data["ids"].values()}

    def ornix_id_for(self, key: str, audio_sha256: str,
                     uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4) -> str:
        entry = self.data["ids"].get(key)
        if entry is not None:
            return entry["ornix_id"]
        taken = self._taken_ids()
        new_id = _new_opaque("ornix_", lambda c: c in taken, uuid_factory)
        self.data["ids"][key] = {"ornix_id": new_id,
                                 "audio_sha256": audio_sha256,
                                 "created_utc": utc_now_iso()}
        return new_id

    # -- speakers --------------------------------------------------------
    def _taken_speakers(self) -> set:
        return {v["speaker_id"] for v in self.data["speakers"].values()}

    def speaker_for(self, source_scope: str, speaker_ref: Optional[str],
                    sample_key: str,
                    uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4) -> str:
        trusted = speaker_scope_key(source_scope, speaker_ref)
        if trusted is None:
            # Unverified speaker: unique per sample, stable across resume.
            key = f"{IDENTITY_SCHEMA}\x1fUNKNOWN\x1f{sample_key}"
        else:
            key = trusted
        entry = self.data["speakers"].get(key)
        if entry is not None:
            return entry["speaker_id"]
        taken = self._taken_speakers()
        new_id = _new_opaque("spk_", lambda c: c in taken, uuid_factory)
        self.data["speakers"][key] = {"speaker_id": new_id,
                                      "created_utc": utc_now_iso()}
        return new_id

    # -- provenance ------------------------------------------------------
    def set_provenance(self, ornix_id: str, record: Dict[str, Any]) -> None:
        self.data["provenance"][ornix_id] = {**record,
                                             "recorded_utc": utc_now_iso()}

    def provenance_for(self, ornix_id: str) -> Optional[Dict[str, Any]]:
        return self.data["provenance"].get(ornix_id)

    def all_ids(self) -> List[str]:
        return sorted(v["ornix_id"] for v in self.data["ids"].values())


@contextmanager
def open_state(state_dir: str):
    st = CanonicalState(state_dir)
    with st:
        yield st
