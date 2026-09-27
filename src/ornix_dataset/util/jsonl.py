"""Append-only JSONL helpers with durable append (spec §6.1, §8).

Manifests are append-only; ``append_jsonl`` flushes + fsyncs each record so a
crash mid-run keeps every already-written row.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Iterator, List


def fsync_enabled() -> bool:
    """True unless bulk mode opts out via ORNIX_FSYNC_APPEND=0.

    Per-row fsync costs ~ms per append (open+flush+fsync+close); a QC file
    emits ~5 rows, so fsync dominates wall time on fast disks. Disabling only
    risks re-doing tail rows after a crash (all writers here are idempotent /
    re-runnable) — it can never corrupt data. Default stays durable.
    """
    v = os.environ.get("ORNIX_FSYNC_APPEND", "1").strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _sync(fh) -> None:
    fh.flush()
    if fsync_enabled():
        os.fsync(fh.fileno())


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        _sync(fh)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        _sync(fh)
    return count


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))
