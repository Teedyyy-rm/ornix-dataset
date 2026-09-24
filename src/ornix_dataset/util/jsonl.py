"""Append-only JSONL helpers with durable append (spec §6.1, §8).

Manifests are append-only; ``append_jsonl`` flushes + fsyncs each record so a
crash mid-run keeps every already-written row.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Iterator, List


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        fh.flush()
        os.fsync(fh.fileno())
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
