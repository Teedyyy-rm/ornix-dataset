"""Checkpoint / crash-recovery (spec §8).

Completed records are persisted (durably appended) BEFORE progress is advanced,
so a crash never loses evidence and re-runs are idempotent: an item whose id is
already marked done is skipped (no double-insert).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Set

from ..util.jsonl import append_jsonl, read_jsonl
from ..util.timeutil import utc_now_iso


@dataclass
class RunProgress:
    run_id: str
    done: Set[str] = field(default_factory=set)
    states: Dict[str, str] = field(default_factory=dict)


class Checkpoint:
    """Append-only per-item state log; last state for an id wins on replay."""

    def __init__(self, path: str, run_id: str):
        self.path = path
        self.run_id = run_id

    def load(self) -> RunProgress:
        prog = RunProgress(self.run_id)
        for rec in read_jsonl(self.path):
            if rec.get("run_id") != self.run_id:
                continue
            item = rec.get("item_id")
            prog.states[item] = rec.get("state")
            if rec.get("state") in ("ACCEPTED", "REJECTED", "REVIEWED", "PACKAGED",
                                    "TECH_VERIFIED", "ANALYZED", "DONE"):
                prog.done.add(item)
        return prog

    def mark(self, item_id: str, state: str, **fields: Any) -> None:
        append_jsonl(self.path, {"run_id": self.run_id, "item_id": item_id,
                                 "state": state, "ts_utc": utc_now_iso(), **fields})

    def is_done(self, item_id: str, prog: RunProgress) -> bool:
        return item_id in prog.done
