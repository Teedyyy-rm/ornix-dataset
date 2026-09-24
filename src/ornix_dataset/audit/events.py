"""Append-only audit event log.

Every state transition / decision writes an event with provenance so a run can
be reconstructed. Events are never mutated (spec §0.2.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..util.jsonl import append_jsonl, read_jsonl
from ..util.timeutil import utc_now_iso


@dataclass
class AuditLog:
    path: str
    run_id: Optional[str] = None
    _seq: int = field(default=0, repr=False)

    def emit(self, event: str, subject: str, **fields: Any) -> Dict[str, Any]:
        self._seq += 1
        record = {
            "seq": self._seq,
            "ts_utc": utc_now_iso(),
            "run_id": self.run_id,
            "event": event,
            "subject": subject,
            **fields,
        }
        append_jsonl(self.path, record)
        return record

    def events(self):
        return read_jsonl(self.path)


def emit(path: str, event: str, subject: str, **fields: Any) -> Dict[str, Any]:
    """Fire-and-forget single event (durable append)."""
    record = {"ts_utc": utc_now_iso(), "event": event, "subject": subject, **fields}
    append_jsonl(path, record)
    return record
