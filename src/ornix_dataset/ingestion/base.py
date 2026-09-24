"""SourceAdapter interface (spec §2.1)."""

from __future__ import annotations

from typing import Iterator

from ..contracts.source import SourceRecord


class SourceAdapter:
    """Read-only scanner. Implementations must not mutate source bytes."""

    name: str = "base"

    def scan(self) -> Iterator[SourceRecord]:  # pragma: no cover - interface
        raise NotImplementedError
