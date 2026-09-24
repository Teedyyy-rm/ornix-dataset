"""Atomic, durable file writes (spec §8: atomic write + fsync, crash recovery).

All persisted records go through a temp-file + fsync + os.replace sequence so a
crash never leaves a half-written manifest. os.replace is atomic on POSIX.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def atomic_write_bytes(path: str, data: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(directory)
    except BaseException:
        _quiet_remove(tmp)
        raise


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def write_json(path: str, obj: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=True))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fsync_dir(directory: str) -> None:
    try:
        dfd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass  # directory fsync not supported on all platforms


def _quiet_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
