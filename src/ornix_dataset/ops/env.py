""".env loading (dependency-free).

The toolkit is fail-closed about secrets: a token is only read from the
environment variable named by the caller (``HF_TOKEN`` by default) and is never
logged. For local convenience the CLI loads a ``.env`` file once at startup into
``os.environ`` so operators can keep a token out of their shell history.

Rules:
- Existing environment variables always win unless ``override=True``.
- ``.env`` is never committed (``.gitignore``); ``.env.example`` is the template.
- Values are never printed; :func:`load_dotenv` returns the *names* it set.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

_PREFIX = "export "


def find_env_file(start: Optional[str] = None) -> Optional[str]:
    """Locate a ``.env`` file: ``$ORNIX_ENV_FILE``, else ``start``/cwd upward."""
    override = os.environ.get("ORNIX_ENV_FILE")
    if override:
        return override if os.path.exists(override) else None
    here = os.path.abspath(start or os.getcwd())
    for _ in range(64):
        candidate = os.path.join(here, ".env")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def _parse_line(line: str) -> Optional[tuple]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith(_PREFIX):
        line = line[len(_PREFIX):].lstrip()
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    if not key or not (key[0].isalpha() or key[0] == "_") \
            or not key.replace("_", "").isalnum():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return key, value


def load_dotenv(path: Optional[str] = None, override: bool = False,
                start: Optional[str] = None) -> List[str]:
    """Load ``.env`` into ``os.environ``; return the names that were set.

    Missing file is a no-op. Existing vars are preserved unless ``override``.
    """
    target = path or find_env_file(start)
    if not target or not os.path.exists(target):
        return []
    set_names: List[str] = []
    with open(target, "r", encoding="utf-8") as fh:
        for line in fh:
            parsed = _parse_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if not override and key in os.environ:
                continue
            os.environ[key] = value
            set_names.append(key)
    return set_names


def env_status() -> Dict[str, bool]:
    """Non-sensitive presence check (never returns the token value)."""
    return {"HF_TOKEN": bool(os.environ.get("HF_TOKEN"))}
