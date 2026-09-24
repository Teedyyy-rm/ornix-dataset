"""Ingestion: source adapters + permission gate (spec Phase 0/1, §6.1)."""

from .permissions import PermissionGate, RightsDecision
from .base import SourceAdapter
from .local import LocalSourceAdapter
from .hf import HfSourceAdapter

__all__ = [
    "PermissionGate",
    "RightsDecision",
    "SourceAdapter",
    "LocalSourceAdapter",
    "HfSourceAdapter",
]
