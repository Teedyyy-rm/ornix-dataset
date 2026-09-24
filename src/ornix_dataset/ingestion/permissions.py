"""Permission / rights gate (spec §0.2.6, Phase 0).

Rights are declared per-source in config and NEVER inferred from a dataset being
public/gated. Unknown or missing rights => LICENSE_REVIEW (quarantine), which can
never reach the publish gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..contracts.enums import RightsStatus


@dataclass
class RightsDecision:
    rights_status: RightsStatus
    redistribution_permitted: bool
    commercial_training_permitted: str  # "true" | "false" | "UNKNOWN"
    reason: str


class PermissionGate:
    """Resolves a source's rights from an explicit config declaration."""

    def __init__(self, declarations: Optional[Dict[str, Dict[str, Any]]] = None):
        # keyed by source_id or source_uri prefix
        self.declarations = declarations or {}

    def evaluate(self, source_id: str, source_uri: str,
                 declared_license: Optional[str] = None) -> RightsDecision:
        decl = self.declarations.get(source_id)
        if decl is None:
            decl = self._match_prefix(source_uri)
        if decl is None:
            return RightsDecision(RightsStatus.LICENSE_REVIEW, False, "UNKNOWN",
                                  "no explicit rights declaration for source")
        status = RightsStatus(decl.get("rights_status", "LICENSE_REVIEW"))
        redistribute = bool(decl.get("redistribution_permitted", False))
        commercial = str(decl.get("commercial_training_permitted", "UNKNOWN"))
        # Fail-closed cross-checks: redistribution only if explicitly approved.
        if status != RightsStatus.REDISTRIBUTION_APPROVED and redistribute:
            return RightsDecision(RightsStatus.LICENSE_REVIEW, False, commercial,
                                  "redistribution flag set without APPROVED status")
        if status == RightsStatus.FORBIDDEN:
            redistribute = False
        return RightsDecision(status, redistribute, commercial,
                              decl.get("note", "declared in config"))

    def _match_prefix(self, source_uri: str) -> Optional[Dict[str, Any]]:
        best = None
        best_len = -1
        for key, decl in self.declarations.items():
            prefix = decl.get("uri_prefix")
            if prefix and source_uri.startswith(prefix) and len(prefix) > best_len:
                best, best_len = decl, len(prefix)
        return best
