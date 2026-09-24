"""Speech-quality adapters (DNSMOS P.835 candidate).

UNAVAILABLE until code + weights license are verified and a checksum-pinned
checkpoint is provisioned. A no-reference MOS is NOT proof of music-free audio
(spec §4 license note) — the policy engine treats SIG/BAK/OVRL as auxiliary
evidence, never as a music/overlap gate.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from ..contracts.enums import MeasurementStatus
from .base import AdapterInfo, Availability, DetectorKind, QualityAdapter, registry


@registry.register(DetectorKind.QUALITY, "dnsmos")
class DnsmosAdapter(QualityAdapter):
    def __init__(self, model_path: Optional[str] = None, weights_sha256: Optional[str] = None,
                 license_ack: bool = False):
        self.model_path = model_path
        self.weights_sha256 = weights_sha256
        self.license_ack = license_ack
        if not (model_path and os.path.exists(model_path)):
            self._reason = "model_path not provided or missing"
        elif not license_ack:
            self._reason = "weights license not acknowledged (verify code vs weights terms)"
        else:
            self._reason = None

    def info(self) -> AdapterInfo:
        if self._reason:
            return AdapterInfo("dnsmos", DetectorKind.QUALITY, Availability.UNAVAILABLE,
                               model_id="dnsmos-p835", reason=self._reason)
        return AdapterInfo("dnsmos", DetectorKind.QUALITY, Availability.AVAILABLE,
                           model_id="dnsmos-p835", weights_sha256=self.weights_sha256)

    def infer(self, mono: np.ndarray, sr: int) -> Dict[str, Any]:
        if self._reason:
            return {"sig": None, "bak": None, "ovrl": None,
                    "status": MeasurementStatus.UNKNOWN.value, "model_id": "dnsmos-p835"}
        raise NotImplementedError("wire DNSMOS ONNX inference once provisioned")  # pragma: no cover
