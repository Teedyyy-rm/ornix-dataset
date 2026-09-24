"""Speaker / overlap adapters (pyannote candidate).

NOT_APPLICABLE unless the policy requires single-speaker targets; UNAVAILABLE
until access-gated checkpoints are provisioned. Diarization never asserts a real
person's identity (spec §1.2, §4).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from ..contracts.enums import MeasurementStatus
from .base import AdapterInfo, Availability, DetectorKind, SpeakerAdapter, registry


@registry.register(DetectorKind.SPEAKER, "pyannote")
class PyannoteAdapter(SpeakerAdapter):
    def __init__(self, model_path: Optional[str] = None, access_ack: bool = False):
        self.model_path = model_path
        self.access_ack = access_ack
        if not (model_path and os.path.exists(model_path)):
            self._reason = "pipeline/checkpoint not provisioned"
        elif not access_ack:
            self._reason = "pyannote access terms not acknowledged"
        else:
            self._reason = None

    def info(self) -> AdapterInfo:
        if self._reason:
            return AdapterInfo("pyannote", DetectorKind.SPEAKER, Availability.UNAVAILABLE,
                               model_id="pyannote-overlap", reason=self._reason)
        return AdapterInfo("pyannote", DetectorKind.SPEAKER, Availability.AVAILABLE,
                           model_id="pyannote-overlap")

    def infer(self, mono: np.ndarray, sr: int) -> Dict[str, Any]:
        if self._reason:
            return {"overlap_intervals": [], "n_speakers": None,
                    "status": MeasurementStatus.UNKNOWN.value}
        raise NotImplementedError("wire pyannote pipeline once provisioned")  # pragma: no cover
