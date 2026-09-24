"""Speaker / overlap adapters (pyannote candidate).

NOT_APPLICABLE unless the policy requires single-speaker targets; UNAVAILABLE
until access-gated checkpoints are provisioned. Diarization never asserts a real
person's identity (spec §1.2, §4).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..contracts.enums import MeasurementStatus
from .base import AdapterInfo, Availability, DetectorKind, SpeakerAdapter, registry


@registry.register(DetectorKind.SPEAKER, "pyannote")
class PyannoteAdapter(SpeakerAdapter):
    """pyannote overlapped-speech detection. GATED: requires accepting the model's
    terms on Hugging Face and a token/provisioned checkpoint. Diarization never
    asserts a real person's identity (spec §1.2, §4)."""

    def __init__(self, model_path: Optional[str] = None, access_ack: bool = False,
                 token: Optional[str] = None):
        self.model_path = model_path
        self.access_ack = access_ack
        self.token = token
        self._pipeline = None
        self._reason: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        if not self.model_path:
            self._reason = "pipeline/checkpoint not provisioned"
            return
        if not self.access_ack:
            self._reason = "pyannote access terms not acknowledged"
            return
        try:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(self.model_path, use_auth_token=self.token)
        except Exception as e:  # pragma: no cover - gated/network/weights dependent
            self._reason = f"pyannote unavailable: {e}"
            self._pipeline = None

    def info(self) -> AdapterInfo:
        if self._reason:
            return AdapterInfo("pyannote", DetectorKind.SPEAKER, Availability.UNAVAILABLE,
                               model_id="pyannote-overlap", reason=self._reason)
        return AdapterInfo("pyannote", DetectorKind.SPEAKER, Availability.AVAILABLE,
                           model_id="pyannote-overlap", license="gated")

    def infer(self, mono: np.ndarray, sr: int) -> Dict[str, Any]:
        if self._pipeline is None:
            return {"overlap_intervals": [], "n_speakers": None,
                    "status": MeasurementStatus.UNKNOWN.value}
        import torch  # pragma: no cover - gated path

        waveform = torch.from_numpy(np.asarray(mono, dtype=np.float32)[None, :])
        out = self._pipeline({"waveform": waveform, "sample_rate": sr})
        intervals = [[round(seg.start, 4), round(seg.end, 4)]
                     for seg in out.get_timeline().support()]
        return {"overlap_intervals": intervals, "n_speakers": None,
                "status": MeasurementStatus.OK.value}
