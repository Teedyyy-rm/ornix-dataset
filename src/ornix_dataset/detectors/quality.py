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
    """DNSMOS P.835 (SIG/BAK/OVRL). UNAVAILABLE until a checksum-pinned ONNX model
    is provisioned AND its weights license is acknowledged (fail-closed). A high
    MOS is auxiliary evidence only — never a music/overlap gate (spec §4)."""

    _SR = 16000
    _INPUT_LENGTH = 9.01
    # non-personalised P.835 polynomial fits (Microsoft DNSMOS local)
    _P_SIG = (-0.08397278, 1.22083953, 0.0052439)
    _P_BAK = (-0.13166888, 1.60915514, -0.39604546)
    _P_OVR = (-0.06766283, 1.11546468, 0.04602535)

    def __init__(self, model_path: Optional[str] = None, weights_sha256: Optional[str] = None,
                 license_ack: bool = False):
        self.model_path = model_path
        self.weights_sha256 = weights_sha256
        self.license_ack = license_ack
        self._session = None
        self._input_name: Optional[str] = None
        self._reason: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        if not (self.model_path and os.path.exists(self.model_path)):
            self._reason = "model_path not provided or missing"
            return
        if not self.license_ack:
            self._reason = "weights license not acknowledged (verify code vs weights terms)"
            return
        try:
            from ..util.hashing import sha256_file
            if self.weights_sha256 and sha256_file(self.model_path) != self.weights_sha256:
                self._reason = "weights sha256 mismatch"
                return
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.model_path,
                                                  providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
        except Exception as e:  # pragma: no cover - depends on runtime/weights
            self._reason = f"onnxruntime/weights unavailable: {e}"
            self._session = None

    def info(self) -> AdapterInfo:
        if self._reason:
            return AdapterInfo("dnsmos", DetectorKind.QUALITY, Availability.UNAVAILABLE,
                               model_id="dnsmos-p835", reason=self._reason)
        return AdapterInfo("dnsmos", DetectorKind.QUALITY, Availability.AVAILABLE,
                           model_id="dnsmos-p835", weights_sha256=self.weights_sha256,
                           license="MIT-code")

    def infer(self, mono: np.ndarray, sr: int) -> Dict[str, Any]:
        if self._session is None:
            return {"sig": None, "bak": None, "ovrl": None,
                    "status": MeasurementStatus.UNKNOWN.value, "model_id": "dnsmos-p835"}
        from .music_noise import _resample_to

        audio = _resample_to(mono, sr, self._SR) if sr != self._SR else mono.astype(np.float32)
        need = int(self._INPUT_LENGTH * self._SR)
        if len(audio) < need:  # tile short clips as the reference implementation does
            reps = int(np.ceil(need / max(1, len(audio))))
            audio = np.tile(audio, reps)
        seg = audio[:need][None, :].astype(np.float32)
        raw = np.asarray(self._session.run(None, {self._input_name: seg})[0]).reshape(-1)
        sig_raw, bak_raw, ovr_raw = float(raw[0]), float(raw[1]), float(raw[2])
        sig = float(np.polyval(self._P_SIG, sig_raw))
        bak = float(np.polyval(self._P_BAK, bak_raw))
        ovrl = float(np.polyval(self._P_OVR, ovr_raw))
        return {"sig": round(sig, 4), "bak": round(bak, 4), "ovrl": round(ovrl, 4),
                "status": MeasurementStatus.OK.value, "model_id": "dnsmos-p835"}
