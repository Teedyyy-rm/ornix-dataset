"""VAD adapters. Silero is the licensed candidate (spec §4); an energy-based
heuristic is always available for windowing but is flagged ``is_heuristic`` so
the policy engine never treats VAD=1 as a clean-signal proof (spec Phase 3).
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ..dsp.features import frame_energy_db
from .base import AdapterInfo, Availability, DetectorKind, VadAdapter, registry


def _merge_gaps(intervals: List[List[float]], gap_s: float) -> List[List[float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


@registry.register(DetectorKind.VAD, "energy")
class EnergyVadAdapter(VadAdapter):
    """Deterministic energy-gate VAD. Heuristic: usable for windowing, not a gate."""

    def __init__(self, db_margin: float = 12.0, min_speech_s: float = 0.15,
                 frame_ms: float = 30.0, rel_threshold: float = 0.25,
                 merge_gap_s: float = 0.3):
        self.db_margin = db_margin
        self.min_speech_s = min_speech_s
        self.frame_ms = frame_ms
        self.rel_threshold = rel_threshold
        self.merge_gap_s = merge_gap_s

    def info(self) -> AdapterInfo:
        return AdapterInfo("energy", DetectorKind.VAD, Availability.AVAILABLE,
                           model_id="dsp-energy", model_revision="0.1.0",
                           license="Apache-2.0", is_heuristic=True)

    def infer(self, mono: np.ndarray, sr: int) -> List[List[float]]:
        energies_db, hop = frame_energy_db(mono, sr, self.frame_ms)
        if energies_db.size == 0:
            return []
        floor = float(np.percentile(energies_db, 15))
        peak = float(np.percentile(energies_db, 95))
        # adaptive: above noise floor by margin OR a fraction into the dynamic range
        thresh = min(floor + self.db_margin, floor + self.rel_threshold * (peak - floor))
        active = energies_db > thresh
        intervals: List[List[float]] = []
        hop_s = hop / sr
        start = None
        for i, a in enumerate(active):
            if a and start is None:
                start = i
            elif not a and start is not None:
                intervals.append([start * hop_s, i * hop_s])
                start = None
        if start is not None:
            intervals.append([start * hop_s, len(active) * hop_s])
        intervals = _merge_gaps(intervals, self.merge_gap_s)
        return [iv for iv in intervals if (iv[1] - iv[0]) >= self.min_speech_s]


@registry.register(DetectorKind.VAD, "silero")
class SileroVadAdapter(VadAdapter):
    """Silero VAD (ONNX). UNAVAILABLE unless a local, checksum-pinned model is
    provided via ``model_path`` (fail-closed — never auto-downloads weights)."""

    def __init__(self, model_path: Optional[str] = None, weights_sha256: Optional[str] = None):
        self.model_path = model_path
        self.weights_sha256 = weights_sha256
        self._session = None
        self._reason: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            self._reason = "model_path not provided or missing"
            return
        try:
            import onnxruntime  # noqa: F401
        except Exception as e:
            self._reason = f"onnxruntime unavailable: {e}"
            return
        try:
            from ..util.hashing import sha256_file

            if self.weights_sha256 and sha256_file(self.model_path) != self.weights_sha256:
                self._reason = "weights sha256 mismatch"
                return
            import onnxruntime as ort

            self._session = ort.InferenceSession(self.model_path,
                                                  providers=["CPUExecutionProvider"])
        except Exception as e:  # pragma: no cover - depends on runtime
            self._reason = f"failed to init session: {e}"

    def info(self) -> AdapterInfo:
        if self._session is None:
            return AdapterInfo("silero", DetectorKind.VAD, Availability.UNAVAILABLE,
                               model_id="silero-vad", reason=self._reason)
        return AdapterInfo("silero", DetectorKind.VAD, Availability.AVAILABLE,
                           model_id="silero-vad", weights_sha256=self.weights_sha256,
                           license="MIT")

    def infer(self, mono: np.ndarray, sr: int) -> List[List[float]]:  # pragma: no cover
        if self._session is None:
            raise RuntimeError(f"silero VAD unavailable: {self._reason}")
        raise NotImplementedError("wire Silero inference once weights are provisioned")
