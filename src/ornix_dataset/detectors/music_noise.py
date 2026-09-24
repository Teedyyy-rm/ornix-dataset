"""Noise / sound-event detectors.

``DspNoiseDetector`` (pure NumPy/SciPy) gives real, deterministic evidence for
hiss, hum/buzz and clipping. It CANNOT confirm music-under-speech, so it never
emits ``MUSIC_BACKGROUND`` — that requires a licensed sound-event model. The
PANNs adapter is the candidate for the music gate and is UNAVAILABLE until
checksum-pinned weights are provisioned (fail-closed, spec §3/§4).
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ..contracts.enums import NoiseLabel, Severity
from ..contracts.quality import NoiseEvent
from ..dsp.features import hum_ratio, signal_stats, spectral_flatness
from .base import AdapterInfo, Availability, DetectorKind, NoiseDetector, registry
from .windowing import plan_windows


@registry.register(DetectorKind.NOISE, "dsp")
class DspNoiseDetector(NoiseDetector):
    """Deterministic DSP evidence. Provisional severity only — policy calibrates."""

    def __init__(self, flatness_hiss: float = 0.4, hum_ratio_thresh: float = 0.15,
                 clip_ratio_thresh: float = 0.005, window_s: float = 1.5):
        self.flatness_hiss = flatness_hiss
        self.hum_ratio_thresh = hum_ratio_thresh
        self.clip_ratio_thresh = clip_ratio_thresh
        self.window_s = window_s

    def info(self) -> AdapterInfo:
        return AdapterInfo("dsp", DetectorKind.NOISE, Availability.AVAILABLE,
                           model_id="dsp-noise", model_revision="0.1.0",
                           license="Apache-2.0", is_heuristic=True)

    def infer(self, mono: np.ndarray, sr: int, speech_intervals: List[List[float]]) -> List[NoiseEvent]:
        events: List[NoiseEvent] = []
        rev = "dsp-noise-0.1.0"
        plan = plan_windows(len(mono) / sr, self.window_s, overlap=0.5)
        speech = _mask(speech_intervals)
        for (s, e) in plan.windows:
            seg = mono[int(s * sr): int(e * sr)]
            if seg.size < 256:
                continue
            overlaps = speech(s, e)
            flat = spectral_flatness(seg, sr)
            if flat is not None and flat > self.flatness_hiss:
                events.append(self._ev(NoiseLabel.HISS_STATIC, s, e, overlaps, flat,
                                       min(1.0, flat), "dsp-flatness", rev))
            hum = hum_ratio(seg, sr)
            if hum is not None and hum > self.hum_ratio_thresh:
                events.append(self._ev(NoiseLabel.HUM_BUZZ, s, e, overlaps, hum,
                                       min(1.0, hum), "dsp-hum", rev))
            st = signal_stats(seg)
            if st.clipping_ratio > self.clip_ratio_thresh:
                sev = Severity.N3 if st.clipping_ratio > 0.02 else Severity.N2
                events.append(NoiseEvent(NoiseLabel.CLIPPING_DISTORTION, s, e, overlaps,
                                         sev, min(1.0, st.clipping_ratio * 10),
                                         st.clipping_ratio, "dsp-clipping", rev))
        return events

    @staticmethod
    def _ev(label, s, e, overlaps, score, conf, det, rev) -> NoiseEvent:
        # provisional severity: never assert N0 (clean) from uncalibrated DSP
        sev = Severity.N2 if overlaps else Severity.N1
        return NoiseEvent(label, s, e, overlaps, sev, float(conf), float(score), det, rev)


def _mask(intervals: List[List[float]]):
    ivs = sorted(intervals or [])

    def overlaps(s: float, e: float) -> bool:
        return any(not (e <= a or s >= b) for a, b in ivs)

    return overlaps


@registry.register(DetectorKind.NOISE, "panns")
class PannsMusicDetector(NoiseDetector):
    """PANNs sound-event tagging (music gate candidate). UNAVAILABLE until a
    checksum-pinned checkpoint is provisioned; never fabricates timestamps."""

    def __init__(self, model_path: Optional[str] = None, weights_sha256: Optional[str] = None):
        self.model_path = model_path
        self.weights_sha256 = weights_sha256
        self._reason = "model_path not provided or missing" if not (
            model_path and os.path.exists(model_path)) else None

    def info(self) -> AdapterInfo:
        if self._reason:
            return AdapterInfo("panns", DetectorKind.NOISE, Availability.UNAVAILABLE,
                               model_id="panns-cnn14", reason=self._reason)
        return AdapterInfo("panns", DetectorKind.NOISE, Availability.AVAILABLE,
                           model_id="panns-cnn14", weights_sha256=self.weights_sha256)

    def infer(self, mono, sr, speech_intervals):  # pragma: no cover
        raise RuntimeError(f"PANNs unavailable: {self._reason}")
