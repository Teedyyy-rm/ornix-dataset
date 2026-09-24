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
    """PANNs sound-event tagging (music gate). Uses framewise SED to emit
    ``MUSIC_BACKGROUND`` intervals. UNAVAILABLE until a checksum-pinned checkpoint
    is provisioned; never fabricates timestamps."""

    _SR = 32000  # PANNs models are trained at 32 kHz

    def __init__(self, model_path: Optional[str] = None, weights_sha256: Optional[str] = None,
                 music_threshold: float = 0.2, min_event_s: float = 0.3):
        self.model_path = model_path
        self.weights_sha256 = weights_sha256
        self.music_threshold = music_threshold
        self.min_event_s = min_event_s
        self._sed = None
        self._music_idx: Optional[List[int]] = None
        self._reason: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        if not (self.model_path and os.path.exists(self.model_path)):
            self._reason = "model_path not provided or missing"
            return
        try:
            from ..util.hashing import sha256_file
            if self.weights_sha256 and sha256_file(self.model_path) != self.weights_sha256:
                self._reason = "weights sha256 mismatch"
                return
            from panns_inference import SoundEventDetection, labels
            self._sed = SoundEventDetection(checkpoint_path=self.model_path, device="cpu")
            music = {"Music", "Musical instrument", "Singing"}
            self._music_idx = [i for i, l in enumerate(labels) if l in music]
            if not self._music_idx:
                self._reason = "AudioSet 'Music' class not found in model labels"
                self._sed = None
        except Exception as e:  # pragma: no cover - depends on runtime/weights
            self._reason = f"panns_inference/weights unavailable: {e}"
            self._sed = None

    def info(self) -> AdapterInfo:
        if self._reason:
            return AdapterInfo("panns", DetectorKind.NOISE, Availability.UNAVAILABLE,
                               model_id="panns-cnn14", detects_music=True, reason=self._reason)
        return AdapterInfo("panns", DetectorKind.NOISE, Availability.AVAILABLE,
                           model_id="panns-cnn14", weights_sha256=self.weights_sha256,
                           detects_music=True, license="Apache-2.0")

    def infer(self, mono, sr, speech_intervals):
        if self._sed is None:
            raise RuntimeError(f"PANNs unavailable: {self._reason}")
        import numpy as _np

        audio = _resample_to(mono, sr, self._SR)
        framewise = self._sed.inference(audio[None, :])[0]  # (n_frames, 527)
        music = framewise[:, self._music_idx].max(axis=1)
        n = len(music)
        dur = len(audio) / self._SR
        hop_s = dur / max(1, n)
        speech = _mask(speech_intervals)
        events: List[NoiseEvent] = []
        run_start: Optional[int] = None
        for i in range(n + 1):
            active = i < n and music[i] >= self.music_threshold
            if active and run_start is None:
                run_start = i
            elif not active and run_start is not None:
                s, e = run_start * hop_s, i * hop_s
                if e - s >= self.min_event_s:
                    score = float(music[run_start:i].mean())
                    sev = Severity.N2 if speech(s, e) else Severity.N1
                    events.append(NoiseEvent(NoiseLabel.MUSIC_BACKGROUND, round(s, 3),
                                             round(e, 3), speech(s, e), sev, score, score,
                                             "panns-sed", "panns-cnn14"))
                run_start = None
        return events


def _resample_to(mono, sr: int, target: int):
    import numpy as _np
    if sr == target:
        return mono.astype(_np.float32)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target)
        return resample_poly(mono, target // g, sr // g).astype(_np.float32)
    except Exception:  # pragma: no cover - scipy always present in dsp extra
        n = int(round(len(mono) * target / sr))
        idx = _np.linspace(0, len(mono) - 1, n)
        return _np.interp(idx, _np.arange(len(mono)), mono).astype(_np.float32)
