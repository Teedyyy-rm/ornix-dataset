"""Signal features + DSP noise diagnostics (spec §3, §4 DSP).

These are *evidence*, not verdicts. Deterministic, CPU-only, dependency-light
(pure NumPy). Used both by the technical gate and as fallback noise detectors
when no ML back-end is licensed/available.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class SignalStats:
    rms: float
    peak: float
    dc_offset: float
    clipping_ratio: float
    max_clipped_run: int
    has_nan: bool
    has_inf: bool
    crest_factor: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def signal_stats(x: np.ndarray, clip_thresh: float = 0.997) -> SignalStats:
    mono = x if x.ndim == 1 else x.mean(axis=1)
    has_nan = bool(np.isnan(mono).any())
    has_inf = bool(np.isinf(mono).any())
    finite = mono[np.isfinite(mono)] if (has_nan or has_inf) else mono
    if finite.size == 0:
        return SignalStats(0, 0, 0, 0, 0, has_nan, has_inf, 0)
    rms = float(np.sqrt(np.mean(finite.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(finite)))
    dc = float(np.mean(finite))
    clipped = np.abs(finite) >= clip_thresh
    clip_ratio = float(np.mean(clipped)) if finite.size else 0.0
    max_run = _max_run(clipped)
    crest = float(peak / rms) if rms > 1e-9 else 0.0
    return SignalStats(rms, peak, dc, clip_ratio, max_run, has_nan, has_inf, crest)


def _max_run(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return int(best)


def frame_energy_db(x: np.ndarray, sr: int, frame_ms: float = 30.0) -> Tuple[np.ndarray, int]:
    """Per-frame energy in dBFS; returns (energies_db, hop_samples)."""
    mono = x if x.ndim == 1 else x.mean(axis=1)
    hop = max(1, int(sr * frame_ms / 1000.0))
    n = mono.shape[0] // hop
    if n == 0:
        return np.array([-120.0]), hop
    frames = mono[: n * hop].reshape(n, hop)
    energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(energy + 1e-12), hop


def noise_floor_db(x: np.ndarray, sr: int) -> float:
    """Estimate noise floor as the 10th percentile of frame energy (dBFS)."""
    energies, _ = frame_energy_db(x, sr)
    return float(np.percentile(energies, 10))


def spectral_flatness(x: np.ndarray, sr: int) -> Optional[float]:
    """Geometric/arithmetic mean ratio of the power spectrum (0..1).

    High flatness => noise-like (hiss/static). Requires SciPy for the STFT; if
    unavailable returns ``None`` so the caller reports UNKNOWN rather than 0.
    """
    try:
        from scipy.signal import welch
    except Exception:
        return None
    mono = (x if x.ndim == 1 else x.mean(axis=1)).astype(np.float64)
    if mono.size < 256:
        return None
    _, psd = welch(mono, fs=sr, nperseg=min(1024, mono.size))
    psd = np.maximum(psd, 1e-20)
    gmean = np.exp(np.mean(np.log(psd)))
    amean = np.mean(psd)
    return float(gmean / amean) if amean > 0 else None


def hum_ratio(x: np.ndarray, sr: int, mains=(50.0, 60.0), bw: float = 2.0) -> Optional[float]:
    """Fraction of spectral energy near mains-hum harmonics (hum/buzz evidence)."""
    try:
        from scipy.signal import welch
    except Exception:
        return None
    mono = (x if x.ndim == 1 else x.mean(axis=1)).astype(np.float64)
    if mono.size < 512:
        return None
    freqs, psd = welch(mono, fs=sr, nperseg=min(4096, mono.size))
    total = float(np.sum(psd)) + 1e-20
    hum = 0.0
    for base in mains:
        for k in range(1, 5):
            f0 = base * k
            band = (freqs >= f0 - bw) & (freqs <= f0 + bw)
            hum += float(np.sum(psd[band]))
    return float(hum / total)
