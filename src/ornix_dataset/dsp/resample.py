"""Band-limited polyphase resampling (spec §0.1 audio output, §2 renderer).

Uses ``scipy.signal.resample_poly`` when available. If SciPy is missing we fall
back to linear interpolation but flag ``quality='linear-degraded'`` so the
renderer can refuse to publish a degraded canonical clip (fail-closed).
"""

from __future__ import annotations

from math import gcd
from typing import Tuple

import numpy as np

from .audio import AudioBuffer


def resample_poly_quality(x: np.ndarray, sr_in: int, sr_out: int) -> Tuple[np.ndarray, str]:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False), "identity"
    try:
        from scipy.signal import resample_poly  # optional dep

        g = gcd(sr_in, sr_out)
        up, down = sr_out // g, sr_in // g
        y = resample_poly(x.astype(np.float64), up, down, axis=0)
        return y.astype(np.float32), "scipy-resample_poly"
    except Exception:
        n_out = int(round(x.shape[0] * sr_out / sr_in))
        if n_out <= 1:
            return np.zeros((max(n_out, 0),), dtype=np.float32), "linear-degraded"
        t_in = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False)
        t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        if x.ndim == 1:
            y = np.interp(t_out, t_in, x)
        else:
            y = np.stack([np.interp(t_out, t_in, x[:, c]) for c in range(x.shape[1])], axis=1)
        return y.astype(np.float32), "linear-degraded"


def resample_to(buf: AudioBuffer, target_sr: int) -> Tuple[AudioBuffer, str]:
    y, method = resample_poly_quality(buf.samples, buf.sample_rate, target_sr)
    return AudioBuffer(y, target_sr), method
