"""Synthetic audio fixtures for tests (spec §10 test matrix).

Deterministic generators (fixed seed) covering clean speech-like signals, tones,
noise, music-like mixtures, clipping, silence and multi-rate cases.
"""

from __future__ import annotations

import wave
from typing import Optional

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def speechlike(sr: int = 24000, dur: float = 3.0, seed: int = 0, f0: float = 130.0) -> np.ndarray:
    """Voiced-ish signal: harmonic stack + fricative bursts (broadband)."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = _rng(seed)
    sig = np.zeros(n)
    for k in range(1, 12):
        sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, np.pi))
    env = 0.35 * (1 + np.sin(2 * np.pi * 3.0 * t)) + 0.5
    sig = sig * env
    # fricative-like broadband bursts give realistic high-frequency energy
    fric = rng.standard_normal(n)
    burst = ((np.sin(2 * np.pi * 1.5 * t) > 0.7)).astype(float)
    sig += 0.15 * fric * burst
    return _norm(sig, 0.6)


def tone(sr: int, dur: float, freq: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def white_noise(sr: int, dur: float, amp: float = 0.1, seed: int = 1) -> np.ndarray:
    return (amp * _rng(seed).standard_normal(int(sr * dur))).astype(np.float64)


def musiclike(sr: int, dur: float, seed: int = 2) -> np.ndarray:
    """Chord + rhythm, distinct from harmonic speech (broadband, steady)."""
    t = np.arange(int(sr * dur)) / sr
    chord = sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0, 523.2))
    beat = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 2.0 * t)))
    return _norm(chord * beat + 0.02 * _rng(seed).standard_normal(t.size), 0.5)


def silence(sr: int, dur: float) -> np.ndarray:
    return np.zeros(int(sr * dur))


def upsampled_lowband(target_sr: int, dur: float, src_sr: int = 8000, seed: int = 0) -> np.ndarray:
    """Narrowband signal band-limited-upsampled to target_sr (spectral cliff)."""
    from math import gcd
    from scipy.signal import resample_poly

    sig = speechlike(src_sr, dur, seed=seed)
    g = gcd(src_sr, target_sr)
    return resample_poly(sig, target_sr // g, src_sr // g)


def clipped(sig: np.ndarray, level: float = 0.5) -> np.ndarray:
    return np.clip(sig / max(np.max(np.abs(sig)), 1e-9) * 1.5, -level * 3, level * 3).clip(-1, 1)


def _norm(x: np.ndarray, peak: float) -> np.ndarray:
    m = np.max(np.abs(x))
    return (x / m * peak) if m > 0 else x


def write_wav(path: str, sig: np.ndarray, sr: int, channels: int = 1) -> None:
    sig = np.clip(sig, -1.0, 1.0)
    pcm = np.round(sig * 32767.0).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm.reshape(-1, 1), channels, axis=1).reshape(-1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def write_wav_stereo(path: str, left: np.ndarray, right: np.ndarray, sr: int) -> None:
    n = min(left.size, right.size)
    inter = np.empty(n * 2, dtype="<i2")
    inter[0::2] = np.round(np.clip(left[:n], -1, 1) * 32767).astype("<i2")
    inter[1::2] = np.round(np.clip(right[:n], -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(inter.tobytes())
