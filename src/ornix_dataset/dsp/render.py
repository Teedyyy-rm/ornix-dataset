"""Canonical audio renderer: 24 kHz mono signed PCM16 WAV (spec §0.1, §2).

Never normalises loudness or denoises implicitly. Downmix policy is explicit and
phase-cancellation is checked upstream. 16-bit conversion has a clipping guard.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

import numpy as np

from ..util.hashing import sha256_file
from .audio import AudioBuffer
from .resample import resample_to

CANONICAL_SR = 24000
CANONICAL_CHANNELS = 1
CANONICAL_ENCODING = "PCM_S16LE"


@dataclass
class RenderRecipe:
    target_sample_rate: int
    channels: int
    encoding: str
    downmix_policy: str
    resample_method: str
    clipped_samples_on_quantize: int
    source_sample_rate: int
    upsampled_from_below_target: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _float_to_pcm16(x: np.ndarray) -> Tuple[np.ndarray, int]:
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = int(np.sum(np.abs(x) > 1.0))
    x = np.clip(x, -1.0, 1.0)
    # symmetric scaling to int16 range with round-half-away-from-zero
    scaled = np.where(x >= 0, x * 32767.0, x * 32768.0)
    return np.round(scaled).astype("<i2"), clipped


def write_wav_pcm16(path: str, mono_int16: np.ndarray, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(mono_int16.tobytes())


def render_canonical_wav(
    buf: AudioBuffer, out_path: str, downmix_policy: str = "mean"
) -> Tuple[str, RenderRecipe]:
    """Render ``buf`` to a canonical 24k mono PCM16 WAV. Returns (audio_sha256, recipe)."""
    src_sr = buf.sample_rate
    mono = buf.to_mono(downmix_policy)
    resampled, method = resample_to(mono, CANONICAL_SR)
    if method == "linear-degraded":
        raise RuntimeError(
            "band-limited resampler unavailable (install scipy); refusing to render "
            "a degraded canonical clip (fail-closed)"
        )
    pcm16, clipped = _float_to_pcm16(resampled.samples)
    write_wav_pcm16(out_path, pcm16, CANONICAL_SR)
    recipe = RenderRecipe(
        target_sample_rate=CANONICAL_SR,
        channels=CANONICAL_CHANNELS,
        encoding=CANONICAL_ENCODING,
        downmix_policy=downmix_policy if buf.channels > 1 else "none",
        resample_method=method,
        clipped_samples_on_quantize=clipped,
        source_sample_rate=src_sr,
        upsampled_from_below_target=src_sr < CANONICAL_SR,
    )
    return sha256_file(out_path), recipe
