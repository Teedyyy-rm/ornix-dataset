"""Canonical audio renderer: mono signed PCM16 WAV at the source's NATIVE rate
(spec §0.1, §2).

Never resamples: decimating a clean high-rate source discards real content and
upsampling cannot restore lost bandwidth, so the output keeps the measured
source sample rate. Never normalises loudness or denoises implicitly either.
Downmix policy is explicit and phase-cancellation is checked upstream. 16-bit
conversion has a clipping guard. Every produced file is re-opened and proven
WAV/mono/PCM16 at the expected rate before it is returned (post-render
verification, invariant I10); a file that fails is deleted and the render fails
closed rather than yielding a bad canonical artifact.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..util.hashing import sha256_file
from .audio import AudioBuffer

CANONICAL_CHANNELS = 1
CANONICAL_ENCODING = "PCM_S16LE"


class RenderVerificationError(RuntimeError):
    """Raised when a written canonical WAV fails post-render verification."""


@dataclass
class RenderRecipe:
    output_sample_rate: int
    channels: int
    encoding: str
    downmix_policy: str
    clipped_samples_on_quantize: int
    source_sample_rate: int
    canonicalization_action: str = "UNKNOWN"
    source_rate_class: Optional[str] = None
    source_container: Optional[str] = None
    source_codec: Optional[str] = None
    source_lossy: Optional[bool] = None
    effective_bandwidth_hz: Optional[float] = None
    low_bandwidth_suspected: bool = False
    canonical_verify_status: str = "UNVERIFIED"

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


def verify_canonical_wav(
    path: str,
    expected_sample_rate: Optional[int] = None,
    expected_n_samples: Optional[int] = None,
    sample_tolerance: int = 1,
) -> Tuple[bool, List[str]]:
    """Re-open a written WAV and prove it is WAV/mono/PCM16 at the expected rate.

    Trusts the file actually written, never the requested parameters. Returns
    (ok, reasons). ``expected_sample_rate`` is checked when given — the release
    path always passes the measured source rate; the canonical tree copy path
    has no per-file rate in its six-field metadata and passes ``None``.
    ``expected_n_samples`` (if given) is checked within a small tolerance;
    NaN/Inf cannot occur in decoded int16 but the read is validated.
    """
    reasons: List[str] = []
    try:
        with wave.open(path, "rb") as wf:
            if expected_sample_rate is not None and \
                    wf.getframerate() != expected_sample_rate:
                reasons.append(
                    f"SR_MISMATCH:{wf.getframerate()}!={expected_sample_rate}")
            if wf.getnchannels() != CANONICAL_CHANNELS:
                reasons.append(f"NOT_MONO:{wf.getnchannels()}")
            if wf.getsampwidth() != 2:
                reasons.append(f"NOT_PCM16:sampwidth={wf.getsampwidth()}")
            n = wf.getnframes()
            raw = wf.readframes(n)
    except Exception as e:  # unreadable / not a WAV
        return False, [f"WAV_READBACK_FAILED:{e}"]

    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size != n:
        reasons.append(f"FRAME_COUNT_MISMATCH:{samples.size}!={n}")
    if not np.isfinite(samples.astype(np.float64)).all():
        reasons.append("NONFINITE_SAMPLES")
    if expected_n_samples is not None and abs(n - expected_n_samples) > sample_tolerance:
        reasons.append(f"DURATION_MISMATCH:{n}!={expected_n_samples}(+-{sample_tolerance})")
    return (not reasons), reasons


def render_canonical_wav(
    buf: AudioBuffer, out_path: str, downmix_policy: str = "mean", admission: Any = None
) -> Tuple[str, RenderRecipe]:
    """Render ``buf`` to a mono PCM16 WAV at its native rate. Returns (sha256, recipe).

    decode-once -> explicit mono downmix -> PCM16 -> WAV -> post-render
    verification. The canonicalization action comes from the measured admission
    report and is never guessed: a missing admission is a programming error and
    fails closed. The written file must verify as WAV/mono/PCM16 at the measured
    source rate, or it is deleted and the render fails closed.
    """
    if admission is None:
        raise ValueError(
            "admission report required: canonicalization action must come from "
            "measured admission, never guessed")
    src_sr = buf.sample_rate
    if not src_sr or int(src_sr) <= 0:
        raise ValueError(f"invalid source sample rate {src_sr!r}; refusing to render")
    mono = buf.to_mono(downmix_policy)
    pcm16, clipped = _float_to_pcm16(mono.samples)
    write_wav_pcm16(out_path, pcm16, src_sr)

    ok, verify_reasons = verify_canonical_wav(
        out_path, expected_sample_rate=src_sr, expected_n_samples=pcm16.shape[0])
    if not ok:
        try:  # quarantine only the generated invalid artifact; never the source
            os.remove(out_path)
        except OSError:
            pass
        raise RenderVerificationError(
            f"canonical post-render verification failed: {verify_reasons}")

    recipe = RenderRecipe(
        output_sample_rate=src_sr,
        channels=CANONICAL_CHANNELS,
        encoding=CANONICAL_ENCODING,
        downmix_policy=downmix_policy if buf.channels > 1 else "none",
        clipped_samples_on_quantize=clipped,
        source_sample_rate=src_sr,
        canonicalization_action=admission.canonicalization_action,
        source_rate_class=getattr(admission, "source_rate_class", None),
        source_container=getattr(admission, "source_container", None),
        source_codec=getattr(admission, "source_codec", None),
        source_lossy=getattr(admission, "source_lossy", None),
        effective_bandwidth_hz=getattr(admission, "effective_bandwidth_hz", None),
        low_bandwidth_suspected=bool(getattr(admission, "low_bandwidth_suspected", False)),
        canonical_verify_status="PASS",
    )
    return sha256_file(out_path), recipe
