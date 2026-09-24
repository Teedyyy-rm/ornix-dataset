"""Technical WAV gate (spec §2, Phase 2, tests T-008..T-010).

Rejects technically broken files and measures *real* properties (never trusts
the header). Detects header/real sample-rate mismatch and effective bandwidth so
an 8 kHz signal stored with a 24 kHz header is flagged, not accepted as native.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .audio import AudioBuffer
from .decode import DecodeError, decode_to_float, ffprobe_info
from .features import SignalStats, signal_stats


@dataclass
class TechnicalThresholds:
    min_duration_s: float = 0.2
    max_duration_s: float = 12.0
    max_duration_pre_segment_s: float = 600.0  # longer -> segmentation, not reject
    max_dc_offset: float = 0.02
    max_clipping_ratio: float = 0.01
    min_rms: float = 1e-4  # below -> effectively silent / no speech level
    duration_tolerance_s: float = 0.05
    bandwidth_ratio_warn: float = 0.45  # eff_bw / (sr/2) below this -> low bandwidth


@dataclass
class TechnicalReport:
    valid: bool
    decision: str  # VALID | REJECT_TECH | ERROR | SEGMENT_CANDIDATE
    reason_codes: List[str] = field(default_factory=list)
    measured_sample_rate: Optional[int] = None
    measured_channels: Optional[int] = None
    measured_duration_s: Optional[float] = None
    declared_duration_s: Optional[float] = None
    codec_name: Optional[str] = None
    format_name: Optional[str] = None
    effective_bandwidth_hz: Optional[float] = None
    low_bandwidth_suspected: bool = False
    stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _effective_bandwidth(x: np.ndarray, sr: int) -> Optional[float]:
    """Highest frequency with meaningful energy (-50 dB from peak PSD).

    This detects the spectral *cliff* left by upsampling/band-limiting (T-008):
    an 8 kHz signal in a 24 kHz container has ~no energy above ~4 kHz.
    """
    try:
        from scipy.signal import welch
    except Exception:
        return None
    mono = (x if x.ndim == 1 else x.mean(axis=1)).astype(np.float64)
    if mono.size < 512:
        return None
    freqs, psd = welch(mono, fs=sr, nperseg=min(4096, mono.size))
    if psd.max() <= 0:
        return 0.0
    thresh = psd.max() * 1e-5  # -50 dB
    above = np.nonzero(psd > thresh)[0]
    if above.size == 0:
        return 0.0
    return float(freqs[above[-1]])


def run_technical_validation(
    path: str,
    thresholds: Optional[TechnicalThresholds] = None,
    declared_duration_s: Optional[float] = None,
) -> TechnicalReport:
    t = thresholds or TechnicalThresholds()
    reasons: List[str] = []
    try:
        info = ffprobe_info(path)
    except DecodeError as e:
        return TechnicalReport(False, "ERROR", [f"PROBE_FAILED:{e}"])
    try:
        buf, _decoder = decode_to_float(path, mono=False)
    except DecodeError as e:
        return TechnicalReport(False, "REJECT_TECH", [f"DECODE_FAILED:{e}"],
                               measured_sample_rate=info.get("sample_rate"),
                               codec_name=info.get("codec_name"),
                               format_name=info.get("format_name"))
    return _evaluate(buf, info, t, declared_duration_s, reasons)


def _evaluate(
    buf: AudioBuffer,
    info: Dict[str, Any],
    t: TechnicalThresholds,
    declared_duration_s: Optional[float],
    reasons: List[str],
) -> TechnicalReport:
    stats: SignalStats = signal_stats(buf.samples)
    sr = buf.sample_rate
    dur = buf.duration_s
    eff_bw = _effective_bandwidth(buf.samples, sr)
    low_bw = False

    if stats.has_nan:
        reasons.append("NAN_SAMPLES")
    if stats.has_inf:
        reasons.append("INF_SAMPLES")
    if buf.n_samples == 0:
        reasons.append("EMPTY_AUDIO")
    if stats.rms < t.min_rms:
        reasons.append("SILENT_OR_NO_SPEECH_LEVEL")
    if stats.clipping_ratio > t.max_clipping_ratio:
        reasons.append(f"CLIPPING:{stats.clipping_ratio:.4f}")
    if abs(stats.dc_offset) > t.max_dc_offset:
        reasons.append(f"DC_OFFSET:{stats.dc_offset:.4f}")

    # header vs measured sample-rate / bandwidth mismatch (T-008)
    if eff_bw is not None and sr > 0:
        nyq = sr / 2.0
        if eff_bw < t.bandwidth_ratio_warn * nyq:
            low_bw = True
            reasons.append(f"LOW_BANDWIDTH_SUSPECTED:eff_bw={eff_bw:.0f}Hz<nyq={nyq:.0f}Hz")

    # duration bounds
    decision = "VALID"
    if buf.n_samples == 0 or stats.has_nan or stats.has_inf:
        decision = "REJECT_TECH"
    elif dur < t.min_duration_s:
        reasons.append(f"TOO_SHORT:{dur:.3f}s<{t.min_duration_s}s")
        decision = "REJECT_TECH"
    elif dur > t.max_duration_s:
        if dur <= t.max_duration_pre_segment_s:
            reasons.append(f"OVER_MAX_DURATION_SEGMENT_CANDIDATE:{dur:.3f}s")
            decision = "SEGMENT_CANDIDATE"
        else:
            reasons.append(f"TOO_LONG:{dur:.3f}s>{t.max_duration_pre_segment_s}s")
            decision = "REJECT_TECH"

    hard = {"NAN_SAMPLES", "INF_SAMPLES", "EMPTY_AUDIO", "SILENT_OR_NO_SPEECH_LEVEL"}
    if any(r in hard or r.startswith("CLIPPING") for r in reasons):
        decision = "REJECT_TECH"

    # declared vs measured duration reconciliation (T-009)
    if declared_duration_s is not None and dur > 0:
        if abs(declared_duration_s - dur) > max(t.duration_tolerance_s, 0.02 * dur):
            reasons.append(
                f"DURATION_MISMATCH:declared={declared_duration_s:.3f}s,measured={dur:.3f}s"
            )

    valid = decision in ("VALID", "SEGMENT_CANDIDATE")
    return TechnicalReport(
        valid=valid,
        decision=decision,
        reason_codes=reasons,
        measured_sample_rate=sr,
        measured_channels=buf.channels,
        measured_duration_s=round(dur, 6),
        declared_duration_s=declared_duration_s,
        codec_name=info.get("codec_name"),
        format_name=info.get("format_name"),
        effective_bandwidth_hz=round(eff_bw, 2) if eff_bw is not None else None,
        low_bandwidth_suspected=low_bw,
        stats=stats.to_dict(),
    )
