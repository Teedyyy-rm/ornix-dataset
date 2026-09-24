"""Source Audio Admission — Level 1 of the clean-HQ contract (spec §Repair).

Answers, from *measured* evidence (never the file extension), whether a source is
eligible for the Ornix clean 24 kHz release and, if so, what canonicalization
action it warrants:

    NATIVE_OR_HIGHER      -> IDENTITY (== 24k) or DOWNSAMPLE (> 24k)
    NEAR_TARGET_UPSAMPLE  -> UPSAMPLE_NEAR_TARGET (only if policy allows)
    LOW_BANDWIDTH_SOURCE  -> REJECT_LOW_BANDWIDTH (strict HQ default)
    NARROWBAND_SOURCE     -> REJECT_NARROWBAND (strict HQ default)

Upsampling a low-rate/narrowband source cannot restore lost bandwidth: a file is
never admitted merely because its output would say 24,000 Hz (invariants I1/I2/I9).
Codec is classified separately from container: extension != codec, WAV != lossless
PCM, and lossy->WAV does not restore information (invariants I3/I4/I5).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from ..contracts.enums import CanonicalizationAction, SourceRateClass
from .technical import TechnicalReport

# --- codec classification (real codec evidence, not the extension) -----------
# libsndfile/ffmpeg codec_name values. Companded/telephony PCM (mu-law, A-law,
# ADPCM, GSM) is treated as lossy: it is not linear PCM and discards amplitude
# resolution. Unknown codecs return None (unknown) — we never reject solely for
# being lossy, so an unknown codec is recorded, not used to gate.
_LOSSLESS = {
    "pcm_s8", "pcm_u8", "pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be",
    "pcm_s32le", "pcm_s32be", "pcm_f32le", "pcm_f32be", "pcm_f64le", "pcm_f64be",
    "flac", "alac", "wavpack", "tta", "truehd", "mlp",
}
_LOSSY = {
    "mp3", "mp3float", "aac", "aac_latm", "vorbis", "opus", "ac3", "eac3",
    "wmav1", "wmav2", "amr_nb", "amrnb", "amr_wb", "amrwb", "gsm", "gsm_ms",
    "g723_1", "g729", "nellymoser", "qdm2", "cook",
    "pcm_mulaw", "pcm_alaw", "adpcm_ms", "adpcm_ima_wav", "adpcm_ima_qt",
    "adpcm_g726", "adpcm_swf", "adpcm_yamaha",
}


def classify_lossy(codec_name: Optional[str]) -> Optional[bool]:
    """True if the codec is lossy/companded, False if lossless, None if unknown."""
    if not codec_name:
        return None
    c = codec_name.strip().lower()
    if c in _LOSSLESS:
        return False
    if c in _LOSSY:
        return True
    return None


@dataclass
class AdmissionConfig:
    """Clean-HQ source-admission policy (configs/audio_profile.yaml).

    Project-policy defaults, not universal audio standards. Any change is a
    versioned config change — do not scatter magic constants in code.
    """

    canonical_sample_rate: int = 24000
    native_min_sample_rate: int = 24000
    conditional_min_sample_rate: int = 22050
    low_bandwidth_min_sample_rate: int = 16000
    allow_near_target_upsample: bool = True
    reject_low_bandwidth_from_clean_hq: bool = True
    reject_narrowband_from_clean_hq: bool = True


@dataclass
class AdmissionReport:
    source_container: Optional[str]
    source_codec: Optional[str]
    source_lossy: Optional[bool]
    source_sample_rate: Optional[int]
    source_channels: Optional[int]
    source_rate_class: Optional[str]
    canonicalization_action: str
    effective_bandwidth_hz: Optional[float]
    low_bandwidth_suspected: bool
    admitted: bool
    reason_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _rate_class(sr: int, cfg: AdmissionConfig) -> SourceRateClass:
    if sr >= cfg.native_min_sample_rate:
        return SourceRateClass.NATIVE_OR_HIGHER
    if sr >= cfg.conditional_min_sample_rate:
        return SourceRateClass.NEAR_TARGET_UPSAMPLE
    if sr >= cfg.low_bandwidth_min_sample_rate:
        return SourceRateClass.LOW_BANDWIDTH_SOURCE
    return SourceRateClass.NARROWBAND_SOURCE


def assess_source(report: TechnicalReport,
                  cfg: Optional[AdmissionConfig] = None) -> AdmissionReport:
    """Classify a source from the technical report's *measured* evidence.

    Reuses the sample rate / codec / bandwidth already measured by the technical
    gate — no second decode. A missing/unmeasurable sample rate is an ERROR
    (fail-closed), never an ACCEPT.
    """
    cfg = cfg or AdmissionConfig()
    codec = report.codec_name
    container = report.format_name
    lossy = classify_lossy(codec)
    sr = report.measured_sample_rate
    reasons: List[str] = []

    if not sr or sr <= 0:
        reasons.append("SOURCE_SAMPLE_RATE_UNMEASURABLE")
        return AdmissionReport(
            source_container=container, source_codec=codec, source_lossy=lossy,
            source_sample_rate=sr, source_channels=report.measured_channels,
            source_rate_class=None,
            canonicalization_action=CanonicalizationAction.ERROR.value,
            effective_bandwidth_hz=report.effective_bandwidth_hz,
            low_bandwidth_suspected=report.low_bandwidth_suspected,
            admitted=False, reason_codes=reasons)

    rate_class = _rate_class(sr, cfg)
    reasons.append(f"SOURCE_RATE_CLASS:{rate_class.value}:sr={sr}")
    if lossy:
        reasons.append(f"SOURCE_LOSSY:codec={codec}")

    # a header that says 24/44.1/48k but whose measured bandwidth is narrow must
    # not silently pass — it is likely an upsampled low-bandwidth source.
    if report.low_bandwidth_suspected:
        reasons.append(
            f"SUSPECTED_UPSAMPLED_SOURCE:eff_bw={report.effective_bandwidth_hz}")

    if rate_class == SourceRateClass.NATIVE_OR_HIGHER:
        action = (CanonicalizationAction.IDENTITY if sr == cfg.canonical_sample_rate
                  else CanonicalizationAction.DOWNSAMPLE)
        admitted = True
    elif rate_class == SourceRateClass.NEAR_TARGET_UPSAMPLE:
        if cfg.allow_near_target_upsample:
            action = CanonicalizationAction.UPSAMPLE_NEAR_TARGET
            admitted = True
            reasons.append("CONDITIONAL_UPSAMPLE_NOT_NATIVE_24K")
        else:
            action = CanonicalizationAction.UPSAMPLE_NEAR_TARGET
            admitted = False
            reasons.append("NEAR_TARGET_UPSAMPLE_NOT_ALLOWED")
    elif rate_class == SourceRateClass.LOW_BANDWIDTH_SOURCE:
        action = CanonicalizationAction.REJECT_LOW_BANDWIDTH
        admitted = not cfg.reject_low_bandwidth_from_clean_hq
        reasons.append("LOW_BANDWIDTH_SOURCE")
    else:  # NARROWBAND_SOURCE
        action = CanonicalizationAction.REJECT_NARROWBAND
        admitted = not cfg.reject_narrowband_from_clean_hq
        reasons.append("NARROWBAND_SOURCE")

    return AdmissionReport(
        source_container=container, source_codec=codec, source_lossy=lossy,
        source_sample_rate=sr, source_channels=report.measured_channels,
        source_rate_class=rate_class.value,
        canonicalization_action=action.value,
        effective_bandwidth_hz=report.effective_bandwidth_hz,
        low_bandwidth_suspected=report.low_bandwidth_suspected,
        admitted=admitted, reason_codes=reasons)
