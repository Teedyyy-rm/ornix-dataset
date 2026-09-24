"""DSP: probe, decode, technical validation, resample, features, canonical render."""

from .admission import (AdmissionConfig, AdmissionReport, assess_source,
                        classify_lossy)
from .audio import AudioBuffer
from .decode import DecodeError, decode_to_float, ffprobe_info, probe_available
from .resample import resample_poly_quality, resample_to
from .render import (RenderRecipe, RenderVerificationError, render_canonical_wav,
                     verify_canonical_wav, write_wav_pcm16)
from .technical import TechnicalReport, run_technical_validation

__all__ = [
    "AdmissionConfig",
    "AdmissionReport",
    "assess_source",
    "classify_lossy",
    "AudioBuffer",
    "DecodeError",
    "decode_to_float",
    "ffprobe_info",
    "probe_available",
    "resample_poly_quality",
    "resample_to",
    "RenderRecipe",
    "RenderVerificationError",
    "render_canonical_wav",
    "verify_canonical_wav",
    "write_wav_pcm16",
    "TechnicalReport",
    "run_technical_validation",
]
