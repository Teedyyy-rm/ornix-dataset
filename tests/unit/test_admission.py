"""Source Audio Admission tests (Level 1) — spec §Repair, tests T1..T9, T11..T13.

Rate-class + canonicalization-action classification from *measured* evidence, and
codec (lossy/lossless) classification from real codec names, never the extension.
"""

import numpy as np
import pytest

from fixtures import synth
from ornix_dataset.contracts.enums import CanonicalizationAction, SourceRateClass
from ornix_dataset.dsp import run_technical_validation
from ornix_dataset.dsp.admission import (AdmissionConfig, assess_source,
                                         classify_lossy)


def _wav(tmp_path, name, sig, sr):
    p = str(tmp_path / name)
    synth.write_wav(p, sig, sr)
    return p


def _assess(tmp_path, name, sr, dur=3.0, cfg=None):
    p = _wav(tmp_path, name, synth.speechlike(sr, dur), sr)
    report = run_technical_validation(p)
    return assess_source(report, cfg or AdmissionConfig()), report


def test_t1_wav_24k_identity(tmp_path):
    adm, _ = _assess(tmp_path, "n24.wav", 24000)
    assert adm.admitted
    assert adm.source_rate_class == SourceRateClass.NATIVE_OR_HIGHER.value
    assert adm.canonicalization_action == CanonicalizationAction.IDENTITY.value
    assert adm.source_sample_rate == 24000


def test_t2_48k_downsample(tmp_path):
    adm, _ = _assess(tmp_path, "n48.wav", 48000)
    assert adm.admitted
    assert adm.canonicalization_action == CanonicalizationAction.DOWNSAMPLE.value
    # target-aware bandwidth: a clean high-rate source is NOT flagged low-bandwidth
    assert not adm.low_bandwidth_suspected


def test_t3_44100_downsample(tmp_path):
    adm, _ = _assess(tmp_path, "n441.wav", 44100)
    assert adm.admitted
    assert adm.canonicalization_action == CanonicalizationAction.DOWNSAMPLE.value


def test_t4_32k_downsample(tmp_path):
    adm, _ = _assess(tmp_path, "n32.wav", 32000)
    assert adm.admitted
    assert adm.canonicalization_action == CanonicalizationAction.DOWNSAMPLE.value


def test_t6_22050_conditional_upsample(tmp_path):
    adm, _ = _assess(tmp_path, "n2205.wav", 22050)
    assert adm.admitted
    assert adm.source_rate_class == SourceRateClass.NEAR_TARGET_UPSAMPLE.value
    assert adm.canonicalization_action == CanonicalizationAction.UPSAMPLE_NEAR_TARGET.value
    assert adm.source_sample_rate == 22050  # provenance keeps the true source rate
    assert any("NOT_NATIVE_24K" in r for r in adm.reason_codes)


def test_t6_22050_rejected_when_upsample_disallowed(tmp_path):
    cfg = AdmissionConfig(allow_near_target_upsample=False)
    adm, _ = _assess(tmp_path, "n2205b.wav", 22050, cfg=cfg)
    assert not adm.admitted
    assert any("NEAR_TARGET_UPSAMPLE_NOT_ALLOWED" in r for r in adm.reason_codes)


def test_t7_16k_rejected_low_bandwidth(tmp_path):
    adm, _ = _assess(tmp_path, "n16.wav", 16000)
    assert not adm.admitted
    assert adm.source_rate_class == SourceRateClass.LOW_BANDWIDTH_SOURCE.value
    assert adm.canonicalization_action == CanonicalizationAction.REJECT_LOW_BANDWIDTH.value
    assert any("LOW_BANDWIDTH_SOURCE" in r for r in adm.reason_codes)


def test_t8_8k_rejected_narrowband(tmp_path):
    adm, _ = _assess(tmp_path, "n8.wav", 8000)
    assert not adm.admitted
    assert adm.source_rate_class == SourceRateClass.NARROWBAND_SOURCE.value
    assert adm.canonicalization_action == CanonicalizationAction.REJECT_NARROWBAND.value
    assert any("NARROWBAND_SOURCE" in r for r in adm.reason_codes)


def test_t9_fake_upsampled_24k_container_flagged(tmp_path):
    # 8 kHz signal band-limited-upsampled into a 24 kHz container: measured rate is
    # 24k (rate class passes) but bandwidth is narrow -> SUSPECTED_UPSAMPLED_SOURCE.
    p = _wav(tmp_path, "fake24.wav", synth.upsampled_lowband(24000, 2.0, src_sr=8000), 24000)
    report = run_technical_validation(p)
    adm = assess_source(report, AdmissionConfig())
    assert adm.low_bandwidth_suspected
    assert any("SUSPECTED_UPSAMPLED_SOURCE" in r for r in adm.reason_codes)


def test_missing_sample_rate_is_error():
    from ornix_dataset.dsp.technical import TechnicalReport
    report = TechnicalReport(valid=True, decision="VALID", measured_sample_rate=None)
    adm = assess_source(report, AdmissionConfig())
    assert not adm.admitted
    assert adm.canonicalization_action == CanonicalizationAction.ERROR.value


def test_classify_lossy_by_codec_not_extension():
    # lossless
    assert classify_lossy("pcm_s16le") is False
    assert classify_lossy("flac") is False
    assert classify_lossy("alac") is False          # T12: ALAC in an m4a container
    # lossy
    assert classify_lossy("mp3") is True            # T10
    assert classify_lossy("aac") is True            # T11: AAC in an m4a container
    assert classify_lossy("opus") is True           # T13
    assert classify_lossy("vorbis") is True         # T13
    assert classify_lossy("pcm_mulaw") is True      # telephony companded PCM in a .wav
    # unknown -> None (recorded, never used to reject on its own)
    assert classify_lossy("some_future_codec") is None
    assert classify_lossy(None) is None
