"""Policy engine tests (spec §10 T-001..T-007, fail-closed)."""

import pytest

from ornix_dataset.contracts.enums import (DecisionState, MeasurementStatus,
                                           NoiseLabel, RightsStatus, Severity)
from ornix_dataset.contracts.quality import NoiseEvent, QualityEvidence
from ornix_dataset.contracts.source import SourceRecord
from ornix_dataset.curation.policy import PolicyConfig, PolicyEngine

PUBLIC_REQUIRED = ["has_speech", "no_clipping", "no_confirmed_music_overlap",
                   "no_severe_noise_overlap", "no_interfering_speech", "quality_ok",
                   "transcript_ok", "rights_ok"]


def _src(rights=RightsStatus.REDISTRIBUTION_APPROVED, redist=True):
    return SourceRecord(source_id="S", source_uri="file://x", source_revision="r",
                        original_file_id="x", source_sha256="a" * 64, source_bytes=1,
                        rights_status=rights, redistribution_permitted=redist)


def _ev(**kw):
    d = dict(segment_id="seg", source_sha256="a" * 64, interval_start_sample=0,
             interval_end_sample=24000, analysis_sample_rate=16000, speech_ratio=0.9,
             clipping_ratio=0.0, transcript_match_status=MeasurementStatus.OK,
             quality_status=MeasurementStatus.OK, speaker_status=MeasurementStatus.OK)
    d.update(kw)
    return QualityEvidence(**d)


def _engine(**over):
    cfg = PolicyConfig(policy_version="v1", required_checks=PUBLIC_REQUIRED,
                       single_speaker_required=True, **over)
    return PolicyEngine(cfg)


ALL_AVAIL = {"music": True, "quality": True, "speaker": True}


def test_t001_clean_accept():
    d = _engine().decide(_ev(), _src(), ALL_AVAIL)
    assert d.decision == DecisionState.ACCEPT


def test_duration_gate_rejects_out_of_window():
    req = ["has_speech", "duration_ok", "rights_ok"]

    def eng():
        return PolicyEngine(PolicyConfig(policy_version="v1", required_checks=req,
                                         min_duration_s=3.0, max_duration_s=12.0))

    # 5.0s (48000 @ 16k) -> in window -> ACCEPT
    ok = eng().decide(_ev(interval_start_sample=0, interval_end_sample=48000,
                          analysis_sample_rate=16000), _src(), ALL_AVAIL)
    assert ok.decision == DecisionState.ACCEPT
    # 2.5s -> below floor -> REJECT
    short = eng().decide(_ev(interval_start_sample=0, interval_end_sample=40000,
                             analysis_sample_rate=16000), _src(), ALL_AVAIL)
    assert short.decision == DecisionState.REJECT
    assert any("DURATION_TOO_SHORT" in r for r in short.reason_codes)
    # 13.0s -> above ceiling -> REJECT
    long = eng().decide(_ev(interval_start_sample=0, interval_end_sample=208000,
                            analysis_sample_rate=16000), _src(), ALL_AVAIL)
    assert long.decision == DecisionState.REJECT
    assert any("DURATION_TOO_LONG" in r for r in long.reason_codes)


def test_t002_music_gate_unavailable_is_review():
    d = _engine().decide(_ev(), _src(), {"music": False, "quality": True, "speaker": True})
    assert d.decision == DecisionState.REVIEW
    assert "MUSIC_GATE_UNAVAILABLE" in d.reason_codes


def test_t002_confirmed_music_overlap_reject():
    ev = _ev(noise_events=[NoiseEvent(NoiseLabel.MUSIC_BACKGROUND, 0, 2, True,
                                      Severity.N2, 0.9, 0.9, "panns", "r")])
    assert _engine().decide(ev, _src(), ALL_AVAIL).decision == DecisionState.REJECT


def test_t004_hiss_overlapping_speech_rejects():
    # hiss that overlaps speech above the configured severity cap -> REJECT
    ev = _ev(noise_events=[NoiseEvent(NoiseLabel.HISS_STATIC, 0, 2, True,
                                      Severity.N3, 0.9, 0.8, "dsp", "r")])
    assert _engine().decide(ev, _src(), ALL_AVAIL).decision == DecisionState.REJECT


def test_t004_hiss_in_silence_and_snr_unknown_still_accepts():
    # same hiss confined to silence (does not overlap speech) does not fail the
    # overlap gate; SNR UNKNOWN is not a required check and must not auto-fail
    ev = _ev(snr_status=MeasurementStatus.UNKNOWN, estimated_snr_db=None,
             noise_events=[NoiseEvent(NoiseLabel.HISS_STATIC, 2.5, 3.0, False,
                                      Severity.N3, 0.9, 0.8, "dsp", "r")])
    assert _engine().decide(ev, _src(), ALL_AVAIL).decision == DecisionState.ACCEPT


def test_t006_high_quality_metric_cannot_override_confirmed_defect():
    # post-denoise clip: quality model reports pristine SIG/BAK, but a confirmed
    # noise artifact overlaps speech -> a high metric must NOT force ACCEPT
    ev = _ev(quality_status=MeasurementStatus.OK, sig=4.8, bak=4.9, ovrl=4.7,
             noise_events=[NoiseEvent(NoiseLabel.HISS_STATIC, 0, 2, True,
                                      Severity.N3, 0.95, 0.9, "dsp", "r")])
    assert _engine().decide(ev, _src(), ALL_AVAIL).decision == DecisionState.REJECT



def test_t005_interfering_speech_reject():
    ev = _ev(speaker_overlap_intervals=[[0.5, 1.0]])
    assert _engine().decide(ev, _src(), ALL_AVAIL).decision == DecisionState.REJECT


def test_t007_clipping_reject():
    ev = _ev(noise_events=[NoiseEvent(NoiseLabel.CLIPPING_DISTORTION, 0, 1, False,
                                      Severity.N3, 0.9, 0.05, "dsp", "r")])
    assert _engine().decide(ev, _src(), ALL_AVAIL).decision == DecisionState.REJECT


def test_t012_rights_unknown_license_review():
    d = _engine().decide(_ev(), _src(RightsStatus.LICENSE_REVIEW, False), ALL_AVAIL)
    assert d.decision == DecisionState.LICENSE_REVIEW


def test_no_speech_reject():
    assert _engine().decide(_ev(speech_ratio=0.1), _src(), ALL_AVAIL).decision == DecisionState.REJECT


def test_transcript_mismatch_reject():
    ev = _ev(transcript_match_status=MeasurementStatus.ERROR)
    assert _engine().decide(ev, _src(), ALL_AVAIL).decision == DecisionState.REJECT


def test_determinism():
    e = _engine()
    d1 = e.decide(_ev(), _src(), ALL_AVAIL)
    d2 = e.decide(_ev(), _src(), ALL_AVAIL)
    assert d1.decision == d2.decision and d1.reason_codes == d2.reason_codes


def test_quality_unavailable_review():
    d = _engine().decide(_ev(), _src(), {"music": True, "quality": False, "speaker": True})
    assert d.decision == DecisionState.REVIEW
