"""Detector registry, availability (fail-closed), windowing, VAD, DSP noise."""

import numpy as np

from fixtures import synth
from ornix_dataset.detectors import (DetectorKind, dilate_intervals, plan_windows,
                                     registry, union_intervals)
from ornix_dataset.detectors.windowing import intersect_intervals, total_duration


def test_registry_lists_adapters():
    assert "energy" in registry.names(DetectorKind.VAD)
    assert "dsp" in registry.names(DetectorKind.NOISE)
    assert "dnsmos" in registry.names(DetectorKind.QUALITY)
    assert "pyannote" in registry.names(DetectorKind.SPEAKER)


def test_licensed_adapters_fail_closed():
    assert not registry.create(DetectorKind.NOISE, "panns").available
    assert not registry.create(DetectorKind.QUALITY, "dnsmos").available
    assert not registry.create(DetectorKind.SPEAKER, "pyannote").available
    assert not registry.create(DetectorKind.VAD, "silero").available


def test_dnsmos_returns_unknown_not_zero():
    out = registry.create(DetectorKind.QUALITY, "dnsmos").infer(np.zeros(1000), 24000)
    assert out["status"] == "UNKNOWN"
    assert out["sig"] is None and out["bak"] is None


def test_energy_vad_covers_speech():
    vad = registry.create(DetectorKind.VAD, "energy")
    iv = vad.infer(synth.speechlike(24000, 4.0), 24000)
    assert total_duration(iv) / 4.0 > 0.7
    assert vad.info().is_heuristic


def test_dsp_noise_flags_hiss():
    nd = registry.create(DetectorKind.NOISE, "dsp")
    noisy = synth.speechlike(24000, 3.0) + synth.white_noise(24000, 3.0, amp=0.25)
    events = nd.infer(noisy, 24000, [[0, 3]])
    assert any(e.label.value == "HISS_STATIC" for e in events)


def test_dsp_noise_never_emits_music():
    nd = registry.create(DetectorKind.NOISE, "dsp")
    events = nd.infer(synth.musiclike(24000, 3.0), 24000, [[0, 3]])
    assert not any(e.label.value == "MUSIC_BACKGROUND" for e in events)


def test_windowing_and_interval_algebra():
    plan = plan_windows(10.0, window_s=1.5, overlap=0.5)
    assert plan.windows[0][0] == 0.0 and plan.windows[-1][1] <= 10.0
    assert union_intervals([[0, 1], [0.5, 2], [3, 4]]) == [[0, 2], [3, 4]]
    assert intersect_intervals([[0, 5]], [[2, 8]]) == [[2, 5]]
    assert total_duration([[0, 1], [2, 3]]) == 2.0
    dil = dilate_intervals([[1, 2]], 0.5, 10)
    assert dil == [[0.5, 2.5]]
