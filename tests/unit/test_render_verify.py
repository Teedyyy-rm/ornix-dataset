"""Canonicalization + post-render verification tests — spec §Canonicalization,
§Post-render verification; tests T5, T15, T16, T17, T18, plus invariants I8/I10.
"""

import wave
from types import SimpleNamespace

import numpy as np
import pytest

from fixtures import synth
from ornix_dataset.dsp.audio import AudioBuffer
from ornix_dataset.dsp.decode import decode_to_float
from ornix_dataset.dsp.render import (RenderVerificationError, render_canonical_wav,
                                      verify_canonical_wav)

sf = pytest.importorskip("soundfile")


def _adm(action="IDENTITY"):
    """Minimal admission stub — render requires the measured action (never guesses)."""
    return SimpleNamespace(canonicalization_action=action)


def _probe_wav(path):
    with wave.open(path, "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()


def test_render_48k_kept_native_verifies(tmp_path):
    buf = AudioBuffer(synth.speechlike(48000, 2.0), 48000)
    out = str(tmp_path / "c.wav")
    sha, recipe = render_canonical_wav(buf, out, admission=_adm())
    fr, ch, sw, _ = _probe_wav(out)
    assert (fr, ch, sw) == (48000, 1, 2)
    assert recipe.output_sample_rate == 48000
    assert recipe.canonicalization_action == "IDENTITY"
    assert recipe.canonical_verify_status == "PASS"
    assert len(sha) == 64


def test_render_requires_admission(tmp_path):
    buf = AudioBuffer(synth.speechlike(24000, 1.0), 24000)
    with pytest.raises(ValueError):
        render_canonical_wav(buf, str(tmp_path / "c.wav"))


def test_t5_flac_24k_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ORNIX_WAV_ONLY", "0")  # legacy transcode path under test
    src = str(tmp_path / "s.flac")
    sf.write(src, synth.speechlike(24000, 2.0), 24000, subtype="PCM_16", format="FLAC")
    buf, _ = decode_to_float(src, mono=True)
    out = str(tmp_path / "c.wav")
    _, recipe = render_canonical_wav(buf, out, admission=_adm())
    assert recipe.canonicalization_action == "IDENTITY"
    assert _probe_wav(out)[:3] == (24000, 1, 2)


def test_t15_stereo_downmixed_to_mono(tmp_path):
    src = str(tmp_path / "st.wav")
    left = synth.speechlike(24000, 2.0, seed=1)
    right = synth.speechlike(24000, 2.0, seed=2)
    synth.write_wav_stereo(src, left, right, 24000)
    buf, _ = decode_to_float(src, mono=False)
    assert buf.channels == 2
    out = str(tmp_path / "c.wav")
    _, recipe = render_canonical_wav(buf, out, admission=_adm())
    assert _probe_wav(out)[1] == 1  # mono
    assert recipe.downmix_policy == "mean"


def test_t17_verify_catches_non_canonical(tmp_path):
    bad_sr = str(tmp_path / "sr.wav")
    synth.write_wav(bad_sr, synth.speechlike(16000, 1.0), 16000)
    ok, reasons = verify_canonical_wav(bad_sr, expected_sample_rate=24000)
    assert not ok and any("SR_MISMATCH" in r for r in reasons)
    # without a rate expectation (canonical tree copy path) only the shape is checked
    ok, _ = verify_canonical_wav(bad_sr)
    assert ok

    bad_ch = str(tmp_path / "ch.wav")
    synth.write_wav_stereo(bad_ch, synth.tone(24000, 1.0, 200),
                           synth.tone(24000, 1.0, 200), 24000)
    ok, reasons = verify_canonical_wav(bad_ch, expected_sample_rate=24000)
    assert not ok and any("NOT_MONO" in r for r in reasons)

    bad_enc = str(tmp_path / "enc.wav")
    with wave.open(bad_enc, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)  # 8-bit, not PCM16
        wf.setframerate(24000)
        wf.writeframes(np.zeros(24000, dtype=np.uint8).tobytes())
    ok, reasons = verify_canonical_wav(bad_enc)
    assert not ok and any("NOT_PCM16" in r for r in reasons)


def test_t17_verify_duration_mismatch(tmp_path):
    good = str(tmp_path / "g.wav")
    buf = AudioBuffer(synth.speechlike(24000, 1.0), 24000)
    render_canonical_wav(buf, good, admission=_adm())
    ok, _ = verify_canonical_wav(good, expected_sample_rate=24000,
                                 expected_n_samples=24000)
    assert ok
    ok, reasons = verify_canonical_wav(good, expected_sample_rate=24000,
                                       expected_n_samples=999)
    assert not ok and any("DURATION_MISMATCH" in r for r in reasons)


def test_t18_idempotent_canonical_sha_and_recipe(tmp_path):
    buf = AudioBuffer(synth.speechlike(44100, 2.0, seed=7), 44100)
    o1, o2 = str(tmp_path / "a.wav"), str(tmp_path / "b.wav")
    sha1, r1 = render_canonical_wav(buf, o1, admission=_adm())
    sha2, r2 = render_canonical_wav(buf, o2, admission=_adm())
    assert sha1 == sha2
    assert r1.to_dict() == r2.to_dict()
