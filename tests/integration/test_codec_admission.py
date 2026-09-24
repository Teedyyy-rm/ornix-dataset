"""Real-codec source-admission integration — spec §Format/codec policy; tests
T10 (MP3), T11 (AAC/m4a), T12 (ALAC/m4a), T13 (Opus/Vorbis), T14 (corrupt),
T19 (originals immutable), T20 (release invariant).

Uses real ffmpeg encode + ffprobe/decoder so classification rests on the actual
codec stream, never the file extension. CI installs ffmpeg so these run for real;
they self-skip only when ffmpeg is genuinely absent.
"""

import os
import shutil
import subprocess

import pytest

from fixtures import synth
from ornix_dataset.contracts.release import ReleaseRow
from ornix_dataset.dsp import run_technical_validation
from ornix_dataset.dsp.admission import AdmissionConfig, assess_source
from ornix_dataset.dsp.decode import decode_to_float
from ornix_dataset.dsp.render import render_canonical_wav, verify_canonical_wav
from ornix_dataset.util.hashing import sha256_file

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg required for real codec admission tests")


def _wav(tmp_path, sr=44100, dur=2.0):
    p = str(tmp_path / "src.wav")
    synth.write_wav(p, synth.speechlike(sr, dur), sr)
    return p


def _encode(src_wav, out, *extra):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src_wav, *extra, out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _classify(path, cfg=None):
    report = run_technical_validation(path)
    return report, assess_source(report, cfg or AdmissionConfig())


def test_t10_mp3_decodes_lossy_provenance_survives(tmp_path):
    src = _wav(tmp_path, sr=44100)
    mp3 = _encode(src, str(tmp_path / "a.mp3"), "-codec:a", "libmp3lame", "-b:a", "192k")
    report, adm = _classify(mp3)
    assert adm.source_lossy is True and adm.source_codec == "mp3"
    assert adm.admitted and adm.canonicalization_action == "DOWNSAMPLE"
    buf, _ = decode_to_float(mp3, mono=True)
    _, recipe = render_canonical_wav(buf, str(tmp_path / "c.wav"), admission=adm)
    # conversion to WAV must NOT clear the lossy provenance
    assert recipe.source_lossy is True
    assert verify_canonical_wav(str(tmp_path / "c.wav"))[0]


def test_t11_aac_m4a_classified_by_codec(tmp_path):
    src = _wav(tmp_path, sr=44100)
    m4a = _encode(src, str(tmp_path / "a.m4a"), "-codec:a", "aac", "-b:a", "192k")
    _report, adm = _classify(m4a)
    assert adm.source_codec == "aac" and adm.source_lossy is True


def test_t12_alac_m4a_classified_lossless(tmp_path):
    src = _wav(tmp_path, sr=44100)
    m4a = _encode(src, str(tmp_path / "l.m4a"), "-codec:a", "alac")
    _report, adm = _classify(m4a)
    # same .m4a container family as AAC, but ALAC is lossless
    assert adm.source_codec == "alac" and adm.source_lossy is False


def test_t13_opus_and_vorbis_classified_by_codec(tmp_path):
    src = _wav(tmp_path, sr=48000)
    opus = _encode(src, str(tmp_path / "o.opus"), "-codec:a", "libopus", "-b:a", "96k")
    _r1, a_opus = _classify(opus)
    assert a_opus.source_codec == "opus" and a_opus.source_lossy is True

    ogg = _encode(src, str(tmp_path / "v.ogg"), "-codec:a", "libvorbis", "-q:a", "5")
    _r2, a_vorb = _classify(ogg)
    assert a_vorb.source_codec == "vorbis" and a_vorb.source_lossy is True


def test_t14_corrupt_file_no_output(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFFxxxxWAVEnonsense-not-audio" * 4)
    report = run_technical_validation(str(bad))
    assert report.decision in ("ERROR", "REJECT_TECH")
    assert not report.valid


def test_t19_original_source_bytes_immutable(tmp_path):
    src = _wav(tmp_path, sr=48000)
    mp3 = _encode(src, str(tmp_path / "a.mp3"), "-codec:a", "libmp3lame", "-b:a", "192k")
    before = sha256_file(mp3)
    buf, _ = decode_to_float(mp3, mono=True)
    render_canonical_wav(buf, str(tmp_path / "c.wav"))
    assert sha256_file(mp3) == before  # canonicalization never touches the source


def test_t20_release_row_invariant_holds(tmp_path):
    import wave

    src = _wav(tmp_path, sr=48000)
    buf, _ = decode_to_float(src, mono=True)
    out = str(tmp_path / "c.wav")
    audio_sha, _ = render_canonical_wav(buf, out)
    with wave.open(out, "rb") as wf:
        fr, ch, sw, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
    assert (fr, ch, sw) == (24000, 1, 2)
    row = ReleaseRow(
        audio_id="ornix_vi_x", audio="audio/c.wav", language="vi", speaker_id="s",
        transcript="t", sample_rate=24000, channels=1, encoding="PCM_S16LE",
        duration_s=round(n / 24000, 6), source_id="SRC", source_sha256="a" * 64,
        audio_sha256=audio_sha, segment_start_sample_source=0,
        segment_end_sample_source=n, rights_record_id="R", quality_evidence_id="Q",
        quality_policy_version="v", quality_gate="ACCEPT", split="train",
        release_id="rel")
    row.validate(release_target="train_only")  # raises if any invariant is violated
