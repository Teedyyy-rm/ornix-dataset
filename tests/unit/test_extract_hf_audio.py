"""Regression tests for scripts/extract_hf_audio.py container recovery.

HF parquet audio commonly stores ``path: null``; the extractor must recover a
real audio extension from the magic bytes, or the loose files land without an
extension and LocalSourceAdapter silently skips them at ingest.
"""

import importlib.util
import os

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..",
                       "scripts", "extract_hf_audio.py")


def _load():
    spec = importlib.util.spec_from_file_location("extract_hf_audio", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sniff_ext_by_magic():
    m = _load()
    assert m._sniff_ext(b"RIFF\x00\x00\x00\x00WAVEfmt ") == ".wav"
    assert m._sniff_ext(b"fLaC\x00\x00") == ".flac"
    assert m._sniff_ext(b"OggS\x00\x02") == ".ogg"
    assert m._sniff_ext(b"ID3\x04\x00") == ".mp3"
    assert m._sniff_ext(b"\xff\xfb\x90\x00") == ".mp3"          # MPEG frame sync
    assert m._sniff_ext(b"\x00\x00\x00\x20ftypM4A ") == ".m4a"
    assert m._sniff_ext(b"garbage-bytes-here") == ".wav"        # fail-safe


def test_named_recovers_extension_when_path_missing():
    m = _load()
    wav = b"RIFF\x00\x00\x00\x00WAVE"
    # path is None (typical parquet audio) -> synth stem + sniffed extension
    assert m._named(None, wav, 7) == "clip_000007.wav"
    # path present but extension-less -> keep the stem, add the sniffed extension
    assert m._named("utt_123", wav, 0) == "utt_123.wav"
    # path already has a real audio extension -> keep it verbatim
    assert m._named("a/b/utt_5.mp3", b"\xff\xfb\x00", 0) == "utt_5.mp3"
