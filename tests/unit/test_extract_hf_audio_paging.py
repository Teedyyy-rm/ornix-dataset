"""Pure-function tests for the enhanced extract_hf_audio helpers (no network)."""

import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "extract_hf_audio",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "extract_hf_audio.py"))
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_meta_row_omits_empty_speaker():
    r = mod._meta_row("a.wav", "xin chao", "vi", None)
    assert r == {"file": "a.wav", "source_transcript": "xin chao", "source_language": "vi"}


def test_meta_row_includes_speaker_ref():
    r = mod._meta_row("a.wav", "t", "vi", "thanh_pham")
    assert r["source_speaker_ref"] == "thanh_pham"


def test_speaker_ref_wins_over_column():
    args = _NS(speaker_ref="fixed_id", speaker_col="speaker")
    assert mod._speaker_for_row(args, "row_val") == "fixed_id"


def test_speaker_column_used_when_no_fixed_ref():
    args = _NS(speaker_ref=None, speaker_col="speaker")
    assert mod._speaker_for_row(args, "alice") == "speaker:alice"


def test_speaker_none_when_neither_set():
    args = _NS(speaker_ref=None, speaker_col=None)
    assert mod._speaker_for_row(args, "ignored") is None


def test_uniq_name_flattens_subdirs():
    assert mod._uniq_name("data/000/audio_0001.wav", b"RIFF....", 0) == "data__000__audio_0001.wav"


def test_uniq_name_recovers_extension_from_magic():
    n = mod._uniq_name("data/clip", b"fLaC0000", 3)
    assert n == "data__clip.flac"
