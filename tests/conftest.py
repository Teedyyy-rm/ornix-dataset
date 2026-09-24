"""Pytest fixtures: synthetic corpora + a tiny end-to-end helper."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))  # make `fixtures` importable

from fixtures import synth  # noqa: E402


@pytest.fixture
def clean_wav(tmp_path):
    p = tmp_path / "clean.wav"
    synth.write_wav(str(p), synth.speechlike(24000, 3.0, seed=1), 24000)
    return str(p)


@pytest.fixture
def pilot_corpus(tmp_path):
    """A small local corpus dir + JSONL metadata; returns (root, metadata_path)."""
    import json

    root = tmp_path / "pilot"
    root.mkdir()
    meta = []
    for spk in range(2):
        for u in range(2):
            fn = f"spk{spk}_utt{u}.wav"
            synth.write_wav(str(root / fn), synth.speechlike(24000, 4.0, seed=spk * 7 + u), 24000)
            meta.append({"file": fn, "source_transcript": f"cau noi {spk} {u}",
                         "source_speaker_ref": f"spk{spk}", "source_language": "vi"})
    synth.write_wav(str(root / "silent.wav"), synth.silence(24000, 2.0), 24000)
    meta.append({"file": "silent.wav", "source_speaker_ref": "spkS", "source_language": "vi"})
    mpath = root / "metadata.jsonl"
    with open(mpath, "w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    return str(root), str(mpath)


@pytest.fixture
def sources_config(tmp_path, pilot_corpus):
    root, mpath = pilot_corpus
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(
        "sources:\n"
        "  - name: local_pilot\n"
        "    type: local\n"
        f"    root: {root}\n"
        "    source_revision: test-rev\n"
        f"    metadata_file: {mpath}\n"
        "    rights:\n"
        "      rights_status: TRAIN_ONLY\n"
        "      redistribution_permitted: false\n"
        "      commercial_training_permitted: UNKNOWN\n",
        encoding="utf-8",
    )
    return str(cfg)
