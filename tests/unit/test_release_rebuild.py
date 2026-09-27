"""Release rebuild must stay verifiable (MANIFEST must describe real bytes).

Regression: build_release hashed the checksum manifest BEFORE writing
RELEASE_READY.json. On a fresh dir that works, but on a REBUILD the previous
run's marker was still on disk, got hashed, and was then overwritten with a
new `built_utc` — leaving MANIFEST.sha256 describing bytes that no longer
existed, so the rebuilt release failed its own verify_release.
"""

import json
import os

import numpy as np
import pytest

from ornix_dataset.exporters import build_release, verify_release
from ornix_dataset.testing.fakes import accepted_row

sf = pytest.importorskip("soundfile")


def _wav_bytes(duration_s=3.0, sr=24000):
    """A real 24 kHz mono PCM16 WAV so verify_release can read it back."""
    n = int(duration_s * sr)
    data = (0.05 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype("float32")
    p = "/tmp/_rebuild_test.wav"
    sf.write(p, data, sr, subtype="PCM_16")
    with open(p, "rb") as fh:
        return fh.read()


def _row(audio_id, blob, release_id="R"):
    r = accepted_row(audio_id, blob, release_id=release_id)
    return r


def _write_audio(canon_dir, audio_id, blob):
    os.makedirs(canon_dir, exist_ok=True)
    p = os.path.join(canon_dir, audio_id)
    with open(p, "wb") as fh:
        fh.write(blob)
    return p


def test_rebuild_release_still_verifies(tmp_path, monkeypatch):
    canon = tmp_path / "canon"
    out = tmp_path / "rel"
    blob = _wav_bytes()
    row = _row("a.wav", blob)
    _write_audio(str(canon), "a.wav", blob)

    # Force a different `built_utc` per build: without this, three rebuilds
    # inside the same clock second would produce byte-identical markers and
    # hide the very corruption being tested.
    counter = {"n": 0}
    import ornix_dataset.exporters.manifest as m
    monkeypatch.setattr(m, "utc_now_iso",
                        lambda: "2020-01-01T00:00:{:02d}Z".format(
                            counter["n"]))

    for attempt in (1, 2, 3):
        counter["n"] = attempt
        art = build_release("R1", [row], str(canon), str(out),
                            export_format="none", release_target="train_only")
        assert art.ready, art.blockers
        res = verify_release(str(out), offline=True)
        assert res.ok, f"attempt {attempt}: {res.errors}"


def test_manifest_covers_ready_marker(tmp_path):
    canon = tmp_path / "canon"
    out = tmp_path / "rel"
    blob = _wav_bytes()
    row = _row("a.wav", blob)
    _write_audio(str(canon), "a.wav", blob)
    build_release("R1", [row], str(canon), str(out),
                  export_format="none", release_target="train_only")
    man = (out / "MANIFEST.sha256").read_text(encoding="utf-8")
    assert "RELEASE_READY.json" in man, man
    # and the recorded hash equals the file on disk
    from ornix_dataset.util.hashing import sha256_file
    rec = {ln.split("  ", 1)[1]: ln.split("  ", 1)[0]
           for ln in man.strip().split("\n") if ln.strip()}
    assert rec["RELEASE_READY.json"] == sha256_file(str(out / "RELEASE_READY.json"))


def test_blocked_release_has_no_stale_ready_marker(tmp_path):
    canon = tmp_path / "canon"
    out = tmp_path / "rel"
    blob = _wav_bytes()
    row = _row("a.wav", blob)
    row["rights_status"] = "TRAIN_ONLY"
    row["redistribution_permitted"] = False
    _write_audio(str(canon), "a.wav", blob)
    # a previous successful build left a READY marker behind
    build_release("R1", [_row("a.wav", blob)], str(canon), str(out),
                  export_format="none", release_target="train_only")
    assert (out / "RELEASE_READY.json").exists()
    art = build_release("R1", [row], str(canon), str(out),
                        export_format="none", release_target="public")
    assert art.ready is False
    assert not (out / "RELEASE_READY.json").exists(), \
        "stale READY marker must not survive a blocked rebuild"
    assert (out / "RELEASE_BLOCKED.json").exists()
