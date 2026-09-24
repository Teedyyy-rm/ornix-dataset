"""Calibration runner (spec §5.1-5.3): real pipeline over a labeled gold set."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fixtures import synth  # noqa: E402
from ornix_dataset.calibration.runner import run_calibration  # noqa: E402
from ornix_dataset.curation.policy import PolicyConfig  # noqa: E402
from ornix_dataset.pipeline import OrnixPipeline  # noqa: E402


def _make_clip(root, clip_id, split, family, is_clean, kind):
    sr = 24000
    if kind == "clean":
        sig = synth.speechlike(sr, 5.0, seed=abs(hash(clip_id)) % 1000)
    elif kind == "noise":
        sig = synth.speechlike(sr, 5.0, seed=2) * 0.3 + 0.7 * synth.white_noise(sr, 5.0)
    else:  # silence
        sig = synth.silence(sr, 3.0)
    path = os.path.join(root, f"{clip_id}.wav")
    synth.write_wav(path, sig, sr)
    return {"clip_id": clip_id, "path": path, "split": split,
            "source_id": f"src_{clip_id}", "speaker_id": f"spk_{family}",
            "recording_family": family, "is_clean": is_clean,
            "labels": [] if is_clean else ["HISS_STATIC"],
            "severity": "N0" if is_clean else "N3", "domain": "studio"}


def _goldset(tmp_path, rows):
    root = tmp_path / "gold"
    root.mkdir()
    written = [_make_clip(str(root), *r) for r in rows]
    jsonl = root / "goldset.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fh:
        for r in written:
            fh.write(json.dumps(r) + "\n")
    return str(jsonl)


def _pipeline(tmp_path):
    policy = PolicyConfig(policy_version="cal-v1",
                          required_checks=["technical_ok", "no_severe_noise"])
    return OrnixPipeline(str(tmp_path / "work"), policy)


def test_calibration_runs_and_reports_metrics(tmp_path):
    gs = _goldset(tmp_path, [
        ("cal_clean_1", "calibration", "A", True, "clean"),
        ("cal_noise_1", "calibration", "B", False, "noise"),
        ("cal_silence_1", "calibration", "B", False, "silence"),
        ("hel_clean_1", "heldout", "C", True, "clean"),
    ])
    pipe = _pipeline(tmp_path)
    res = run_calibration(gs, pipe, str(tmp_path / "work"), split="calibration")
    assert res.ok, res.blockers
    assert "false_accept_rate" in res.report
    assert "false_reject_rate" in res.report
    assert res.report["coverage"]["total"] == 3  # only the calibration split ran
    assert len(res.per_clip) == 3


def test_calibration_fails_closed_on_leakage(tmp_path):
    # same speaker/family in both splits => tuning-on-test => must refuse
    gs = _goldset(tmp_path, [
        ("cal_1", "calibration", "SHARED", True, "clean"),
        ("hel_1", "heldout", "SHARED", True, "clean"),
    ])
    pipe = _pipeline(tmp_path)
    res = run_calibration(gs, pipe, str(tmp_path / "work"))
    assert not res.ok
    assert any("GOLDSET_LEAKAGE" in b for b in res.blockers)
