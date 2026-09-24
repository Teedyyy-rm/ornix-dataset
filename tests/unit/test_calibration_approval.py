"""Calibration harness + approval-gate tests (spec §5.2, §7 T-013)."""

import pytest

from ornix_dataset.calibration.goldset import GoldClip, check_separation, stratify_summary
from ornix_dataset.calibration.metrics import (calibration_report, classification_metrics,
                                               false_accept_rate, false_reject_rate)


def _clip(cid, split, src, spk, fam, clean):
    return GoldClip(cid, f"/x/{cid}.wav", split, src, spk, fam, clean)


def test_separation_detects_leak():
    clips = [_clip("a", "calibration", "src1", "spk1", "fam1", True),
             _clip("b", "heldout", "src1", "spk2", "fam2", False)]
    sep = check_separation(clips)
    assert "src1" in sep["source"]


def test_separation_clean():
    clips = [_clip("a", "calibration", "src1", "spk1", "fam1", True),
             _clip("b", "heldout", "src2", "spk2", "fam2", False)]
    sep = check_separation(clips)
    assert sep["source"] == [] and sep["speaker"] == [] and sep["family"] == []


def test_metrics():
    m = classification_metrics([True, True, False, False], [True, False, False, False])
    assert m["tp"] == 1 and m["fn"] == 1 and m["tn"] == 2
    assert 0 <= m["f1"] <= 1


def test_false_accept_reject_rates():
    gold_noisy = [True, True, False, False]
    decisions = ["ACCEPT", "REJECT", "REJECT", "ACCEPT"]
    assert false_accept_rate(gold_noisy, decisions) == 0.5
    assert false_reject_rate(gold_noisy, decisions) == 0.5


def test_calibration_report_reports_review_bucket():
    rep = calibration_report([True, False], ["REVIEW", "ACCEPT"])
    assert "review_fraction" in rep["coverage"]


def test_stratify_summary():
    clips = [_clip("a", "calibration", "s", "p", "f", True)]
    clips[0].labels = ["HISS_STATIC"]
    s = stratify_summary(clips)
    assert s["n_clips"] == 1 and s["clean"] == 1


# --- approval gate (T-013) ---
from ornix_dataset.publishing.approval import (ApprovalReceipt, release_digest,
                                               validate_approval)


def _built_release(tmp_path):
    from ornix_dataset.exporters import build_release
    from fixtures import synth
    import os

    audio = tmp_path / "audio_src"
    audio.mkdir()
    synth.write_wav(str(audio / "seg.wav"), synth.speechlike(24000, 2.0), 24000)
    from ornix_dataset.util.hashing import sha256_file
    sha = sha256_file(str(audio / "seg.wav"))
    row = {"audio_id": "ornix_vi_x", "audio": "audio/seg.wav", "language": "vi",
           "speaker_id": "s", "transcript": "t", "sample_rate": 24000, "channels": 1,
           "encoding": "PCM_S16LE", "duration_s": 2.0, "source_id": "S",
           "source_sha256": "a" * 64, "audio_sha256": sha,
           "segment_start_sample_source": 0, "segment_end_sample_source": 1,
           "rights_record_id": "R", "quality_evidence_id": "E",
           "quality_policy_version": "v1", "quality_gate": "ACCEPT", "split": "train",
           "release_id": "rel-1"}
    out = str(tmp_path / "rel")
    art = build_release("rel-1", [row], str(audio), out)
    return art


def test_t013_approval_digest_mismatch_blocks(tmp_path):
    art = _built_release(tmp_path)
    assert art.ready
    receipt = ApprovalReceipt(release_digest="deadbeef" * 8, repo_id="org/ds",
                              revision="refs/pr/x", max_bytes=10 ** 9,
                              operator_id="op", expires_utc="2099-01-01T00:00:00Z",
                              policy_version="v1", license_ack=True)
    ok, reasons = validate_approval(receipt, art.release_dir, "org/ds")
    assert not ok and "RELEASE_DIGEST_MISMATCH" in reasons


def test_t013_expired_approval_blocks(tmp_path):
    art = _built_release(tmp_path)
    digest = release_digest(art.release_dir)
    receipt = ApprovalReceipt(release_digest=digest, repo_id="org/ds", revision="r",
                              max_bytes=10 ** 9, operator_id="op",
                              expires_utc="2000-01-01T00:00:00Z", policy_version="v1",
                              license_ack=True)
    ok, reasons = validate_approval(receipt, art.release_dir, "org/ds")
    assert not ok and "APPROVAL_EXPIRED" in reasons
