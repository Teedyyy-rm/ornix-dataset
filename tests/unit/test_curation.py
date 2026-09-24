"""Curation tests: segmentation (T-003), dedup (T-011), split, transcript."""

import numpy as np

from ornix_dataset.curation.dedup import (exact_duplicates, fingerprint,
                                          near_duplicate_groups)
from ornix_dataset.curation.segmentation import (low_energy_points, plan_segments)
from ornix_dataset.curation.split import assign_splits, split_leakage
from ornix_dataset.curation.transcript import char_error_rate, transcript_match_status
from ornix_dataset.contracts.enums import MeasurementStatus
from fixtures import synth


def test_t003_rescue_middle_segment():
    # music 0-2, speech 2-6, music 6-8 -> keep the clean middle
    plan = plan_segments([[2, 6]], [[0, 2], [6, 8]], max_duration_s=12)
    assert len(plan.intervals_s) == 1
    s, e = plan.intervals_s[0]
    assert 1.9 <= s <= 2.1 and 5.9 <= e <= 6.1


def test_long_speech_split_under_max():
    plan = plan_segments([[0, 30]], [], max_duration_s=12)
    assert all((e - s) <= 12.001 for s, e in plan.intervals_s)


def test_low_energy_points_finds_silence_troughs():
    sr = 24000
    # speech - silence - speech: a clear low-energy trough in the middle
    sig = np.concatenate([synth.speechlike(sr, 4.0, seed=3),
                          synth.silence(sr, 1.0),
                          synth.speechlike(sr, 4.0, seed=4)])
    pts = low_energy_points(sig, sr)
    assert pts, "expected at least one low-energy trough"
    # a trough should land inside the silent gap around t=4..5s
    assert any(4.0 <= p <= 5.0 for p in pts)


def test_plan_segments_snaps_long_span_to_silence():
    # a >max span with a silence trough near the arithmetic midpoint -> snap to it
    plan = plan_segments([[0, 20]], [], max_duration_s=12, silence_points=[10.0],
                         snap_tol_s=1.0)
    cuts = [round(e, 2) for s, e in plan.intervals_s]
    assert 10.0 in cuts  # cut landed on the silence trough, not the arithmetic 10.05
    assert 0 not in plan.uncertain_indices  # snapped cut is not uncertain


def test_plan_segments_flags_uncertain_when_no_silence():
    plan = plan_segments([[0, 20]], [], max_duration_s=12, silence_points=[],
                         snap_tol_s=1.0)
    # forced to cut without a silence trough -> first clip flagged boundary-uncertain
    assert plan.uncertain_indices, "arithmetic cut must be marked uncertain"
    assert all((e - s) <= 12.001 for s, e in plan.intervals_s)


def test_exact_duplicates():
    groups = exact_duplicates({"a": "x", "b": "x", "c": "y"})
    assert groups == [["a", "b"]]


def test_t011_near_duplicate_detection():
    sig = synth.speechlike(24000, 3.0, seed=5)
    fp1 = fingerprint(sig, 24000)
    fp2 = fingerprint(sig * 0.9 + 0.001 * synth.white_noise(24000, 3.0), 24000)
    fp3 = fingerprint(synth.musiclike(24000, 3.0), 24000)
    groups = near_duplicate_groups({"a": fp1, "b": fp2, "c": fp3}, threshold=0.98)
    assert any("a" in g and "b" in g for g in groups)
    assert not any("c" in g for g in groups)


def test_lsh_dedup_matches_bruteforce_at_scale():
    # many distinct clips + a few exact-fingerprint twins; LSH must recover the twins
    fps = {}
    for i in range(120):
        fps[f"u{i}"] = fingerprint(synth.speechlike(24000, 2.0, seed=i), 24000)
    # inject 3 near-duplicate pairs (identical fingerprints)
    for a, b in [("u10", "d10"), ("u40", "d40"), ("u90", "d90")]:
        fps[b] = fps[a].copy()
    groups = near_duplicate_groups(fps, threshold=0.999, n_planes=12, n_tables=10)
    for a, b in [("u10", "d10"), ("u40", "d40"), ("u90", "d90")]:
        assert any(a in g and b in g for g in groups), f"{a}/{b} not grouped"


def test_split_no_leakage():
    gk = {f"item{i}": f"spk{i % 6}" for i in range(60)}
    asg = assign_splits(gk)
    assert split_leakage(gk, asg) == []
    assert set(asg.values()) <= {"train", "validation", "test"}


def test_split_deterministic():
    gk = {f"i{i}": f"g{i % 4}" for i in range(20)}
    assert assign_splits(gk, seed=42) == assign_splits(gk, seed=42)


def test_transcript_status():
    assert transcript_match_status(None) == MeasurementStatus.NOT_APPLICABLE
    assert transcript_match_status("xin chao", verified=True) == MeasurementStatus.OK
    assert transcript_match_status("xin chao") == MeasurementStatus.UNKNOWN
    assert transcript_match_status("xin chao toi la ornix",
                                   "hoan toan khac biet noi dung") == MeasurementStatus.ERROR


def test_cer():
    assert char_error_rate("abc", "abc") == 0.0
    assert char_error_rate("abcd", "abce") == 0.25
