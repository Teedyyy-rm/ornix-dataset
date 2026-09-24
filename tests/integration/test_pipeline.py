"""End-to-end pipeline integration (spec §2, §8; idempotency + leakage-safe splits)."""

import os

import pytest

from ornix_dataset.audit.events import AuditLog
from ornix_dataset.config import (build_detectors, build_gate_and_adapters,
                                   load_policy_config, technical_thresholds)
from ornix_dataset.contracts.enums import DecisionState
from ornix_dataset.pipeline import OrnixPipeline, RunPaths
from ornix_dataset.util.jsonl import read_jsonl


def _pipeline(tmp_path, policy_path):
    policy = load_policy_config(policy_path)
    return OrnixPipeline(str(tmp_path / "work"), policy)


def _run(tmp_path, sources_config, policy_path, run_id="r1"):
    _gate, adapters, _ = build_gate_and_adapters(sources_config)
    pipe = _pipeline(tmp_path, policy_path)
    paths = RunPaths.create(pipe.workdir, run_id)
    audit = AuditLog(paths.audit)
    combined = pipe.run(adapters[0], paths, audit)
    return pipe, paths, combined


def test_pilot_end_to_end_accepts_clean_speech(tmp_path, sources_config):
    _pipe, paths, combined = _run(tmp_path, sources_config,
                                  "configs/quality_policy.pilot.yaml")
    # 4 clean speech clips accept, silent clip rejected as NO_SPEECH
    assert len(combined.accepted) == 4
    decisions = [e.decision for e in combined.evidences]
    # silent clip is stopped at the technical gate (fail-closed) before policy
    assert DecisionState.REJECT_TECH in decisions or DecisionState.REJECT in decisions
    assert all(r.quality_gate == "ACCEPT" for r in combined.accepted)
    assert all(r.sample_rate == 24000 and r.channels == 1 for r in combined.accepted)


def test_public_policy_fail_closed_zero_accept(tmp_path, sources_config):
    # strict public policy: TRAIN_ONLY sources are not redistributable -> no ACCEPT
    _pipe, _paths, combined = _run(tmp_path, sources_config,
                                   "configs/quality_policy.example.yaml")
    assert len(combined.accepted) == 0
    states = {e.decision for e in combined.evidences}
    assert DecisionState.ACCEPT not in states
    assert DecisionState.LICENSE_REVIEW in states


def test_ingest_idempotent(tmp_path, sources_config):
    _gate, adapters, _ = build_gate_and_adapters(sources_config)
    pipe = _pipeline(tmp_path, "configs/quality_policy.pilot.yaml")
    paths = RunPaths.create(pipe.workdir, "r1")
    audit = AuditLog(paths.audit)
    first = pipe.ingest_and_stage(adapters[0], paths, audit)
    n_manifest = len(list(read_jsonl(paths.source_manifest)))
    second = pipe.ingest_and_stage(adapters[0], paths, audit)
    assert len(first) == 5 and len(second) == 0
    assert len(list(read_jsonl(paths.source_manifest))) == n_manifest


def test_run_deterministic(tmp_path, sources_config):
    _p1, paths1, c1 = _run(tmp_path, sources_config,
                           "configs/quality_policy.pilot.yaml", run_id="a")
    _p2, paths2, c2 = _run(tmp_path, sources_config,
                           "configs/quality_policy.pilot.yaml", run_id="b")
    d1 = sorted((e.segment_id.split("_", 1)[0], e.decision.value) for e in c1.evidences)
    d2 = sorted((e.segment_id.split("_", 1)[0], e.decision.value) for e in c2.evidences)
    assert [x[1] for x in d1] == [x[1] for x in d2]
    assert len(c1.accepted) == len(c2.accepted)


def test_split_no_leakage(tmp_path, sources_config):
    _pipe, paths, combined = _run(tmp_path, sources_config,
                                  "configs/quality_policy.pilot.yaml")
    rows = read_jsonl(paths.accepted)
    # each speaker (group) must live in exactly one split
    by_group = {}
    for r in rows:
        by_group.setdefault(r["speaker_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_group.values())


def test_build_detectors_fail_closed():
    ds = build_detectors("configs/models.lock.example.yaml")
    avail = ds.availability
    # licensed models unavailable offline -> availability stays False (fail-closed)
    assert avail["music"] is False
    assert avail["quality"] is False
    assert avail["speaker"] is False
    # DSP noise + energy VAD are always constructible (heuristic, no weights)
    assert ds.noise is not None and ds.vad is not None


def test_technical_thresholds_from_profile():
    t = technical_thresholds("configs/audio_profile.yaml")
    assert t.max_duration_s > 0
