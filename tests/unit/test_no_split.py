"""ORNIX_NO_SPLIT=1: splittable sources are discarded whole, never salvaged."""

import os

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

import ornix_dataset.pipeline as pipeline_mod
from ornix_dataset.audit.events import AuditLog
from ornix_dataset.contracts.enums import DecisionState, IngestStatus, RightsStatus
from ornix_dataset.contracts.source import SourceRecord
from ornix_dataset.curation.policy import PolicyConfig
from ornix_dataset.curation.segmentation import SegmentPlan
from ornix_dataset.pipeline import OrnixPipeline, RunPaths, no_split_enabled


def _wav(path, duration_s=8.0, sr=24000):
    n = int(duration_s * sr)
    data = (0.1 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype("float32")
    sf.write(path, data, sr)


def _pipe(workdir):
    policy = PolicyConfig(policy_version="test-nosplit-v1",
                          required_checks=["rights_ok"])
    return OrnixPipeline(workdir, policy)


def _src(path):
    return SourceRecord(
        source_id="SRC_nosplit", source_uri=f"file://{path}",
        source_revision="local", original_file_id=os.path.basename(path),
        source_sha256="7" * 64, source_bytes=os.path.getsize(path),
        source_language="vi", source_transcript="xin chao",
        rights_status=RightsStatus.REDISTRIBUTION_APPROVED,
        redistribution_permitted=True,
        ingest_status=IngestStatus.INGESTED, staged_path=path)


def _two_interval_plan(*a, **k):
    return SegmentPlan(intervals_s=[[0.0, 4.0], [4.0, 8.0]], reason="test")


def test_no_split_flag_parsing(monkeypatch):
    monkeypatch.delenv("ORNIX_NO_SPLIT", raising=False)
    assert no_split_enabled() is False
    monkeypatch.setenv("ORNIX_NO_SPLIT", "1")
    assert no_split_enabled() is True
    monkeypatch.setenv("ORNIX_NO_SPLIT", "0")
    assert no_split_enabled() is False


def test_default_splits_into_two_accepts(tmp_path, monkeypatch):
    monkeypatch.delenv("ORNIX_NO_SPLIT", raising=False)
    monkeypatch.setattr(pipeline_mod, "plan_segments", _two_interval_plan)
    p = os.path.join(str(tmp_path), "s.wav")
    _wav(p)
    paths = RunPaths.create(str(tmp_path), "r")
    audit = AuditLog(paths.audit, run_id="r")
    res = _pipe(str(tmp_path)).analyze_source(_src(p), paths, audit)
    assert [e.decision for e in res.evidences] == \
        [DecisionState.ACCEPT, DecisionState.ACCEPT]
    assert len(res.accepted) == 2


def test_no_split_discards_whole_source(tmp_path, monkeypatch):
    monkeypatch.setenv("ORNIX_NO_SPLIT", "1")
    monkeypatch.setattr(pipeline_mod, "plan_segments", _two_interval_plan)
    p = os.path.join(str(tmp_path), "s.wav")
    _wav(p)
    paths = RunPaths.create(str(tmp_path), "r")
    audit = AuditLog(paths.audit, run_id="r")
    res = _pipe(str(tmp_path)).analyze_source(_src(p), paths, audit)
    assert len(res.evidences) == 1
    assert res.evidences[0].decision == DecisionState.REJECT
    assert "REQUIRES_SEGMENTATION" in res.evidences[0].reason_codes
    assert res.accepted == []
