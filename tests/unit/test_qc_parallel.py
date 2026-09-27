"""cli.qc_records parallel determinism (ORNIX_QC_WORKERS). Offline, DSP-only."""

import json
import os

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from ornix_dataset import cli
from ornix_dataset.audit.events import AuditLog
from ornix_dataset.contracts.enums import IngestStatus, RightsStatus
from ornix_dataset.contracts.source import SourceRecord
from ornix_dataset.curation.policy import PolicyConfig
from ornix_dataset.pipeline import OrnixPipeline, RunPaths
from ornix_dataset.util.jsonl import load_jsonl


def _wav(path, duration_s=4.0, sr=24000, freq=440.0):
    n = int(duration_s * sr)
    data = (0.1 * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype("float32")
    sf.write(path, data, sr)


def _pipe(workdir):
    policy = PolicyConfig(policy_version="test-parallel-v1",
                          required_checks=["rights_ok"])
    return OrnixPipeline(workdir, policy)


def _records(root, n=6):
    recs = []
    for i in range(n):
        p = os.path.join(root, f"s{i}.wav")
        _wav(p, freq=440.0 + i * 40)
        recs.append(SourceRecord(
            source_id=f"SRC_{i}", source_uri=f"file://{p}",
            source_revision="local", original_file_id=f"s{i}.wav",
            source_sha256=f"{i:064d}", source_bytes=os.path.getsize(p),
            source_language="vi", source_transcript="xin chao",
            rights_status=RightsStatus.REDISTRIBUTION_APPROVED,
            redistribution_permitted=True,
            ingest_status=IngestStatus.INGESTED, staged_path=p))
    return recs


def _run_once(tmp, workers):
    from ornix_dataset.cli import qc_records
    rundir = os.path.join(str(tmp), f"run-w{workers}")
    paths = RunPaths.create(rundir, "r")
    audit = AuditLog(paths.audit, run_id="r")
    pipe = _pipe(rundir)
    srcdir = os.path.join(str(tmp), "src")
    os.makedirs(srcdir, exist_ok=True)
    recs = _records(srcdir, n=6)
    # Stage copies (pipeline copies file:// into immutable staging).
    combined = qc_records(pipe, recs, paths, audit)
    ev = load_jsonl(paths.evidence)
    acc = load_jsonl(paths.accepted)
    return ev, acc


def test_qc_records_parallel_matches_sequential(tmp_path, monkeypatch):
    srcdir = os.path.join(str(tmp_path), "src")
    os.makedirs(srcdir, exist_ok=True)
    monkeypatch.setenv("ORNIX_QC_WORKERS", "1")
    _ = cli.qc_workers()  # import surface used
    ev1, acc1 = _run_once(tmp_path / "a", 1)
    monkeypatch.setenv("ORNIX_QC_WORKERS", "6")
    # Fresh source dir so hashes/paths stay comparable; compare decisions.
    ev2, acc2 = _run_once(tmp_path / "b", 6)
    assert len(ev1) == len(ev2) > 0
    d1 = [(e.get("decision"), tuple(e.get("reason_codes", []))) for e in ev1]
    d2 = [(e.get("decision"), tuple(e.get("reason_codes", []))) for e in ev2]
    assert d1 == d2
    assert len(acc1) == len(acc2)
