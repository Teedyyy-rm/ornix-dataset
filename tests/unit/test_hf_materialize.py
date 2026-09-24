"""HF materialization (spec Phase 1, §12): download -> verify content SHA -> stage."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fixtures import synth  # noqa: E402
from ornix_dataset.contracts.enums import IngestStatus  # noqa: E402
from ornix_dataset.contracts.source import SourceRecord  # noqa: E402
from ornix_dataset.ingestion.hf import HfSourceAdapter  # noqa: E402
from ornix_dataset.ingestion.permissions import PermissionGate  # noqa: E402
from ornix_dataset.util.hashing import sha256_file  # noqa: E402


def _adapter():
    gate = PermissionGate({})
    return HfSourceAdapter(repo_id="org/ds", gate=gate, revision="deadbeef")


def _record(sha, reasons=None):
    return SourceRecord(
        source_id="SRC_1", source_uri="hf://datasets/org/ds@deadbeef/a.wav",
        source_revision="deadbeef", original_file_id="a.wav",
        source_sha256=sha, source_bytes=0, reason_codes=list(reasons or []))


def test_materialize_verifies_content_sha_and_stages_immutable(tmp_path, monkeypatch):
    blob = tmp_path / "downloaded.wav"
    synth.write_wav(str(blob), synth.speechlike(24000, 1.0, seed=1), 24000)
    true_sha = sha256_file(str(blob))

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kw: str(blob))

    rec = _record(true_sha)  # remote blob sha known + matches
    dest = tmp_path / "staging"
    staged = _adapter().materialize(rec, str(dest))
    assert os.path.exists(staged)
    assert rec.staged_path == staged
    assert sha256_file(staged) == true_sha
    # immutable staging (read-only)
    assert (os.stat(staged).st_mode & 0o222) == 0


def test_materialize_rejects_content_sha_mismatch(tmp_path, monkeypatch):
    blob = tmp_path / "downloaded.wav"
    synth.write_wav(str(blob), synth.speechlike(24000, 1.0, seed=2), 24000)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kw: str(blob))

    rec = _record("f" * 64)  # pinned sha will NOT match downloaded bytes
    with pytest.raises(RuntimeError, match="content sha mismatch"):
        _adapter().materialize(rec, str(tmp_path / "staging"))


def test_materialize_binds_sha_when_remote_unverified(tmp_path, monkeypatch):
    blob = tmp_path / "downloaded.wav"
    synth.write_wav(str(blob), synth.speechlike(24000, 1.0, seed=3), 24000)
    true_sha = sha256_file(str(blob))

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kw: str(blob))

    # no remote blob sha was available -> provenance is bound to downloaded bytes
    rec = _record("0" * 64, reasons=["SOURCE_SHA_UNVERIFIED_FROM_REMOTE"])
    _adapter().materialize(rec, str(tmp_path / "staging"))
    assert rec.source_sha256 == true_sha


def test_analyze_source_emits_error_when_staged_path_missing(tmp_path):
    # a record that claims INGESTED but has no staged bytes must NOT be silently
    # dropped — the pipeline emits an ERROR evidence (no hidden skip, no-loss).
    from ornix_dataset.audit.events import AuditLog
    from ornix_dataset.curation.policy import PolicyConfig
    from ornix_dataset.pipeline import OrnixPipeline, RunPaths

    policy = PolicyConfig(policy_version="v1", required_checks=["rights_ok"])
    pipe = OrnixPipeline(str(tmp_path / "work"), policy)
    paths = RunPaths.create(pipe.workdir, "r1")
    audit = AuditLog(paths.audit)
    rec = SourceRecord(
        source_id="SRC_1", source_uri="hf://datasets/org/ds@x/a.wav",
        source_revision="x", original_file_id="a.wav", source_sha256="a" * 64,
        source_bytes=0, ingest_status=IngestStatus.INGESTED, staged_path=None)
    res = pipe.analyze_source(rec, paths, audit)
    assert res.evidences and res.evidences[0].decision.value == "ERROR"
    assert "STAGED_PATH_MISSING" in res.evidences[0].reason_codes
