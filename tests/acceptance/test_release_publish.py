"""Acceptance tests: release verify (T-015), publish gates (T-013/T-014),
reproducibility (T-016). Publishing stays fully offline / dry-run."""

import os

import pytest

from ornix_dataset.audit.events import AuditLog
from ornix_dataset.config import build_gate_and_adapters, load_policy_config
from ornix_dataset.exporters import build_release, verify_release
from ornix_dataset.pipeline import OrnixPipeline, RunPaths
from ornix_dataset.publishing.approval import (ApprovalReceipt, release_digest,
                                               validate_approval)
from ornix_dataset.publishing.hf import PublishStatus, StagedPublisher
from ornix_dataset.util.jsonl import read_jsonl


def _run_pipeline(tmp_path, sources_config, run_id="acc"):
    policy = load_policy_config("configs/quality_policy.pilot.yaml")
    _gate, adapters, _ = build_gate_and_adapters(sources_config)
    pipe = OrnixPipeline(str(tmp_path / "work"), policy)
    paths = RunPaths.create(pipe.workdir, run_id)
    audit = AuditLog(paths.audit)
    combined = pipe.run(adapters[0], paths, audit)
    return paths, combined


def _build(tmp_path, sources_config, out_name="rel"):
    paths, combined = _run_pipeline(tmp_path, sources_config)
    rows = list(read_jsonl(paths.accepted))
    assert rows, "pilot run must accept rows to build a release"
    for r in rows:
        r["release_id"] = "ornix-rel-1"
    out = str(tmp_path / out_name)
    art = build_release("ornix-rel-1", rows, paths.canonical_dir, out)
    return art


def test_t015_release_verify_ok(tmp_path, sources_config):
    art = _build(tmp_path, sources_config)
    assert art.ready, art.blockers
    res = verify_release(art.release_dir)
    assert res.ok, res.errors
    assert res.n_rows == 4
    assert res.total_duration_s > 0


def test_t015_verify_detects_tamper(tmp_path, sources_config):
    art = _build(tmp_path, sources_config)
    # corrupt one audio file -> sha + manifest checks must fail
    audio_dir = os.path.join(art.release_dir, "audio")
    victim = os.path.join(audio_dir, sorted(os.listdir(audio_dir))[0])
    with open(victim, "ab") as fh:
        fh.write(b"\x00\x01\x02\x03")
    res = verify_release(art.release_dir)
    assert not res.ok
    assert any("SHA_MISMATCH" in e or "MANIFEST_SHA_MISMATCH" in e for e in res.errors)


def test_t014_publish_dry_run_no_network(tmp_path, sources_config):
    art = _build(tmp_path, sources_config)
    pub = StagedPublisher()
    res = pub.publish(art.release_dir, "org/ornix-vi", dry_run=True)
    assert res.status == PublishStatus.DRY_RUN
    assert res.plan["n_files"] > 0
    assert res.remote_commit_sha is None
    # no marker written on dry-run
    assert not os.path.exists(os.path.join(art.release_dir, "PUBLISHED_VERIFIED.json"))


def test_t013_publish_blocked_without_valid_approval(tmp_path, sources_config, monkeypatch):
    art = _build(tmp_path, sources_config)
    # write an approval file bound to the WRONG digest
    import yaml
    bad = {"release_digest": "0" * 64, "repo_id": "org/ornix-vi",
           "revision": "refs/pr/ornix-staging", "max_bytes": 10 ** 12,
           "operator_id": "op", "expires_utc": "2099-01-01T00:00:00Z",
           "policy_version": "ornix-qc-pilot-v1", "license_ack": True}
    apath = tmp_path / "approval.yaml"
    apath.write_text(yaml.safe_dump(bad), encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_should_not_be_used")
    pub = StagedPublisher()
    res = pub.publish(art.release_dir, "org/ornix-vi", dry_run=False, approval_path=str(apath))
    assert res.status == PublishStatus.BLOCKED
    assert "RELEASE_DIGEST_MISMATCH" in res.reasons
    assert not os.path.exists(os.path.join(art.release_dir, "PUBLISHED_VERIFIED.json"))


def test_t013_valid_approval_but_no_token_blocks(tmp_path, sources_config, monkeypatch):
    art = _build(tmp_path, sources_config)
    import yaml
    good = {"release_digest": release_digest(art.release_dir), "repo_id": "org/ornix-vi",
            "revision": "refs/pr/ornix-staging", "max_bytes": 10 ** 12,
            "operator_id": "op", "expires_utc": "2099-01-01T00:00:00Z",
            "policy_version": "ornix-qc-pilot-v1", "license_ack": True}
    apath = tmp_path / "approval.yaml"
    apath.write_text(yaml.safe_dump(good), encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pub = StagedPublisher()
    res = pub.publish(art.release_dir, "org/ornix-vi", dry_run=False, approval_path=str(apath))
    assert res.status == PublishStatus.BLOCKED
    assert any("no token" in r for r in res.reasons)


def test_t016_reproducible_release_digest(tmp_path, sources_config):
    art1 = _build(tmp_path, sources_config, out_name="rel_a")
    art2 = _build(tmp_path, sources_config, out_name="rel_b")
    # same accepted content -> identical per-file audio hashes
    m1 = {r["audio_id"]: r["audio_sha256"] for r in read_jsonl(
        os.path.join(art1.release_dir, "release_manifest.jsonl"))}
    m2 = {r["audio_id"]: r["audio_sha256"] for r in read_jsonl(
        os.path.join(art2.release_dir, "release_manifest.jsonl"))}
    assert m1 == m2
