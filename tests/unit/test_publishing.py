"""Publishing hardening: exact remote set match + approval revision binding."""

import os

import pytest

from ornix_dataset.publishing.approval import ApprovalReceipt, validate_approval
from ornix_dataset.publishing.verification import remote_verify


class _FakeApi:
    def __init__(self, files):
        self._files = files

    def list_repo_files(self, repo_id, repo_type="dataset", revision=None):
        return list(self._files)


def _release(tmp_path):
    d = tmp_path / "rel"
    d.mkdir()
    (d / "release_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (d / "MANIFEST.sha256").write_text("x  release_manifest.jsonl\n", encoding="utf-8")
    return d


def test_remote_verify_flags_extra_remote_file(tmp_path, monkeypatch):
    d = _release(tmp_path)
    local = {"release_manifest.jsonl", "MANIFEST.sha256"}
    # remote has a stale shard we never built -> must be flagged (exact set, not subset)
    api = _FakeApi(local | {"audio/STALE.wav", ".gitattributes"})

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: str(d / a[1]) if len(a) > 1 else str(d / k["filename"]))

    res = remote_verify(api, "org/ds", "refs/pr/x", str(d), sample=0)
    assert not res.ok
    assert any("EXTRA_REMOTE_FILE:audio/STALE.wav" in e for e in res.errors)
    # .gitattributes is repo-internal and must be ignored
    assert not any(".gitattributes" in e for e in res.errors)


def test_remote_verify_flags_missing_remote_file(tmp_path):
    d = _release(tmp_path)
    api = _FakeApi({"release_manifest.jsonl"})  # MANIFEST.sha256 absent remotely
    res = remote_verify(api, "org/ds", "refs/pr/x", str(d), sample=0)
    assert not res.ok
    assert any("REMOTE_MISSING:MANIFEST.sha256" in e for e in res.errors)


def _receipt(digest, revision="refs/pr/ornix-staging"):
    return ApprovalReceipt(
        release_digest=digest, repo_id="org/ds", revision=revision,
        max_bytes=10 ** 12, operator_id="op", expires_utc="2999-01-01T00:00:00Z",
        policy_version="v1", license_ack=True)


def test_validate_approval_binds_revision(tmp_path):
    from ornix_dataset.publishing.approval import release_digest

    d = tmp_path / "rel"
    d.mkdir()
    (d / "MANIFEST.sha256").write_text("x  a\n", encoding="utf-8")
    (d / "RELEASE_READY.json").write_text("{}", encoding="utf-8")
    digest = release_digest(str(d))

    # approval authorizes refs/pr/ornix-staging; publishing to a different ref must block
    ok, reasons = validate_approval(_receipt(digest, revision="refs/pr/other"),
                                    str(d), "org/ds", revision="refs/pr/ornix-staging")
    assert not ok
    assert "REVISION_MISMATCH" in reasons

    ok2, reasons2 = validate_approval(_receipt(digest), str(d), "org/ds",
                                      revision="refs/pr/ornix-staging")
    assert ok2, reasons2
