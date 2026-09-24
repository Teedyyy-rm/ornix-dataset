"""MD-005 Gate 5 — PUBLISH_VERIFIED (FakeHub, all offline)."""

import hashlib
import json
import os

import pytest
import yaml

from ornix_dataset.campaign import (
    CampaignStore,
    list_releases,
    path_prefix_for,
    prepare_release,
    publish_release,
    release_id_for,
    staging_revision_for,
)
from ornix_dataset.campaign.processing import qc_paths
from ornix_dataset.publishing.hf import marker_path_for
from ornix_dataset.util.jsonl import load_jsonl

SHA = "f" * 40
GB = 1024**3


class FakeHub:
    """In-memory Hub: remote path -> bytes, with scripted upload failures."""

    def __init__(self):
        self.remote = {}
        self.upload_failures = []
        self.upload_calls = 0
        self.branches = []

    # -- publisher surface --
    def repo_info(self, repo_id, repo_type="dataset"):
        return {"id": repo_id}

    def create_branch(self, repo_id, branch, repo_type="dataset", exist_ok=True):
        self.branches.append(branch)

    def upload_folder(self, folder_path, repo_id, repo_type="dataset",
                      revision=None, path_in_repo=None, commit_message=""):
        self.upload_calls += 1
        if self.upload_failures:
            raise self.upload_failures.pop(0)
        for dp, _ds, fs in os.walk(folder_path):
            for f in fs:
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, folder_path)
                key = f"{path_in_repo}/{rel}" if path_in_repo else rel
                with open(full, "rb") as fh:
                    self.remote[key] = fh.read()

        class _Commit:
            oid = "c" * 40
        return _Commit()

    def list_repo_files(self, repo_id, repo_type="dataset", revision=None):
        return list(self.remote)


def fake_download_factory(hub):
    def _dl(repo_id, filename, repo_type="dataset", revision=None):
        p = f"/tmp/fakehub-{abs(hash(filename)) % 999999}.bin"
        with open(p, "wb") as fh:
            fh.write(hub.remote[filename])
        return p
    return _dl


def wire_hub(monkeypatch, hub):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        fake_download_factory(hub))
    monkeypatch.setenv("HF_TOKEN", "dummy")


def accepted_row(audio_id, audio_bytes, release_id="R"):
    sha = hashlib.sha256(audio_bytes).hexdigest()
    return {"audio_id": audio_id, "audio": audio_id, "language": "vi",
            "speaker_id": "spk-0", "transcript": "x",
            "sample_rate": 24000, "channels": 1, "encoding": "PCM_S16LE",
            "duration_s": 3.0, "source_id": "SRC_" + sha[:16],
            "source_sha256": sha, "audio_sha256": sha,
            "segment_start_sample_source": 0,
            "segment_end_sample_source": 72000,
            "rights_record_id": "rr", "quality_evidence_id": "qe",
            "quality_policy_version": "v1", "quality_gate": "ACCEPT",
            "split": "train", "release_id": release_id}


def ready_batch(tmp_path, name="a.wav", n_extra_shards=0):
    """White-box RELEASE_READY batch: accepted rows + canonical audio + gate."""
    from ornix_dataset.campaign import Budget

    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pub", [{"repo": f"org/A@{SHA}"}],
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024)
    (b,) = store.plan_job_batches(jid, [(name, 100)], budget=budget)
    batch = store.load_batch(b.batch_id)
    paths = qc_paths(store, jid, b.batch_id)
    audio = os.urandom(48000)
    with open(os.path.join(paths.canonical_dir, name), "wb") as fh:
        fh.write(audio)
    row = accepted_row(name, audio)
    with open(os.path.join(paths.root, "BATCH_ACCEPTED.jsonl"), "w") as fh:
        fh.write(json.dumps(row) + "\n")
    gate = {"assignment": {name: "train"}, "duplicates": []}
    os.makedirs(os.path.join(store.root, "releases"), exist_ok=True)
    with open(os.path.join(store.root, "releases", f"{jid}.GATE.json"),
              "w") as fh:
        fh.write(json.dumps(gate))
    batch.status = "RELEASE_READY"
    store.save_batch(batch)
    return store, jid, batch


def approval_for(release_dir, revision, repo_id="org/dest"):
    from ornix_dataset.publishing.approval import release_digest

    return {"release_digest": release_digest(release_dir),
            "repo_id": repo_id, "revision": revision,
            "max_bytes": 10**12, "operator_id": "op",
            "expires_utc": "2999-01-01T00:00:00Z",
            "policy_version": "v1", "license_ack": True}


def test_prepare_refuses_unready_batch(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pub", [{"repo": f"org/A@{SHA}"}],
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    (b,) = store.plan_job_batches(jid, [("a.wav", 1)])
    rep = prepare_release(store, jid, b.batch_id)
    assert rep["ok"] is False and "PLANNED" in rep["reason"]


def test_prepare_builds_namespaced_ready_release(tmp_path):
    store, jid, batch = ready_batch(tmp_path)
    rep = prepare_release(store, jid, batch.batch_id, export_format="none")
    assert rep["ok"] is True and rep["ready"] is True, rep
    assert rep["release_id"].startswith("pub--")
    assert rep["revision"].startswith("ornix/")
    assert rep["path_prefix"].startswith("releases/")
    assert os.path.exists(os.path.join(rep["release_dir"], "RELEASE_READY.json"))
    assert os.path.exists(os.path.join(rep["release_dir"], "audio", "a.wav"))
    assert len(list_releases(store)) == 1


def test_publish_success_marker_outside_and_receipt(tmp_path, monkeypatch):
    hub = FakeHub()
    wire_hub(monkeypatch, hub)
    store, jid, batch = ready_batch(tmp_path)
    rep = prepare_release(store, jid, batch.batch_id, export_format="none")
    ap = tmp_path / "appr.yaml"
    ap.write_text(yaml.safe_dump(
        approval_for(rep["release_dir"], rep["revision"])), encoding="utf-8")
    pub = publish_release(store, rep["release_id"], "org/dest", str(ap),
                          full_hash=True)
    assert pub["ok"] is True and pub["status"] == "PUBLISHED_VERIFIED"
    assert pub["report"]["upload_attempts"] == 1
    # marker BESIDE the release dir, never inside it
    assert not os.path.exists(
        os.path.join(rep["release_dir"], "PUBLISHED_VERIFIED.json"))
    assert os.path.exists(marker_path_for(rep["release_dir"]))
    assert marker_path_for(rep["release_dir"]).endswith(
        ".PUBLISHED_VERIFIED.json")
    # namespaced remote layout
    assert any(k.startswith(rep["path_prefix"] + "/") for k in hub.remote)
    # receipt: full-hash verified, exact commit
    rec = json.load(open(pub["receipt"], encoding="utf-8"))
    assert rec["full_hash_verified"] is True
    assert rec["remote_commit_sha"] == "c" * 40
    assert rec["release_digest"] == rep["digest"]


def test_interrupted_upload_retries_and_continues(tmp_path, monkeypatch):
    hub = FakeHub()
    hub.upload_failures = [RuntimeError("conn reset")]
    wire_hub(monkeypatch, hub)
    store, jid, batch = ready_batch(tmp_path)
    rep = prepare_release(store, jid, batch.batch_id, export_format="none")
    ap = tmp_path / "appr.yaml"
    ap.write_text(yaml.safe_dump(
        approval_for(rep["release_dir"], rep["revision"])), encoding="utf-8")
    pub = publish_release(store, rep["release_id"], "org/dest", str(ap))
    assert pub["ok"] is True
    assert pub["report"]["upload_attempts"] == 2
    assert hub.upload_calls == 2


def test_hard_upload_failure_is_upload_failed_not_verify(tmp_path, monkeypatch):
    hub = FakeHub()
    hub.upload_failures = [RuntimeError("down")] * 5
    wire_hub(monkeypatch, hub)
    store, jid, batch = ready_batch(tmp_path)
    rep = prepare_release(store, jid, batch.batch_id, export_format="none")
    ap = tmp_path / "appr.yaml"
    ap.write_text(yaml.safe_dump(
        approval_for(rep["release_dir"], rep["revision"])), encoding="utf-8")
    pub = publish_release(store, rep["release_id"], "org/dest", str(ap))
    assert pub["ok"] is False and pub["status"] == "UPLOAD_FAILED"
    assert "upload failed after 3 attempt(s)" in pub["reasons"][0]
    assert not os.path.exists(marker_path_for(rep["release_dir"]))
    assert not os.path.exists(os.path.join(
        store.root, "releases", f"{rep['release_id']}.REMOTE_VERIFIED.json"))


def test_partial_upload_fails_verify_with_missing(tmp_path, monkeypatch):
    class _DropOne(FakeHub):
        def upload_folder(self, folder_path, repo_id, repo_type="dataset",
                          revision=None, path_in_repo=None, commit_message=""):
            super().upload_folder(folder_path, repo_id, repo_type,
                                  revision, path_in_repo, commit_message)
            # simulate a partial commit: one file never landed
            victim = next(k for k in self.remote if k.endswith(".jsonl"))
            del self.remote[victim]

    hub = _DropOne()
    wire_hub(monkeypatch, hub)
    store, jid, batch = ready_batch(tmp_path)
    rep = prepare_release(store, jid, batch.batch_id, export_format="none")
    ap = tmp_path / "appr.yaml"
    ap.write_text(yaml.safe_dump(
        approval_for(rep["release_dir"], rep["revision"])), encoding="utf-8")
    pub = publish_release(store, rep["release_id"], "org/dest", str(ap))
    assert pub["ok"] is False and pub["status"] == "REMOTE_VERIFY_FAILED"
    assert any("REMOTE_MISSING" in e for e in pub["reasons"])


def test_sibling_releases_coexist_in_one_repo(tmp_path, monkeypatch):
    hub = FakeHub()
    wire_hub(monkeypatch, hub)
    store, jid, batch = ready_batch(tmp_path, name="a.wav")
    r1 = prepare_release(store, jid, batch.batch_id, export_format="none")
    # second batch in the same job
    from ornix_dataset.campaign import Budget
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024)
    (b2,) = store.plan_job_batches(jid, [("b.wav", 100)], budget=budget)
    bb = store.load_batch(b2.batch_id)
    paths = qc_paths(store, jid, bb.batch_id)
    audio = os.urandom(48000)
    with open(os.path.join(paths.canonical_dir, "b.wav"), "wb") as fh:
        fh.write(audio)
    with open(os.path.join(paths.root, "BATCH_ACCEPTED.jsonl"), "w") as fh:
        fh.write(json.dumps(accepted_row("b.wav", audio)) + "\n")
    bb.status = "RELEASE_READY"
    store.save_batch(bb)
    gate = json.load(open(os.path.join(store.root, "releases", f"{jid}.GATE.json")))
    gate["assignment"]["b.wav"] = "validation"
    json.dump(gate, open(os.path.join(store.root, "releases", f"{jid}.GATE.json"), "w"))
    r2 = prepare_release(store, jid, bb.batch_id, export_format="none")
    assert r1["release_id"] != r2["release_id"]
    assert r1["revision"] != r2["revision"]
    assert r1["path_prefix"] != r2["path_prefix"]
    for rep in (r1, r2):
        ap = tmp_path / f"appr-{rep['release_id'][-4:]}.yaml"
        ap.write_text(yaml.safe_dump(
            approval_for(rep["release_dir"], rep["revision"])), encoding="utf-8")
        pub = publish_release(store, rep["release_id"], "org/dest", str(ap))
        assert pub["ok"] is True, pub
    # both namespaces present, neither verify flagged the sibling as EXTRA
    assert any(k.startswith(r1["path_prefix"] + "/") for k in hub.remote)
    assert any(k.startswith(r2["path_prefix"] + "/") for k in hub.remote)


def test_naming_helpers():
    rid = release_id_for("camp", "org--A-abc", "b-123")
    assert rid == "camp--org--A-abc--b-123"
    assert staging_revision_for(rid) == f"ornix/{rid}"
    assert path_prefix_for(rid) == f"releases/{rid}"
    with pytest.raises(ValueError):
        release_id_for("camp", "org/A", "b-1")


def test_cli_release_publish_fail_closed(tmp_path, capsys):
    from ornix_dataset.cli import main

    root = str(tmp_path / "store")
    assert main(["campaign", "release", "--root", root,
                 "--job", "j", "--batch", "b"]) == 2
    assert main(["campaign", "publish", "--root", root, "--release", "r",
                 "--repo", "org/d", "--approval", "a.yaml"]) == 2
