"""Incremental push isolation tests (no network).

Fakes HfApi + hf_hub_download: proves a verified file is never re-uploaded
(ledger isolation), tree diffs push only new/changed bytes, and a failed
verification does not advance the ledger (retry pushes the same set).
"""

import json
import os
import wave

import pytest

from ornix_dataset.publishing.incremental import (
    load_ledger,
    push_tree,
)

huggingface_hub = pytest.importorskip("huggingface_hub")

ROW = {"audio": "audio/01/Ornix_0000001.wav",
       "text": "xin chao", "file_name": "audio/01/Ornix_0000001.wav",
       "speaker": "spk_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "duration": 1.0,
       "language": "vi"}


def _wav(path, n=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x02" * n)


def _tree(tmp_path, rows):
    ds = tmp_path / "tree"
    for r in rows:
        _wav(str(ds / "train" / r["audio"]))
    mp = ds / "train" / "metadata.jsonl"
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return str(ds)


class _Commit:
    def __init__(self, oid):
        self.oid = oid


class _FakeApi:
    def __init__(self, tree):
        self.tree = tree
        self.commits = []  # list of sorted rel lists, one per create_commit

    def repo_info(self, repo_id, repo_type=None):
        return {"id": repo_id}

    def create_repo(self, *a, **k):
        return {"id": a[0] if a else None}

    def create_commit(self, repo_id, operations, commit_message=None,
                      revision=None, repo_type=None):
        rels = sorted(op.path_in_repo for op in operations)
        self.commits.append(rels)
        return _Commit(f"c{len(self.commits)}")


def _fake_download_factory(tree, bad=()):
    def _dl(repo_id, filename, repo_type=None, revision=None):
        if filename in bad:
            p = os.path.join(tree, "..", "corrupt.bin")
            with open(p, "wb") as fh:
                fh.write(b"corrupt")
            return p
        return os.path.join(tree, filename)
    return _dl


@pytest.fixture
def hub(monkeypatch, tmp_path):
    api_holder = {}

    def _factory(token=None):
        return api_holder["api"]

    monkeypatch.setattr(huggingface_hub, "HfApi", _factory)
    return api_holder


def _run_push(tmp_path, monkeypatch, hub, rows, bad=(), state=None):
    tree = _tree(tmp_path, rows)
    hub["api"] = _FakeApi(tree)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        _fake_download_factory(tree, bad))
    monkeypatch.setenv("HF_TOKEN", "tok")
    st = state or str(tmp_path / "state")
    pol = tmp_path / "pol.yaml"
    pol.write_text("x: 1\n")
    rep = push_tree(tree, "org/ds", st, revision="main",
                    policy_path=str(pol))
    return tree, st, rep


def test_first_push_uploads_all_then_noop(tmp_path, monkeypatch, hub):
    tree, st, rep = _run_push(tmp_path, monkeypatch, hub, [ROW])
    assert rep["ok"] and rep["status"] == "PUBLISHED_VERIFIED"
    assert rep["pushed"] > 0 and rep["commit_sha"] == "c1"
    first = set(hub["api"].commits[0])
    assert any(f.endswith(".wav") for f in first)

    ledger = load_ledger(st)
    assert all((ledger.get(f) or {}).get("sha256") for f in first)

    # Second push with an unchanged tree uploads nothing.
    rep2 = push_tree(tree, "org/ds", st, revision="main",
                     policy_path=str(tmp_path / "pol.yaml"))
    assert rep2["status"] == "NO_PUSH_NEEDED" and rep2["pushed"] == 0
    assert len(hub["api"].commits) == 1


def test_diff_push_uploads_only_new_bytes(tmp_path, monkeypatch, hub):
    tree, st, rep = _run_push(tmp_path, monkeypatch, hub, [ROW])
    assert rep["ok"]
    row2 = dict(ROW,
                audio="audio/02/Ornix_0000002.wav",
                file_name="audio/02/Ornix_0000002.wav")
    _wav(os.path.join(tree, "train", row2["audio"]))
    with open(os.path.join(tree, "train", "metadata.jsonl"), "a",
              encoding="utf-8") as fh:
        fh.write(json.dumps(row2, sort_keys=True) + "\n")
    rep2 = push_tree(tree, "org/ds", st, revision="main",
                     policy_path=str(tmp_path / "pol.yaml"))
    assert rep2["ok"]
    pushed = set(rep2["pushed_files"])
    assert {"train/metadata.jsonl", "train/" + row2["audio"]} <= pushed
    # No previously pushed wav is re-uploaded.
    assert "train/" + ROW["audio"] not in pushed


def test_verify_failure_does_not_advance_ledger(tmp_path, monkeypatch, hub):
    tree, st, rep = _run_push(tmp_path, monkeypatch, hub, [ROW],
                              bad={"train/metadata.jsonl"})
    assert not rep["ok"] and rep["status"] == "REMOTE_VERIFY_FAILED"
    assert load_ledger(st) == {}
    # Retry with a healthy readback pushes the same set again (no dupes).
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        _fake_download_factory(tree))
    rep2 = push_tree(tree, "org/ds", st, revision="main",
                     policy_path=str(tmp_path / "pol.yaml"))
    assert rep2["ok"] and rep2["pushed"] > 0
    assert len(hub["api"].commits) == 2
    assert set(hub["api"].commits[0]) == set(hub["api"].commits[1])
