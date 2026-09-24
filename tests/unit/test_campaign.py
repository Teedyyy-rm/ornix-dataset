"""MD-001 Gate 1 — CAMPAIGN_VERIFIED (all offline, fake resolver)."""

import json
import os

import pytest

from ornix_dataset.campaign import (
    CampaignStore,
    batch_id_for,
    normalize_repo_ref,
    plan_batches,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


def fake_resolver(mapping):
    def _resolve(repo_id, revision):
        key = (repo_id, revision)
        if key not in mapping:
            raise RuntimeError(f"unresolvable {repo_id}@{revision}")
        return mapping[key]
    return _resolve


def resolver_ab(repo_id, revision):
    return {"main": SHA_A, "v1": SHA_B}[(repo_id, revision) and revision]


# -- URL normalization --------------------------------------------------------

def test_ref_bare_id_and_at_revision():
    assert normalize_repo_ref("org/name") == ("org/name", None)
    assert normalize_repo_ref("org/name@v1") == ("org/name", "v1")
    assert normalize_repo_ref("datasets/org/name") == ("org/name", None)


def test_ref_full_urls():
    assert normalize_repo_ref(
        "https://huggingface.co/datasets/org/name") == ("org/name", None)
    assert normalize_repo_ref(
        "https://huggingface.co/datasets/org/name/tree/main") == ("org/name", "main")
    assert normalize_repo_ref(
        "http://huggingface.co/datasets/org/name/blob/v1/f/x.wav") == ("org/name", "v1")


def test_ref_rejects_garbage():
    for bad in ["", "justname", "a/b/c", "https://evil.com/datasets/org/name",
                "https://huggingface.co/org/name", "org/../etc"]:
        with pytest.raises(ValueError):
            normalize_repo_ref(bad)


# -- create: idempotent, pinned, no URL in paths ------------------------------

def test_create_pins_and_rejects_duplicates(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    datasets = [{"repo": "org/A", "revision": "main"},
                {"repo": "https://huggingface.co/datasets/org/B/tree/v1"}]
    c1, out1 = store.create_campaign("demo", datasets, resolver=resolver_ab)
    assert [o["duplicate"] for o in out1] == [False, False]
    assert out1[0]["pinned_sha"] == SHA_A and out1[1]["pinned_sha"] == SHA_B
    assert c1.job_ids == [out1[0]["job_id"], out1[1]["job_id"]]

    c2, out2 = store.create_campaign("demo", datasets, resolver=resolver_ab)
    assert [o["duplicate"] for o in out2] == [True, True]
    assert [o["job_id"] for o in out2] == c1.job_ids
    assert [o["pinned_sha"] for o in out2] == [SHA_A, SHA_B]
    # no duplicate job files, pins unchanged
    assert sorted(os.listdir(store.jobs_dir)) == sorted(f"{j}.json" for j in c1.job_ids)
    assert store.load_campaign().job_ids == c1.job_ids


def test_create_needs_no_network_for_full_sha(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))

    def _boom(repo_id, revision):
        raise AssertionError("resolver must not be called for full SHAs")

    _, out = store.create_campaign(
        "demo", [{"repo": f"org/A@{SHA_A}"}], resolver=_boom)
    assert out[0]["pinned_sha"] == SHA_A and out[0]["status"] == "CREATED"


def test_no_raw_url_in_filesystem_artifacts(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign(
        "demo", [{"repo": "https://huggingface.co/datasets/Org/Name/tree/main"}],
        resolver=resolver_ab)
    jid = out[0]["job_id"]
    assert "/" not in jid and "http" not in jid and " " not in jid
    for root, _ds, files in os.walk(str(tmp_path)):
        for f in files:
            assert "http" not in f and "://" not in f


def test_unresolvable_becomes_blocked_not_crash(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign(
        "demo", [{"repo": "org/A", "revision": "nope"}],
        resolver=fake_resolver({}))
    assert out[0]["status"] == "BLOCKED"
    job = store.load_job(out[0]["job_id"])
    assert job.pinned_sha is None and job.blockers


def test_empty_datasets_fail_closed(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    with pytest.raises(ValueError):
        store.create_campaign("demo", [], resolver=resolver_ab)


def test_root_holding_other_campaign_rejected(tmp_path):
    s = CampaignStore(str(tmp_path / "c"))
    s.create_campaign("one", [{"repo": f"org/A@{SHA_A}"}], resolver=resolver_ab)
    with pytest.raises(ValueError):
        s.create_campaign("two", [{"repo": f"org/B@{SHA_B}"}], resolver=resolver_ab)


# -- resume: idempotent, pins immutable, drift reported ------------------------

def test_resume_twice_identical_and_pins_unchanged(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    store.create_campaign("demo", [{"repo": "org/A", "revision": "main"}],
                          resolver=resolver_ab)
    r1 = store.resume(resolver=resolver_ab)
    r2 = store.resume(resolver=resolver_ab)
    assert r1 == r2
    assert r1["jobs"][0]["action"] == "none"
    assert store.load_job(r1["jobs"][0]["job_id"]).pinned_sha == SHA_A


def test_resume_retries_blocked_and_reports_drift(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("demo", [{"repo": "org/A", "revision": "main"}],
                                   resolver=fake_resolver({}))
    assert out[0]["status"] == "BLOCKED"
    # operator fixes network: resume pins it (adopting the canonical job_id)
    r = store.resume(resolver=resolver_ab)
    assert r["jobs"][0]["action"] == "pinned"
    assert r["jobs"][0]["pinned_sha"] == SHA_A
    jid = r["jobs"][0]["job_id"]
    assert not jid.endswith("-unpinned")
    # upstream moved: drift reported, stored pin UNCHANGED
    moved = fake_resolver({("org/A", "main"): SHA_B})
    r2 = store.resume(resolver=moved, check_remote=True)
    assert r2["drift"][0]["upstream_now"] == SHA_B
    assert r2["drift"][0]["pinned_sha"] == SHA_A
    assert store.load_job(jid).pinned_sha == SHA_A


# -- batches: stable ids, convention, checkpoint -------------------------------

def test_batch_ids_stable_and_ordered(tmp_path):
    files = [(f"f/{i:03d}.wav", 1000) for i in range(5)]
    b1 = plan_batches("org/A", SHA_A, None, files, max_files=2)
    b2 = plan_batches("org/A", SHA_A, None, files, max_files=2)
    assert [b.batch_id for b in b1] == [b.batch_id for b in b2]
    assert [b.index for b in b1] == [0, 1, 2]
    assert sum(len(b.files) for b in b1) == 5
    # different content => different ids
    b3 = plan_batches("org/A", SHA_A, None, files + [("f/x.wav", 1)], max_files=2)
    assert {x.batch_id for x in b3} != {x.batch_id for x in b1}
    # helper agrees with model rule
    assert b1[0].batch_id == batch_id_for("org/A", SHA_A, 0, b1[0].files)


def test_plan_job_batches_convention_and_replan_stable(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("demo", [{"repo": f"org/A@{SHA_A}"}],
                                   resolver=resolver_ab)
    jid = out[0]["job_id"]
    files = [("a.wav", 10), ("b.wav", None)]  # unknown size tolerated
    batches = store.plan_job_batches(jid, files, max_files=10)
    assert len(batches) == 1
    b = batches[0]
    assert b.n_unknown_bytes == 1 and b.total_bytes == 10
    assert b.checkpoint_rel == f"checkpoints/{jid}/{b.batch_id}.jsonl"
    assert store.checkpoint_file(jid, b.batch_id).endswith(
        os.path.join("checkpoints", jid, f"{b.batch_id}.jsonl"))
    again = store.plan_job_batches(jid, files, max_files=10)
    assert [x.batch_id for x in again] == [x.batch_id for x in batches]
    assert store.load_job(jid).status == "BATCHED"


def test_oversize_file_gets_own_flagged_batch():
    batches = plan_batches("org/A", SHA_A, None,
                           [("small.wav", 10), ("huge.wav", 10**9), ("s2.wav", 10)],
                           max_files=100, max_bytes=1000)
    assert len(batches) == 3
    assert batches[1].files == ["huge.wav"]
    assert batches[1].result.get("oversize") is True


def test_plan_unpinned_job_fails_closed(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("demo", [{"repo": "org/A"}],
                                   resolver=fake_resolver({}))
    with pytest.raises(ValueError):
        store.plan_job_batches(out[0]["job_id"], [("a.wav", 1)])


def test_batch_checkpoint_roundtrip(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("demo", [{"repo": f"org/A@{SHA_A}"}],
                                   resolver=resolver_ab)
    (b,) = store.plan_job_batches(out[0]["job_id"], [("a.wav", 5)])
    ckpt = store.batch_checkpoint(b)
    ckpt.mark("a.wav", "DONE")
    prog = ckpt.load()
    assert ckpt.is_done("a.wav", prog)


# -- CLI -----------------------------------------------------------------------

def test_cli_create_status_resume_offline(tmp_path, capsys):
    from ornix_dataset.cli import main

    inp = tmp_path / "camp.yaml"
    inp.write_text(
        f"name: cli-demo\ndatasets:\n  - repo: org/A@{SHA_A}\n"
        f"  - repo: org/B@{SHA_B}\n", encoding="utf-8")
    root = str(tmp_path / "store")
    assert main(["campaign", "create", "--input", str(inp), "--root", root]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["outcomes"]) == 2
    assert main(["campaign", "status", "--root", root]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["n_jobs"] == 2 and st["by_status"] == {"CREATED": 2}
    # create again: duplicates, same pins
    assert main(["campaign", "create", "--input", str(inp), "--root", root]) == 0
    out2 = json.loads(capsys.readouterr().out)
    assert all(o["duplicate"] for o in out2["outcomes"])
    assert main(["campaign", "resume", "--root", root]) == 0
    r = json.loads(capsys.readouterr().out)
    assert [j["action"] for j in r["jobs"]] == ["none", "none"]
