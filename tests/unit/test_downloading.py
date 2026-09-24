"""MD-003 Gate 3 — DOWNLOAD_VERIFIED (adapter level, all offline)."""

import os
import threading
from types import SimpleNamespace

import pytest

from ornix_dataset.campaign import (
    Budget,
    CampaignStore,
    ReservationLedger,
    batch_staging_dir,
    downloaded_files,
    fetch_job_inventory,
    read_manifest,
    run_batch,
)
from ornix_dataset.ingestion.hf_downloader import DownloadConfig
from ornix_dataset.testing.fakes import FakeNet

SHA = "d" * 40
GB = 1024**3


def make_store(tmp, files, resolver=None, **budget_kw):
    store = CampaignStore(str(tmp / "c"))
    _, out = store.create_campaign(
        "dl", [{"repo": f"org/A@{SHA}"}],
        workspace={"min_free_bytes": 1024},
        resolver=resolver or (lambda r, v: SHA))
    jid = out[0]["job_id"]
    kw = {"max_files_per_batch": 100, "max_batch_bytes": 10 * GB}
    kw.update(budget_kw)
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024, **kw)
    batches = store.plan_job_batches(jid, files, budget=budget)
    return store, jid, batches, budget


def payloads(net, names):
    for n in names:
        net.add(n, os.urandom(2048))


# -- Gate 3: concurrent verified download ---------------------------------------

def test_batch_downloads_concurrently_and_hands_off_verified(tmp_path):
    names = [f"f/{i}.wav" for i in range(4)]
    net = FakeNet(tmp_path)
    payloads(net, names)
    net.barrier = threading.Barrier(4)
    store, jid, batches, budget = make_store(
        tmp_path, [(n, 2048) for n in names])
    (b,) = batches
    ledger = ReservationLedger(budget.stage_caps)
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024, ledger=ledger,
                    cfg=DownloadConfig(file_workers=4, retry_jitter=False),
                    download_fn=net,
                    tree={n: (2048, net.sha(n)) for n in names})
    assert rep["ok"] and rep["reason"] == "complete"
    assert net.peak > 1  # truly concurrent at adapter level
    assert rep["network_bytes"] == 4 * 2048
    assert ledger.totals()["held_batches"] == []  # released in finally

    batch = store.load_batch(b.batch_id)
    assert batch.status == "IN_PROGRESS"
    assert downloaded_files(store, batch) == sorted(names)
    rows = read_manifest(batch_staging_dir(store.root, jid, b.batch_id))
    assert len(rows) == 4
    for r in rows:
        assert r["source_uri"].startswith(f"hf://datasets/org/A@{SHA}/")
        assert r["verify_method"] == "FULL_SHA256"
        assert os.path.exists(os.path.join(
            batch_staging_dir(store.root, jid, b.batch_id), r["staged_path"]))


def test_no_reservation_no_start_no_side_effects(tmp_path):
    names = ["a.wav"]
    net = FakeNet(tmp_path)
    payloads(net, names)
    store, jid, batches, _ = make_store(tmp_path, [(names[0], 100)])
    rep = run_batch(store, jid, batches[0].batch_id,
                    caps={"cache": 1, "staging": 1, "qc": 1, "upload": 1},
                    min_free_bytes=1024, download_fn=net)
    assert rep == {"ok": False, "admitted": False,
                   "reason": "no-reservation-or-disk"}
    assert net.calls == []
    assert store.load_batch(batches[0].batch_id).status == "PLANNED"
    assert not os.path.exists(
        batch_staging_dir(store.root, jid, batches[0].batch_id))


def test_corrupt_bytes_never_handed_off(tmp_path):
    names = ["a.wav", "b.wav"]
    net = FakeNet(tmp_path)
    payloads(net, names)
    store, jid, batches, budget = make_store(tmp_path, [(n, 2048) for n in names])
    (b,) = batches
    # remote claims a sha the bytes will never match
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024, download_fn=net,
                    tree={n: (2048, "f" * 64) for n in names})
    assert rep["ok"] is False and rep["admitted"] is True
    assert store.load_batch(b.batch_id).status == "PLANNED"  # retryable
    assert read_manifest(batch_staging_dir(store.root, jid, b.batch_id)) == []


def test_partial_failure_resumes_without_duplicates(tmp_path):
    names = [f"f/{i}.wav" for i in range(4)]
    net = FakeNet(tmp_path)
    payloads(net, names)
    # persistent outage on one file: retries exhaust, the attempt fails
    net.errors["f/2.wav"] = [RuntimeError("net down")] * 10
    store, jid, batches, budget = make_store(
        tmp_path, [(n, 2048) for n in names])
    (b,) = batches
    kw = {"caps": budget.stage_caps, "min_free_bytes": 1024, "download_fn": net,
          "tree": {n: (2048, net.sha(n)) for n in names}}
    r1 = run_batch(store, jid, b.batch_id, **kw)
    assert r1["ok"] is False  # fatal abort on first attempt
    assert store.load_batch(b.batch_id).status == "PLANNED"
    net.errors.clear()  # outage over: resume must finish the remainder
    r2 = run_batch(store, jid, b.batch_id, **kw)
    assert r2["ok"] is True and r2["reason"] == "complete"
    rows = read_manifest(batch_staging_dir(store.root, jid, b.batch_id))
    assert len(rows) == 4  # rewritten, never duplicated
    assert downloaded_files(store, store.load_batch(b.batch_id)) == sorted(names)


def test_rerun_complete_batch_is_noop(tmp_path):
    names = ["a.wav"]
    net = FakeNet(tmp_path)
    payloads(net, names)
    store, jid, batches, budget = make_store(tmp_path, [(names[0], 2048)])
    kw = {"caps": budget.stage_caps, "min_free_bytes": 1024, "download_fn": net,
          "tree": {names[0]: (2048, net.sha(names[0]))}}
    assert run_batch(store, jid, batches[0].batch_id, **kw)["ok"] is True
    n_calls = len(net.calls)
    rep = run_batch(store, jid, batches[0].batch_id, **kw)
    assert rep["ok"] is True and rep["reason"] == "already-complete"
    assert len(net.calls) == n_calls  # zero new bytes


def test_blocked_batch_and_unpinned_job_refused(tmp_path):
    net = FakeNet(tmp_path)
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("dl", [{"repo": "org/A"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: (_ for _ in ()).throw(
                                       RuntimeError("offline")))
    jid = out[0]["job_id"]
    rep = run_batch(store, jid, "b-deadbeef",
                    caps={"cache": 9**18, "staging": 9**18,
                          "qc": 9**18, "upload": 9**18},
                    min_free_bytes=1, download_fn=net)
    assert rep["reason"] == "job-unpinned"

    store2, jid2, batches2, budget2 = make_store(tmp_path, [("huge.wav", 10**18)])
    (big,) = batches2
    assert big.status == "BLOCKED"  # oversize from MD-002
    rep2 = run_batch(store2, jid2, big.batch_id, caps=budget2.stage_caps,
                     min_free_bytes=1024, download_fn=net)
    assert rep2 == {"ok": False, "admitted": False, "reason": "batch-blocked"}


# -- inventory -------------------------------------------------------------------

class FakeApi:
    def __init__(self, entries):
        self.entries = entries
        self.calls = []

    def list_repo_tree(self, repo_id, revision=None, **kw):
        self.calls.append((repo_id, revision))
        return self.entries


def entry(path, size=None, oid=None, typ="file"):
    return SimpleNamespace(type=typ, path=path, size=size,
                           lfs={"oid": oid} if oid else {})


def test_fetch_inventory_at_pinned_commit_only(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("dl", [{"repo": f"org/A@{SHA}"}],
                                   resolver=lambda r, v: SHA)
    job = store.load_job(out[0]["job_id"])
    api = FakeApi([entry("a.wav", 100, "a" * 64), entry("dir/", None, None, "dir"),
                   entry("notes.txt", 5)])
    inv = fetch_job_inventory(api, job)
    assert api.calls == [("org/A", SHA)]  # pinned SHA, never a branch
    assert inv == [("a.wav", 100, "a" * 64), ("notes.txt", 5, None)]
    with pytest.raises(ValueError):
        fetch_job_inventory(api, SimpleNamespace(pinned_sha=None, job_id="x"))


# -- CLI --------------------------------------------------------------------------

def test_cli_download_and_inventory_fail_closed(tmp_path, capsys):
    from ornix_dataset.cli import main

    root = str(tmp_path / "store")
    assert main(["campaign", "download", "--root", root,
                 "--job", "nope", "--batch", "nope"]) == 2
    assert main(["campaign", "inventory", "--root", root, "--job", "nope"]) == 2


# -- G2: per-file handoff ----------------------------------------------------------

def test_verified_file_hands_off_before_batch_finishes(tmp_path):
    """Handoff lands mid-run: while slow.wav is still downloading, some file
    must already be checkpoint-marked AND present in BATCH_MANIFEST (no
    batch-end wait). Completion order is arbitrary; the guarantee is timing."""
    import time

    from ornix_dataset.campaign import batch_staging_dir
    from ornix_dataset.campaign.downloading import manifest_path

    net = FakeNet(tmp_path)
    net.add("fast.wav", os.urandom(1024))
    net.add("slow.wav", os.urandom(1024))
    store, jid, batches, budget = make_store(
        tmp_path, [("fast.wav", 1024), ("slow.wav", 1024)])
    (b,) = batches
    staging = batch_staging_dir(store.root, jid, b.batch_id)
    seen = {}

    def _handler(path, force=False):
        if path == "slow.wav":
            # bounded wait for the first mid-run handoff (must land while
            # this very download is still in flight)
            deadline = time.monotonic() + 15
            prog = None
            while time.monotonic() < deadline:
                prog = store.batch_checkpoint(
                    store.load_batch(b.batch_id)).load()
                if any(s == "DOWNLOADED" for s in prog.states.values()):
                    break
                time.sleep(0.02)
            seen["ckpt"] = dict(prog.states)
            seen["manifest"] = [
                r.get("original_file_id")
                for r in read_manifest(staging)] if os.path.exists(
                    manifest_path(staging)) else []
        p = os.path.join(net.dir, path.replace("/", "_"))
        with open(p, "wb") as fh:
            fh.write(net.payload[path])
        return p

    net.handler = _handler
    ledger = ReservationLedger(budget.stage_caps)
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024, ledger=ledger,
                    cfg=DownloadConfig(file_workers=1, retry_jitter=False),
                    download_fn=net,
                    tree={"fast.wav": (1024, net.sha("fast.wav")),
                          "slow.wav": (1024, net.sha("slow.wav"))})
    assert rep["ok"] is True
    assert list(seen["ckpt"].values()) == ["DOWNLOADED"] or (
        len(seen["ckpt"]) >= 1
        and set(seen["ckpt"].values()) == {"DOWNLOADED"})
    assert set(seen["manifest"]) <= {"fast.wav", "slow.wav"}
    assert 1 <= len(seen["manifest"]) <= 2
    rows = read_manifest(staging)
    assert sorted(r["original_file_id"] for r in rows) == ["fast.wav", "slow.wav"]
