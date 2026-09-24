"""MD-004 Gate 4 — PROCESSING_VERIFIED (fake analyzer + mock transport)."""

import os
import threading

from ornix_dataset.campaign import (
    Budget,
    CampaignStore,
    ReservationLedger,
    gate_batch_release,
    manifest_to_source_record,
    qc_done_files,
    run_batch,
    run_qc_batch,
)
from ornix_dataset.ingestion.hf_downloader import DownloadConfig
from ornix_dataset.testing.fakes import FakeAnalyzer, FakeNet
from ornix_dataset.util.jsonl import append_jsonl, load_jsonl

SHA = "e" * 40
GB = 1024**3


def downloaded_store(tmp, batches_files, payload_size=2048):
    """Create campaign, plan, and download every batch. Returns store/job/batches."""
    net = FakeNet(tmp)
    store = CampaignStore(str(tmp / "c"))
    _, out = store.create_campaign("proc", [{"repo": f"org/A@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    flat = [(f, payload_size) for files in batches_files for f in files]
    for f, _ in flat:
        net.add(f, os.urandom(payload_size))
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=100, max_batch_bytes=10 * GB)
    planned = store.plan_job_batches(jid, flat, budget=budget)
    # regroup planned batches to match requested grouping is unnecessary:
    # tests below use one batch per store; multi-batch tests plan separately.
    assert len(planned) == 1
    b = planned[0]
    tree = {f: (payload_size, net.sha(f)) for f, _ in flat}
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024,
                    cfg=DownloadConfig(file_workers=4, retry_jitter=False),
                    download_fn=net, tree=tree)
    assert rep["ok"], rep
    return store, jid, planned, net


def test_qc_full_pass_and_evidence_preserved(tmp_path):
    store, jid, (b,), _ = downloaded_store(tmp_path, [["a.wav", "b.wav", "c.wav"]])
    rep = run_qc_batch(store, jid, b.batch_id, FakeAnalyzer())
    assert rep["ok"] and rep["reason"] == "complete"
    assert rep["n_evidence"] == 3 and rep["n_accepted"] == 3
    assert rep["pending"] == []
    assert store.load_batch(b.batch_id).status == "BATCH_PROCESSED"
    assert qc_done_files(store, store.load_batch(b.batch_id)) == \
        ["a.wav", "b.wav", "c.wav"]


def test_shard_container_blocked_never_qc(tmp_path):
    store, jid, (b,), _ = downloaded_store(tmp_path, [["a.wav", "shard.parquet"]])
    rep = run_qc_batch(store, jid, b.batch_id, FakeAnalyzer())
    assert rep["ok"] and rep["n_shard_blocked"] == 1 and rep["n_accepted"] == 1
    rows = load_jsonl(os.path.join(
        store.root, "qc", "runs", f"{jid}__{b.batch_id}", "evidence.jsonl"))
    blocked = [r for r in rows if r.get("decision") == "BLOCKED"]
    assert len(blocked) == 1 and "parquet" in blocked[0]["reason"]


def test_qc_error_isolated_and_retryable(tmp_path):
    store, jid, (b,), _ = downloaded_store(tmp_path, [["a.wav", "bad.wav"]])
    r1 = run_qc_batch(store, jid, b.batch_id, FakeAnalyzer(fail_on={"bad.wav"}))
    assert r1["ok"] is False and r1["pending"] == ["bad.wav"]
    assert store.load_batch(b.batch_id).status == "IN_PROGRESS"
    r2 = run_qc_batch(store, jid, b.batch_id, FakeAnalyzer())
    assert r2["ok"] is True
    assert store.load_batch(b.batch_id).status == "BATCH_PROCESSED"


def test_qc_refuses_undownloaded_batch(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("proc", [{"repo": f"org/A@{SHA}"}],
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    (b,) = store.plan_job_batches(jid, [("a.wav", 10)])
    rep = run_qc_batch(store, jid, b.batch_id, FakeAnalyzer())
    assert rep["ok"] is False and "PLANNED" in rep["reason"]


def test_source_identity_stable_from_manifest(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("proc", [{"repo": f"org/A@{SHA}"}],
                                   resolver=lambda r, v: SHA)
    job = store.load_job(out[0]["job_id"])
    row = {"source_uri": f"hf://datasets/org/A@{SHA}/a.wav",
           "original_file_id": "a.wav", "sha256": "9" * 64, "size": 10,
           "staged_path": "a.wav"}
    s1 = manifest_to_source_record(row, job, "/staging")
    s2 = manifest_to_source_record(row, job, "/staging")
    assert s1.source_id == s2.source_id and s1.source_id.startswith("SRC_")
    assert s1.source_revision == SHA and s1.staged_path == "/staging/a.wav"


def _two_batch_job(tmp_path, payload_size=1024):
    net = FakeNet(tmp_path)
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("proc2", [{"repo": f"org/A@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    names = ["a.wav", "b.wav"]
    for n in names:
        net.add(n, os.urandom(payload_size))
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=1, max_batch_bytes=10 * GB)
    batches = store.plan_job_batches(jid, [(n, payload_size) for n in names],
                                     budget=budget)
    assert len(batches) == 2
    tree = {n: (payload_size, net.sha(n)) for n in names}
    kw = {"caps": budget.stage_caps, "min_free_bytes": 1024,
          "cfg": DownloadConfig(file_workers=2, retry_jitter=False),
          "download_fn": net, "tree": tree}
    for b in batches:
        assert run_batch(store, jid, b.batch_id, **kw)["ok"]
    return store, jid, batches


def test_gate_refuses_until_global_evidence_whole(tmp_path):
    store, jid, batches = _two_batch_job(tmp_path)
    run_qc_batch(store, jid, batches[0].batch_id, FakeAnalyzer())
    rep = gate_batch_release(store, jid, batches[0].batch_id)
    assert rep["ok"] is False and rep["reason"] == "global-evidence-incomplete"
    assert rep["missing_batches"] == [batches[1].batch_id]
    assert store.load_batch(batches[0].batch_id).status == "BATCH_PROCESSED"
    # second batch processed => gate passes for both
    run_qc_batch(store, jid, batches[1].batch_id, FakeAnalyzer())
    for b in batches:
        g = gate_batch_release(store, jid, b.batch_id)
        assert g["ok"] is True, g
        assert store.load_batch(b.batch_id).status == "RELEASE_READY"
    assert os.path.exists(os.path.join(store.root, "releases",
                                       f"{jid}.GATE.json"))


def test_gate_reports_duplicates_without_dropping(tmp_path):
    # identical bytes in two files: duplicate group reported, release allowed
    net = FakeNet(tmp_path)
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("dup", [{"repo": f"org/A@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    blob = os.urandom(1024)
    net.add("a.wav", blob)
    net.add("b.wav", blob)
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=1, max_batch_bytes=10 * GB)
    batches = store.plan_job_batches(jid, [("a.wav", 1024), ("b.wav", 1024)],
                                     budget=budget)
    kw = {"caps": budget.stage_caps, "min_free_bytes": 1024,
          "download_fn": net,
          "tree": {"a.wav": (1024, net.sha("a.wav")),
                   "b.wav": (1024, net.sha("b.wav"))}}
    for b in batches:
        assert run_batch(store, jid, b.batch_id, **kw)["ok"]
        assert run_qc_batch(store, jid, b.batch_id, FakeAnalyzer())["ok"]
    g = gate_batch_release(store, jid, batches[0].batch_id)
    assert g["ok"] is True
    assert len(g["duplicates"]) == 1
    assert sorted(g["duplicates"][0]) == ["a.wav", "b.wav"]


def test_download_overlaps_qc(tmp_path):
    # Gate 4 core: batch B downloads while batch A is in QC
    net = FakeNet(tmp_path)
    net.delay = 0.5  # slow transport forces real temporal overlap
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("ov", [{"repo": f"org/A@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    for n in ("a.wav", "b.wav"):
        net.add(n, os.urandom(1024))
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=1, max_batch_bytes=10 * GB)
    ba, bb = store.plan_job_batches(jid, [("a.wav", 1024), ("b.wav", 1024)],
                                    budget=budget)
    tree = {"a.wav": (1024, net.sha("a.wav")),
            "b.wav": (1024, net.sha("b.wav"))}
    kw = {"caps": budget.stage_caps, "min_free_bytes": 1024,
          "cfg": DownloadConfig(file_workers=2, retry_jitter=False),
          "download_fn": net, "tree": tree}
    assert run_batch(store, jid, ba.batch_id, **kw)["ok"]
    dl_out, qc_out = {}, {}
    t_dl = threading.Thread(
        target=lambda: dl_out.update(
            run_batch(store, jid, bb.batch_id, **kw)))
    t_qc = threading.Thread(
        target=lambda: qc_out.update(
            run_qc_batch(store, jid, ba.batch_id, FakeAnalyzer(delay=0.3))))
    t_dl.start()
    t_qc.start()
    t_dl.join(timeout=60)
    t_qc.join(timeout=60)
    assert dl_out.get("ok") is True and qc_out.get("ok") is True
    assert store.load_batch(bb.batch_id).status == "IN_PROGRESS"
    assert store.load_batch(ba.batch_id).status == "BATCH_PROCESSED"


def test_cli_qc_gate_fail_closed(tmp_path, capsys):
    from ornix_dataset.cli import main

    root = str(tmp_path / "store")
    assert main(["campaign", "qc", "--root", root, "--job", "j",
                 "--batch", "b", "--policy", "p"]) == 2
    assert main(["campaign", "gate", "--root", root,
                 "--job", "j", "--batch", "b"]) == 2
