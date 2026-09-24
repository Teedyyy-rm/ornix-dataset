"""MD-006 Gate 6 — CLEANUP_VERIFIED (FakeHub + mock transport)."""

import json
import os

import yaml

from ornix_dataset.campaign import (
    Budget,
    CampaignStore,
    ReservationLedger,
    WatermarkGate,
    batch_staging_dir,
    cleanup_batch,
    eligibility,
    gate_batch_release,
    manifest_archive_path,
    prepare_release,
    publish_release,
    pump_campaign,
    run_batch,
    run_qc_batch,
)
from ornix_dataset.campaign.processing import qc_paths
from ornix_dataset.ingestion.hf_downloader import DownloadConfig
from ornix_dataset.testing.fakes import (
    FakeAnalyzer,
    FakeHub,
    FakeNet,
    approval_for,
    wire_hub,
)
from test_publish_campaign import ready_batch

SHA = "g" * 40
GB = 1024**3


def published_batch(tmp_path, monkeypatch, full_hash=True, name="a.wav"):
    """A batch driven download->qc->gate->prepare->publish (all fakes)."""
    hub = FakeHub()
    wire_hub(monkeypatch, hub)
    net = FakeNet(tmp_path)
    net.add(name, os.urandom(2048))
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("cl", [{"repo": f"org/A@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=100, max_batch_bytes=10 * GB)
    (b,) = store.plan_job_batches(jid, [(name, 2048)], budget=budget)
    kw = {"caps": budget.stage_caps, "min_free_bytes": 1024,
          "cfg": DownloadConfig(file_workers=2, retry_jitter=False),
          "download_fn": net, "tree": {name: (2048, net.sha(name))}}
    assert run_batch(store, jid, b.batch_id, **kw)["ok"]
    # accepted row matching the REAL staged sha (valid release input)
    from ornix_dataset.campaign import read_manifest
    (mrow,) = read_manifest(batch_staging_dir(store.root, jid, b.batch_id))

    class _AcceptAll(FakeAnalyzer):
        def __call__(self, src, paths, audit):
            from types import SimpleNamespace
            from ornix_dataset.util.jsonl import append_jsonl
            append_jsonl(paths.evidence, {"source_id": src.source_id,
                                          "decision": "ACCEPT"})
            row = {"audio_id": src.original_file_id, "audio": src.original_file_id,
                   "language": "vi", "speaker_id": "spk-0", "transcript": "x",
                   "sample_rate": 24000, "channels": 1, "encoding": "PCM_S16LE",
                   "duration_s": 3.0, "source_id": src.source_id,
                   "source_sha256": mrow["sha256"],
                   "audio_sha256": mrow["sha256"],
                   "segment_start_sample_source": 0,
                   "segment_end_sample_source": 72000,
                   "rights_record_id": "rr", "quality_evidence_id": "qe",
                   "quality_policy_version": "v1", "quality_gate": "ACCEPT",
                   "split": "train", "release_id": "R"}
            return SimpleNamespace(evidences=[{}], accepted=[
                SimpleNamespace(to_dict=lambda r=row: dict(r))])

    assert run_qc_batch(store, jid, b.batch_id, _AcceptAll())["ok"]
    assert gate_batch_release(store, jid, b.batch_id)["ok"]
    # canonical audio must exist for build_release: plant a blob, then point
    # the accepted row shas at it (self-consistent test fixture)
    paths = qc_paths(store, jid, b.batch_id)
    import hashlib
    blob = os.urandom(48000)
    with open(os.path.join(paths.canonical_dir, name), "wb") as fh:
        fh.write(blob)
    # rewrite accepted row shas to the canonical blob (same content family)
    acc_file = os.path.join(paths.root, "BATCH_ACCEPTED.jsonl")
    rows = [json.loads(line) for line in open(acc_file, encoding="utf-8")]
    digest = hashlib.sha256(blob).hexdigest()
    for r in rows:
        r["audio_sha256"] = digest
        r["source_sha256"] = digest
    with open(acc_file, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    # gate receipt already written; refresh duplicates note (content changed
    # only in the test harness, shas stay self-consistent)
    rep = prepare_release(store, jid, b.batch_id, export_format="none")
    assert rep["ok"], rep
    ap = tmp_path / "appr.yaml"
    ap.write_text(yaml.safe_dump(
        approval_for(rep["release_dir"], rep["revision"])), encoding="utf-8")
    pub = publish_release(store, rep["release_id"], "org/dest", str(ap),
                          full_hash=full_hash)
    assert pub["ok"], pub
    return store, jid, b, hub


def test_cleanup_refuses_without_receipt(tmp_path, monkeypatch):
    from test_publish_campaign import ready_batch
    hub = FakeHub()
    wire_hub(monkeypatch, hub)
    store, jid, batch = ready_batch(tmp_path)
    staging = batch_staging_dir(store.root, jid, batch.batch_id)
    os.makedirs(staging, exist_ok=True)
    rep = cleanup_batch(store, jid, batch.batch_id)
    assert rep["ok"] is False and rep["reason"] == "no-release"
    assert os.path.isdir(staging)


def test_cleanup_refuses_sample_only_receipt(tmp_path, monkeypatch):
    store, jid, b, _ = published_batch(tmp_path, monkeypatch, full_hash=False)
    staging = batch_staging_dir(store.root, jid, b.batch_id)
    rep = cleanup_batch(store, jid, b.batch_id)
    assert rep["ok"] is False and rep["reason"] == "sample-only-receipt"
    assert os.path.isdir(staging)


def test_cleanup_deletes_regenerable_keeps_evidence(tmp_path, monkeypatch):
    store, jid, b, _ = published_batch(tmp_path, monkeypatch, full_hash=True)
    staging = batch_staging_dir(store.root, jid, b.batch_id)
    paths = qc_paths(store, jid, b.batch_id)
    assert os.path.isdir(staging) and os.path.isdir(paths.canonical_dir)
    # plant an evidence file that must survive
    ev = os.path.join(paths.root, "evidence.jsonl")
    assert os.path.exists(ev)
    rep = cleanup_batch(store, jid, b.batch_id)
    assert rep["ok"] is True and rep["freed_bytes"] > 0, rep
    assert not os.path.exists(staging)
    assert not os.path.exists(paths.canonical_dir)
    assert os.path.exists(ev)  # evidence kept
    assert os.path.exists(manifest_archive_path(store, jid, b.batch_id))
    assert store.load_batch(b.batch_id).status == "DONE"
    # rerun converges (idempotent)
    rep2 = cleanup_batch(store, jid, b.batch_id)
    assert rep2["ok"] is True and rep2["freed_bytes"] == 0


def test_cleanup_refuses_tampered_release(tmp_path, monkeypatch):
    store, jid, b, _ = published_batch(tmp_path, monkeypatch, full_hash=True)
    batch = store.load_batch(b.batch_id)
    rdir = batch.result["release"]["release_dir"]
    with open(os.path.join(rdir, "TAMPERED.txt"), "w") as fh:
        fh.write("evil")
    rep = cleanup_batch(store, jid, b.batch_id)
    assert rep["ok"] is False and rep["reason"] == "release-dir-changed"
    assert os.path.isdir(batch_staging_dir(store.root, jid, b.batch_id))


def test_cleanup_refuses_in_flight_and_keeps_foreign_files(tmp_path, monkeypatch):
    store, jid, b, _ = published_batch(tmp_path, monkeypatch, full_hash=True)
    ledger = ReservationLedger({s: 10**18 for s in
                                ("cache", "staging", "qc", "upload")})
    ledger.held[b.batch_id] = {"cache": 1, "staging": 1, "qc": 1, "upload": 1}
    rep = cleanup_batch(store, jid, b.batch_id, ledger=ledger)
    assert rep["ok"] is False and rep["reason"] == "batch-in-flight"
    # foreign batch staging untouched by another batch's cleanup
    foreign = os.path.join(store.root, "staging", jid, "b-foreign")
    os.makedirs(foreign, exist_ok=True)
    with open(os.path.join(foreign, "x.bin"), "w") as fh:
        fh.write("keep")
    ledger.held.clear()
    rep2 = cleanup_batch(store, jid, b.batch_id, ledger=ledger)
    assert rep2["ok"] is True
    assert os.path.exists(os.path.join(foreign, "x.bin"))


def test_crash_mid_cleanup_converges(tmp_path, monkeypatch):
    store, jid, b, _ = published_batch(tmp_path, monkeypatch, full_hash=True)
    staging = batch_staging_dir(store.root, jid, b.batch_id)
    # simulate crash halfway: staging half-gone, status not DONE
    for f in os.listdir(staging):
        if not f.endswith(".jsonl"):
            os.remove(os.path.join(staging, f))
    rep = cleanup_batch(store, jid, b.batch_id)
    assert rep["ok"] is True
    assert store.load_batch(b.batch_id).status == "DONE"


def test_pump_drives_two_datasets_to_done(tmp_path, monkeypatch):
    hub = FakeHub()
    wire_hub(monkeypatch, hub)
    net = FakeNet(tmp_path)
    for n in ("a.wav", "b.wav"):
        net.add(n, os.urandom(2048))
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pump", [{"repo": f"org/A@{SHA}"},
                                            {"repo": f"org/B@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jobs = [o["job_id"] for o in out]
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=10, max_batch_bytes=10 * GB)
    trees = {}
    for jid, name in zip(jobs, ("a.wav", "b.wav")):
        store.plan_job_batches(jid, [(name, 2048)], budget=budget)
        trees[name] = (2048, net.sha(name))
    ledger = ReservationLedger(budget.stage_caps)
    wm = WatermarkGate(1024, 2048)

    def _download(j, b):
        return run_batch(store, j, b, caps=budget.stage_caps,
                         min_free_bytes=1024, ledger=ledger,
                         cfg=DownloadConfig(file_workers=2, retry_jitter=False),
                         download_fn=net, tree=trees)

    def _qc(j, b):
        # accept with the REAL staged sha so release validation passes
        from types import SimpleNamespace
        from ornix_dataset.campaign import read_manifest
        from ornix_dataset.util.jsonl import append_jsonl
        (mrow,) = [r for r in read_manifest(batch_staging_dir(
            store.root, j, b)) if True]
        paths = qc_paths(store, j, b)
        import hashlib as _hl
        blob = os.urandom(48000)
        with open(os.path.join(paths.canonical_dir,
                               mrow["original_file_id"]), "wb") as fh:
            fh.write(blob)
        digest = _hl.sha256(blob).hexdigest()
        row = {"audio_id": mrow["original_file_id"],
               "audio": mrow["original_file_id"], "language": "vi",
               "speaker_id": "spk-0", "transcript": "x",
               "sample_rate": 24000, "channels": 1, "encoding": "PCM_S16LE",
               "duration_s": 3.0, "source_id": "SRC_x",
               "source_sha256": digest, "audio_sha256": digest,
               "segment_start_sample_source": 0,
               "segment_end_sample_source": 72000,
               "rights_record_id": "rr", "quality_evidence_id": "qe",
               "quality_policy_version": "v1", "quality_gate": "ACCEPT",
               "split": "train", "release_id": "R"}

        class _A(FakeAnalyzer):
            def __call__(self, src, p, audit):
                append_jsonl(p.evidence, {"source_id": src.source_id,
                                          "decision": "ACCEPT"})
                return SimpleNamespace(
                    evidences=[{}],
                    accepted=[SimpleNamespace(to_dict=lambda r=row: dict(r))])

        return run_qc_batch(store, j, b, _A())

    def _publish(rid):
        from ornix_dataset.campaign import prepare_release as _prep
        # prepare happens via prepare_one; publish needs an approval file
        rec = json.load(open(os.path.join(
            store.root, "releases", f"{rid}.RELEASE.json"), encoding="utf-8"))
        ap = os.path.join(tmp_path, f"{rid}.yaml")
        with open(ap, "w") as fh:
            yaml.safe_dump(approval_for(rec["release_dir"], rec["revision"]),
                           fh)
        return publish_release(store, rid, "org/dest", ap, full_hash=True)

    from ornix_dataset.campaign import prepare_release as _prep
    rep = pump_campaign(store, ledger, wm, _download, _qc,
                        lambda j, b: gate_batch_release(store, j, b),
                        prepare_one=lambda j, b: _prep(
                            store, j, b, export_format="none"),
                        publish_one=_publish,
                        cleanup_one=lambda j, b: cleanup_batch(
                            store, j, b, ledger=ledger),
                        max_steps=60)
    actions = [e["action"] for e in rep["log"]]
    assert "cleanup" in actions and "publish" in actions, actions
    for jid in jobs:
        assert store.load_job(jid).status == "DONE"
        for bid in store.load_job(jid).batch_ids:
            assert store.load_batch(bid).status == "DONE"
    # intermediates gone, evidence + releases kept
    for jid in jobs:
        for bid in store.load_job(jid).batch_ids:
            assert not os.path.exists(batch_staging_dir(store.root, jid, bid))


def test_cli_cleanup_pump_fail_closed(tmp_path, capsys):
    from ornix_dataset.cli import main

    root = str(tmp_path / "store")
    assert main(["campaign", "cleanup", "--root", root,
                 "--job", "j", "--batch", "b"]) == 2
    assert main(["campaign", "pump", "--root", root]) == 2


# -- G1: overlapped pump ----------------------------------------------------------

def _overlap_world(tmp_path, tag):
    import time as _t

    net = FakeNet(tmp_path / tag)
    net.delay = 0.4
    for n in ("a.wav", "b.wav"):
        net.add(n, os.urandom(2048))
    store = CampaignStore(str(tmp_path / f"c-{tag}"))
    _, out = store.create_campaign(f"ov-{tag}", [{"repo": f"org/A@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=1, max_batch_bytes=10 * GB)
    batches = store.plan_job_batches(jid, [("a.wav", 2048), ("b.wav", 2048)],
                                     budget=budget)
    assert len(batches) == 2
    ledger = ReservationLedger(budget.stage_caps)
    wm = WatermarkGate(1024, 2048)
    tree = {n: (2048, net.sha(n)) for n in ("a.wav", "b.wav")}

    def _download(j, b, admitted=False):
        return run_batch(store, j, b, caps=budget.stage_caps,
                         min_free_bytes=1024, ledger=ledger,
                         cfg=DownloadConfig(file_workers=1, retry_jitter=False),
                         download_fn=net, tree=tree, _admitted=admitted)

    def _qc(j, b):
        return run_qc_batch(store, j, b, FakeAnalyzer(delay=0.4))

    def _pump(overlap):
        import time
        t0 = time.monotonic()
        rep = pump_campaign(
            store, ledger, wm, _download, _qc,
            lambda j, b: {"ok": False, "reason": "no-gate-in-this-test"},
            max_steps=8, overlap=overlap, min_free_bytes=1024)
        return rep, time.monotonic() - t0

    return _pump


def test_overlap_downloads_next_while_qc_runs(tmp_path):
    seq_rep, seq_wall = _overlap_world(tmp_path, "seq")(False)
    ov_rep, ov_wall = _overlap_world(tmp_path, "ov")(True)
    for rep in (seq_rep, ov_rep):
        by_id = {}
        for e in rep["log"]:
            by_id.setdefault(e["batch"], []).append(e["action"])
        # both batches fully downloaded+QC'd in both modes
        assert len(by_id) == 2
    assert any(e.get("overlapped") for e in ov_rep["log"])
    assert not any(e.get("overlapped") for e in seq_rep["log"])
    assert ov_wall < seq_wall  # one full stage saved by overlapping


def test_overlap_yields_to_watermark(tmp_path):
    net = FakeNet(tmp_path)
    net.add("a.wav", os.urandom(64))
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("ovw", [{"repo": f"org/A@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=10, max_batch_bytes=10 * GB)
    batches = store.plan_job_batches(jid, [("a.wav", 64)], budget=budget)
    ledger = ReservationLedger(budget.stage_caps)
    closed = WatermarkGate(high_watermark_bytes=10**30,
                           low_watermark_bytes=10**30)
    rep = pump_campaign(
        store, ledger, closed,
        lambda j, b, admitted=False: run_batch(
            store, j, b, caps=budget.stage_caps, min_free_bytes=1024,
            ledger=ledger, download_fn=net, _admitted=admitted),
        lambda j, b: {"ok": False},
        lambda j, b: {"ok": False},
        max_steps=4, overlap=True, min_free_bytes=1024)
    assert any(e["action"] == "stopped-disk" for e in rep["log"])
    assert net.calls == []
    assert store.load_batch(batches[0].batch_id).status == "PLANNED"
