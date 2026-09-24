"""MD-007 Gate 7 — CAMPAIGN_E2E_VERIFIED (mock network, REAL disk/state).

Full lifecycle on real filesystem state: create → inventory → plan →
download → QC → gate → prepare → publish → cleanup → DONE, for two datasets.
Failure scenarios (checksum mismatch, crash mid-run, disk watermark) use the
same code paths as production; only the byte transport and the Hub are faked.
"""

import hashlib
import json
import os
from types import SimpleNamespace

import yaml

from ornix_dataset.campaign import (
    Budget,
    CampaignStore,
    ReservationLedger,
    WatermarkGate,
    batch_staging_dir,
    cleanup_batch,
    fetch_job_inventory,
    gate_batch_release,
    list_releases,
    prepare_release,
    publish_release,
    pump_campaign,
    read_manifest,
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
from ornix_dataset.util.jsonl import append_jsonl, load_jsonl

SHA = "h" * 40
GB = 1024**3


class FakeTreeApi:
    def __init__(self, files):
        self.files = files  # [(path, size)]

    def list_repo_tree(self, repo_id, revision=None, **kw):
        assert len(revision) == 40  # pinned only, never a branch
        return [SimpleNamespace(type="file", path=p, size=s, lfs={})
                for p, s in self.files]


def build_world(tmp_path, monkeypatch, specs):
    """specs: {repo_suffix: [filenames]}. Returns wired world dict."""
    hub = FakeHub()
    wire_hub(monkeypatch, hub)
    net = FakeNet(tmp_path)
    store = CampaignStore(str(tmp_path / "c"))
    datasets = [{"repo": f"org/{suf}@{SHA}"} for suf in specs]
    _, out = store.create_campaign("e2e", datasets,
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jobs = {}
    for o, (suf, names) in zip(out, specs.items()):
        for n in names:
            net.add(f"{suf}/{n}", os.urandom(2048))
        tree = FakeTreeApi([(f"{suf}/{n}", 2048) for n in names])
        job = store.load_job(o["job_id"])
        inv = fetch_job_inventory(tree, job)
        budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                        max_files_per_batch=1, max_batch_bytes=10 * GB)
        batches = store.plan_job_batches(
            job.job_id, [(p, s) for p, s, _ in inv], budget=budget)
        jobs[o["job_id"]] = {"names": [f"{suf}/{n}" for n in names],
                             "batches": [b.batch_id for b in batches]}
    return {"hub": hub, "net": net, "store": store, "jobs": jobs}


def accepting_analyzer(store, job_id, batch_id):
    """QC analyzer planting canonical audio; rows carry true staged shas."""
    from ornix_dataset.campaign import read_manifest as _rm
    (mrow,) = [r for r in _rm(batch_staging_dir(
        store.root, job_id, batch_id))][:1]
    paths = qc_paths(store, job_id, batch_id)
    blob = os.urandom(48000)
    # canonical layout mirrors build_release: canonical_dir/basename(audio)
    with open(os.path.join(paths.canonical_dir,
                           mrow["original_file_id"].split("/")[-1]), "wb") as fh:
        fh.write(blob)
    digest = hashlib.sha256(blob).hexdigest()

    class _A(FakeAnalyzer):
        def __call__(self, src, p, audit):
            append_jsonl(p.evidence, {"source_id": src.source_id,
                                      "decision": "ACCEPT"})
            row = {"audio_id": src.original_file_id,
                   "audio": src.original_file_id.split("/")[-1],
                   "language": "vi", "speaker_id": "spk-0", "transcript": "x",
                   "sample_rate": 24000, "channels": 1, "encoding": "PCM_S16LE",
                   "duration_s": 3.0, "source_id": src.source_id,
                   "source_sha256": mrow["sha256"], "audio_sha256": digest,
                   "segment_start_sample_source": 0,
                   "segment_end_sample_source": 72000,
                   "rights_record_id": "rr", "quality_evidence_id": "qe",
                   "quality_policy_version": "v1", "quality_gate": "ACCEPT",
                   "split": "train", "release_id": "R"}
            return SimpleNamespace(
                evidences=[{}],
                accepted=[SimpleNamespace(to_dict=lambda r=row: dict(r))])

    return _A()


def pump_all(world, tmp_path, max_steps=100):
    store, net, hub = world["store"], world["net"], world["hub"]
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024)
    ledger = ReservationLedger(budget.stage_caps)
    wm = WatermarkGate(1024, 2048)
    trees = {n: (2048, net.sha(n)) for names in
             (v["names"] for v in world["jobs"].values()) for n in names}

    def _download(j, b):
        return run_batch(store, j, b, caps=budget.stage_caps,
                         min_free_bytes=1024, ledger=ledger,
                         cfg=DownloadConfig(file_workers=2, retry_jitter=False),
                         download_fn=net, tree=trees)

    def _qc(j, b):
        return run_qc_batch(store, j, b, accepting_analyzer(store, j, b))

    def _publish(rid):
        rec = json.load(open(os.path.join(
            store.root, "releases", f"{rid}.RELEASE.json"), encoding="utf-8"))
        ap = os.path.join(str(tmp_path), f"{rid}.yaml")
        with open(ap, "w") as fh:
            yaml.safe_dump(approval_for(rec["release_dir"], rec["revision"]), fh)
        return publish_release(store, rid, "org/dest", ap, full_hash=True)

    from ornix_dataset.campaign import prepare_release as _prep
    return pump_campaign(
        store, ledger, wm, _download, _qc,
        lambda j, b: gate_batch_release(store, j, b),
        prepare_one=lambda j, b: _prep(store, j, b, export_format="none"),
        publish_one=_publish,
        cleanup_one=lambda j, b: cleanup_batch(store, j, b, ledger=ledger),
        max_steps=max_steps)


def test_full_campaign_two_datasets_end_to_end(tmp_path, monkeypatch):
    world = build_world(tmp_path, monkeypatch,
                        {"DS1": ["a.wav", "b.wav"], "DS2": ["c.wav", "d.wav"]})
    store = world["store"]
    rep = pump_all(world, tmp_path, max_steps=120)
    actions = [e["action"] for e in rep["log"]]
    for want in ("download", "qc", "gate", "prepare", "publish", "cleanup"):
        assert want in actions, actions
    assert len(list_releases(store)) == 4  # one release per batch, no dupes
    for jid in store.load_campaign().job_ids:
        assert store.load_job(jid).status == "DONE"
        for bid in store.load_job(jid).batch_ids:
            batch = store.load_batch(bid)
            assert batch.status == "DONE"
            assert not os.path.exists(batch_staging_dir(store.root, jid, bid))
            archived = load_jsonl(os.path.join(
                store.root, "releases", f"{jid}.{bid}.MANIFEST.jsonl"))
            assert len(archived) == 1  # manifest archived exactly once
    # evidence + releases + receipts kept for recovery/audit
    assert rep["steps"] <= 120


def test_crash_mid_campaign_resumes_to_done(tmp_path, monkeypatch):
    world = build_world(tmp_path, monkeypatch, {"DS1": ["a.wav", "b.wav"]})
    store = world["store"]
    pump_all(world, tmp_path, max_steps=3)  # "crash": stop early, drop objects
    mid = [store.load_batch(b).status
           for j in store.load_campaign().job_ids
           for b in store.load_job(j).batch_ids]
    assert "DONE" not in mid  # genuinely interrupted
    del world  # same disk, fresh objects — like a process restart
    world2 = build_world(tmp_path, monkeypatch, {"DS1": ["a.wav", "b.wav"]})
    rep = pump_all(world2, tmp_path, max_steps=200)
    store2 = world2["store"]
    assert rep["steps"] <= 200
    for jid in store2.load_campaign().job_ids:
        assert store2.load_job(jid).status == "DONE"


def test_checksum_mismatch_bounded_no_loss(tmp_path, monkeypatch):
    world = build_world(tmp_path, monkeypatch, {"DS1": ["good.wav", "bad.wav"]})
    store, net = world["store"], world["net"]
    # remote claims a sha the bytes will never match (one bad file)
    trees = {n: (2048, net.sha(n)) for j in world["jobs"].values()
             for n in j["names"]}
    [bad] = [n for n in trees if n.endswith("bad.wav")]
    trees[bad] = (2048, "0" * 64)
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024)
    ledger = ReservationLedger(budget.stage_caps)
    wm = WatermarkGate(1024, 2048)
    jid = store.load_campaign().job_ids[0]
    bids = store.load_job(jid).batch_ids

    def _download(j, b):
        return run_batch(store, j, b, caps=budget.stage_caps,
                         min_free_bytes=1024, ledger=ledger,
                         cfg=DownloadConfig(file_workers=2, retry_jitter=False),
                         download_fn=net, tree=trees)

    rep = pump_campaign(store, ledger, wm, _download,
                        lambda j, b: {"ok": False},
                        lambda j, b: gate_batch_release(store, j, b),
                        max_steps=10)
    assert rep["steps"] <= 10  # bounded: pump terminates
    states = sorted(store.load_batch(b).status for b in bids)
    assert states == ["IN_PROGRESS", "PLANNED"]  # good done, bad retryable
    for b in bids:
        rows = read_manifest(batch_staging_dir(store.root, jid, b))
        assert all(r["verify_method"] == "FULL_SHA256" for r in rows)


def test_disk_watermark_stops_admission(tmp_path, monkeypatch):
    world = build_world(tmp_path, monkeypatch, {"DS1": ["a.wav"]})
    store, net = world["store"], world["net"]
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024)
    ledger = ReservationLedger(budget.stage_caps)
    closed = WatermarkGate(high_watermark_bytes=10**30,
                           low_watermark_bytes=10**30)
    jid = store.load_campaign().job_ids[0]
    bid = store.load_job(jid).batch_ids[0]
    rep = pump_campaign(
        store, ledger, closed,
        lambda j, b: run_batch(store, j, b, caps=budget.stage_caps,
                               min_free_bytes=1024, ledger=ledger,
                               download_fn=net),
        lambda j, b: {"ok": False},
        lambda j, b: gate_batch_release(store, j, b),
        max_steps=5)
    assert any(e["action"] == "stopped-disk" for e in rep["log"])
    assert net.calls == []  # zero bytes moved while stopped
    assert store.load_batch(bid).status == "PLANNED"
