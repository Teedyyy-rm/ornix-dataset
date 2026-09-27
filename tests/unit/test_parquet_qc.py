"""Parquet expansion + parallel QC (anti-slow-download work). Offline."""

import os

import pytest

pyarrow = pytest.importorskip("pyarrow")

from ornix_dataset.campaign import (
    Budget,
    CampaignStore,
    expand_batch_parquet,
    is_parquet_path,
    qc_workers,
    run_batch,
    run_qc_batch,
)
from ornix_dataset.campaign.downloading import read_manifest
from ornix_dataset.ingestion.hf_downloader import DownloadConfig
from ornix_dataset.testing.fakes import FakeAnalyzer, FakeNet
from ornix_dataset.util.jsonl import load_jsonl

SHA = "f" * 40
GB = 1024**3


def _wav_bytes(duration_s=1.0, sr=24000):
    import numpy as np
    import soundfile as sf
    n = int(duration_s * sr)
    data = (0.1 * np.sin(2 * np.pi * 440 *
                         np.arange(n) / sr)).astype("float32")
    path = os.path.join("/tmp", f"t-{os.getpid()}-{duration_s}.wav")
    sf.write(path, data, sr)
    with open(path, "rb") as fh:
        return fh.read()


def _parquet_bytes(n_clips=3):
    import pyarrow as pa
    import pyarrow.parquet as pq
    blob = _wav_bytes()
    audio = [{"bytes": blob, "path": f"clip_{i}.wav"} for i in range(n_clips)]
    text = [f"cau {i}" for i in range(n_clips)]
    tbl = pa.table({"audio": audio, "transcription": text})
    path = os.path.join("/tmp", f"shard-{os.getpid()}-{n_clips}.parquet")
    pq.write_table(tbl, path)
    with open(path, "rb") as fh:
        return fh.read()


def _downloaded_parquet_store(tmp_path, n_clips=3):
    net = FakeNet(tmp_path)
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pq", [{"repo": f"org/P@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    blob = _parquet_bytes(n_clips)
    net.add("data/shard.parquet", blob)
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=100, max_batch_bytes=10 * GB)
    (b,) = store.plan_job_batches(jid, [("data/shard.parquet", len(blob))],
                                  budget=budget)
    tree = {"data/shard.parquet": (len(blob), net.sha("data/shard.parquet"))}
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024,
                    cfg=DownloadConfig(file_workers=1, retry_jitter=False),
                    download_fn=net, tree=tree)
    assert rep["ok"], rep
    return store, jid, b


def _multi_shard_store(tmp_path, n_shards=2, n_clips=3):
    """One batch holding several parquet shards (the disk-bounding case)."""
    net = FakeNet(tmp_path)
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pqb", [{"repo": f"org/M@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    files, tree = [], {}
    for i in range(n_shards):
        name = f"data/part-{i}.parquet"
        blob = _parquet_bytes(n_clips + i)
        net.add(name, blob)
        files.append((name, len(blob)))
        tree[name] = (len(blob), net.sha(name))
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=100, max_batch_bytes=10 * GB)
    (b,) = store.plan_job_batches(jid, files, budget=budget)
    return store, jid, b, net, budget, tree


def _parquet_files_on_disk(staging):
    out = []
    for dp, _ds, fs in os.walk(staging):
        out.extend(f for f in fs if f.lower().endswith(".parquet"))
    return out


def test_download_extracts_and_drops_parent(tmp_path):
    store, jid, b, net, budget, tree = _multi_shard_store(tmp_path, 1, 3)
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024,
                    cfg=DownloadConfig(file_workers=1, retry_jitter=False),
                    download_fn=net, tree=tree)
    assert rep["ok"], rep
    from ornix_dataset.campaign.downloading import batch_staging_dir
    staging = batch_staging_dir(store.root, jid, b.batch_id)
    # Bounded disk: the parent shard bytes are gone, clips are on disk.
    assert _parquet_files_on_disk(staging) == []
    rows = read_manifest(staging)
    clips = [r for r in rows if "#row-" in r["original_file_id"]]
    assert len(clips) == 3
    assert all(os.path.exists(os.path.join(staging, c["staged_path"]))
               for c in clips)
    assert rep["n_extracted"] == 1
    assert rep["parents_dropped"] == 1
    assert rep["n_parents"] == 1 and rep["n_clips"] == 3
    # Provenance survives on the parent row even though bytes are gone.
    parent = [r for r in rows if r["original_file_id"] == "data/part-0.parquet"]
    assert parent and parent[0]["sha256"] == net.sha("data/part-0.parquet")
    # Checkpoint terminal state is EXTRACTED (not resurrected to DOWNLOADED).
    from ornix_dataset.campaign.downloading import downloaded_files
    assert downloaded_files(store, store.load_batch(b.batch_id)) == \
        ["data/part-0.parquet"]


def test_only_one_shard_resident_at_a_time(tmp_path, monkeypatch):
    store, jid, b, net, budget, tree = _multi_shard_store(tmp_path, 3, 2)
    import ornix_dataset.campaign.downloading as dl_mod
    observed = []
    real = dl_mod.expand_parquet_row

    def _spy(staging, row, *a, **k):
        # How many parent shards are on disk at expansion time?
        observed.append(len(_parquet_files_on_disk(staging)))
        return real(staging, row, *a, **k)

    monkeypatch.setattr(dl_mod, "expand_parquet_row", _spy)
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024,
                    cfg=DownloadConfig(file_workers=1, verify_workers=1,
                                       retry_jitter=False),
                    download_fn=net, tree=tree)
    assert rep["ok"], rep
    assert observed, "expansion never ran"
    # verify_workers=1 => exactly one shard is staged/unpacked/dropped at a
    # time: a parent container is never resident while another one waits.
    assert max(observed) == 1, observed
    from ornix_dataset.campaign.downloading import batch_staging_dir
    staging = batch_staging_dir(store.root, jid, b.batch_id)
    assert _parquet_files_on_disk(staging) == []


def test_resident_shards_bounded_by_verify_workers(tmp_path, monkeypatch):
    """The bound scales with staging concurrency, and is never unbounded."""
    store, jid, b, net, budget, tree = _multi_shard_store(tmp_path, 3, 2)
    import ornix_dataset.campaign.downloading as dl_mod
    observed = []
    real = dl_mod.expand_parquet_row

    def _spy(staging, row, *a, **k):
        observed.append(len(_parquet_files_on_disk(staging)))
        return real(staging, row, *a, **k)

    monkeypatch.setattr(dl_mod, "expand_parquet_row", _spy)
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024,
                    cfg=DownloadConfig(file_workers=4, verify_workers=2,
                                       retry_jitter=False),
                    download_fn=net, tree=tree)
    assert rep["ok"], rep
    assert max(observed) <= 2, observed
    from ornix_dataset.campaign.downloading import batch_staging_dir
    staging = batch_staging_dir(store.root, jid, b.batch_id)
    assert _parquet_files_on_disk(staging) == []


def test_rerun_reaps_parent_left_by_crash(tmp_path):
    store, jid, b, net, budget, tree = _multi_shard_store(tmp_path, 1, 2)
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024,
                    cfg=DownloadConfig(file_workers=1, retry_jitter=False),
                    download_fn=net, tree=tree)
    assert rep["ok"] and rep["parents_dropped"] == 1
    # Simulate a crash after clips landed but before the parent was unlinked.
    from ornix_dataset.campaign.downloading import batch_staging_dir
    staging = batch_staging_dir(store.root, jid, b.batch_id)
    ghost = os.path.join(staging, "data__part-0.parquet")
    with open(ghost, "wb") as fh:
        fh.write(b"leftover")
    again = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                      min_free_bytes=1024,
                      cfg=DownloadConfig(file_workers=1, retry_jitter=False),
                      download_fn=net, tree=tree)
    assert again["ok"] and again["reason"] == "already-complete", again
    assert "data/part-0.parquet" in again.get("parents_reaped", [])
    assert not os.path.exists(ghost)


def test_unexpandable_parquet_keeps_bytes(tmp_path):
    """Fail-closed: an unreadable shard is never dropped without clips."""
    net = FakeNet(tmp_path)
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pqbad2", [{"repo": f"org/N@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    bad = b"definitely not parquet"
    net.add("junk.parquet", bad)
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=100, max_batch_bytes=10 * GB)
    (b,) = store.plan_job_batches(jid, [("junk.parquet", len(bad))],
                                  budget=budget)
    tree = {"junk.parquet": (len(bad), net.sha("junk.parquet"))}
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024,
                    cfg=DownloadConfig(file_workers=1, retry_jitter=False),
                    download_fn=net, tree=tree)
    assert rep["ok"], rep
    assert rep["n_extract_blocked"] == 1
    assert rep["parents_dropped"] == 0
    from ornix_dataset.campaign.downloading import batch_staging_dir
    staging = batch_staging_dir(store.root, jid, b.batch_id)
    assert len(_parquet_files_on_disk(staging)) == 1
    # and QC turns it into auditable evidence, not a silent drop
    rq = run_qc_batch(store, jid, b.batch_id, FakeAnalyzer())
    assert rq["n_shard_blocked"] == 1, rq


def test_is_parquet_path():
    assert is_parquet_path("a/b.parquet")
    assert is_parquet_path("x.PARQUET")
    assert not is_parquet_path("a/b.wav")
    assert not is_parquet_path("a/b.parquet#row-3".split("#")[0] + ".wav")


def test_qc_workers_env(monkeypatch):
    monkeypatch.setenv("ORNIX_QC_WORKERS", "1")
    assert qc_workers() == 1
    monkeypatch.setenv("ORNIX_QC_WORKERS", "99")
    assert qc_workers() == 16
    monkeypatch.delenv("ORNIX_QC_WORKERS")
    assert 2 <= qc_workers() <= 16


def test_expand_parquet_carries_transcript(tmp_path):
    """Clips + transcript land in the manifest at DOWNLOAD time."""
    store, jid, b = _downloaded_parquet_store(tmp_path, n_clips=3)
    from ornix_dataset.campaign.downloading import batch_staging_dir
    staging = batch_staging_dir(store.root, jid, b.batch_id)
    rows = read_manifest(staging)
    clips = [r for r in rows if "#row-" in r["original_file_id"]]
    assert len(clips) == 3
    assert sorted(c["source_transcript"] for c in clips) == \
        ["cau 0", "cau 1", "cau 2"]
    # QC-time expansion is now only a safety net: it must be a no-op.
    job = store.load_job(jid)
    rep = expand_batch_parquet(store, job, store.load_batch(b.batch_id),
                               staging)
    assert rep["n_clips_added"] == 0, rep
    assert all(r.get("note") == "already-expanded" for r in rep["reports"]), rep


def test_qc_parquet_end_to_end_no_block(tmp_path):
    store, jid, b = _downloaded_parquet_store(tmp_path, n_clips=3)
    rep = run_qc_batch(store, jid, b.batch_id, FakeAnalyzer())
    assert rep["ok"], rep
    assert rep["n_shard_blocked"] == 0, rep
    assert rep["n_accepted"] == 3, rep
    # clips were expanded at download time, so QC extracts nothing new
    assert rep["n_extracted_clips"] == 0, rep


def test_qc_parallel_matches_sequential(tmp_path, monkeypatch):
    store, jid, b = _downloaded_parquet_store(tmp_path, n_clips=6)
    monkeypatch.setenv("ORNIX_QC_WORKERS", "1")
    r1 = run_qc_batch(store, jid, b.batch_id, FakeAnalyzer())
    assert r1["ok"], r1
    from ornix_dataset.campaign.processing import qc_paths
    ev1 = load_jsonl(os.path.join(
        qc_paths(store, jid, b.batch_id).root, "evidence.jsonl"))

    # Reset batch to IN_PROGRESS for a second pass with more workers.
    import json as _json
    b2 = store.load_batch(b.batch_id)
    b2.status = "IN_PROGRESS"
    store.save_batch(b2)
    ckpt = store.batch_checkpoint(b2)
    for row in read_manifest(
            os.path.join(store.root, "staging", jid, b.batch_id)):
        fid = row["original_file_id"]
        if fid.endswith(".parquet") and "#row-" not in fid:
            continue
        ckpt.mark(fid, "PENDING_REDO")
    # Fresh QC dir for the parallel pass.
    import shutil as _sh
    _sh.rmtree(qc_paths(store, jid, b.batch_id).root, ignore_errors=True)
    monkeypatch.setenv("ORNIX_QC_WORKERS", "6")
    r2 = run_qc_batch(store, jid, b.batch_id, FakeAnalyzer())
    assert r2["ok"], r2
    ev2 = load_jsonl(os.path.join(
        qc_paths(store, jid, b.batch_id).root, "evidence.jsonl"))
    assert r1["n_evidence"] == r2["n_evidence"] == 6
    assert r1["n_accepted"] == r2["n_accepted"] == 6
    # Same evidence order in both modes (deterministic merge).
    s1 = _json.dumps(ev1, sort_keys=True)
    s2 = _json.dumps(ev2, sort_keys=True)
    assert s1 == s2


def test_corrupt_parquet_stays_blocked(tmp_path):
    net = FakeNet(tmp_path)
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pqbad", [{"repo": f"org/Q@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    bad = b"not a parquet at all"
    net.add("bad.parquet", bad)
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024,
                    max_files_per_batch=100, max_batch_bytes=10 * GB)
    (b,) = store.plan_job_batches(jid, [("bad.parquet", len(bad))],
                                  budget=budget)
    tree = {"bad.parquet": (len(bad), net.sha("bad.parquet"))}
    rep = run_batch(store, jid, b.batch_id, caps=budget.stage_caps,
                    min_free_bytes=1024,
                    cfg=DownloadConfig(file_workers=1, retry_jitter=False),
                    download_fn=net, tree=tree)
    assert rep["ok"], rep
    rq = run_qc_batch(store, jid, b.batch_id, FakeAnalyzer())
    assert rq["n_shard_blocked"] == 1, rq
    assert rq["n_accepted"] == 0, rq
