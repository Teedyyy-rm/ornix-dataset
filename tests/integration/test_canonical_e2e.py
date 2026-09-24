"""Real-WAV integration for the canonical dataset (T5/T6/T16/T17/T18).

Runs the actual local pipeline and a real campaign batch (real 24 kHz WAVs, no
mock audio), then normalizes, verifies, loads and publishes through FakeHub.
"""

import json
import os

import pytest
import yaml

from fixtures import synth

from ornix_dataset.audit.events import AuditLog
from ornix_dataset.canonical import (
    CANONICAL_FIELDS,
    export_campaign_batch,
    export_run,
    load_ornix_dataset,
    read_wav_int16,
    sample_from_accepted_row,
    split_stats,
    verify_canonical_dataset,
)
from ornix_dataset.canonical.publish import (
    finalize_dataset,
    publish_dataset,
    receipt_path_for,
)
from ornix_dataset.campaign import Budget, CampaignStore
from ornix_dataset.campaign.processing import qc_paths
from ornix_dataset.config import build_gate_and_adapters, load_policy_config
from ornix_dataset.pipeline import OrnixPipeline, RunPaths
from ornix_dataset.publishing.hf import marker_path_for
from ornix_dataset.testing.fakes import FakeHub, approval_for, wire_hub
from ornix_dataset.util.hashing import sha256_file

SHA = "f" * 40
GB = 1024**3


def _run_local_pipeline(tmp_path, sources_config):
    _gate, adapters, _ = build_gate_and_adapters(sources_config)
    pipe = OrnixPipeline(str(tmp_path / "work"), load_policy_config(
        "configs/quality_policy.pilot.yaml"))
    paths = RunPaths.create(pipe.workdir, "r1")
    audit = AuditLog(paths.audit)
    combined = pipe.run(adapters[0], paths, audit)
    return paths, combined


# T16 (and T8/T13 on real pipeline output) ------------------------------------
def test_real_pipeline_to_canonical_and_loader(tmp_path, sources_config):
    paths, combined = _run_local_pipeline(tmp_path, sources_config)
    assert len(combined.accepted) == 4
    ds = tmp_path / "Ornix-Datasets"
    rep = export_run(str(tmp_path / "work"), "r1", str(ds),
                     str(tmp_path / "state"), require_redistributable=False)
    assert rep["n_rows"] == 4, rep["blocked"]

    v = verify_canonical_dataset(str(ds))
    assert v.ok, v.errors

    rows = load_ornix_dataset(str(ds))
    assert len(rows) == 4
    for r in rows:
        assert set(r) == set(CANONICAL_FIELDS)
        assert r["audio"] == r["file_name"]
        samples, sr = read_wav_int16(str(ds / "train" / r["audio"]))
        assert sr == 24000 and samples.size > 0
        assert abs(r["duration"] - samples.size / sr) < 1e-6
    texts = sorted(r["text"] for r in rows)
    assert texts == ["cau noi 0 0", "cau noi 0 1", "cau noi 1 0", "cau noi 1 1"]
    # two source speakers, never merged; all rows in one split set
    assert len({r["speaker"] for r in rows}) == 2

    # re-export is idempotent (T5 on the real flow)
    before = {r["audio"] for r in rows}
    export_run(str(tmp_path / "work"), "r1", str(ds), str(tmp_path / "state"),
               require_redistributable=False)
    assert {r["audio"] for r in load_ornix_dataset(str(ds))} == before


def _campaign_batch(store, jid, name, audio, text, speaker_ref, split,
                    seed_sha):
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024)
    (b,) = store.plan_job_batches(jid, [(name, len(audio))], budget=budget)
    batch = store.load_batch(b.batch_id)
    p = qc_paths(store, jid, b.batch_id)
    with open(os.path.join(p.canonical_dir, name), "wb") as fh:
        fh.write(audio)
    row = {"audio_id": name, "audio": f"audio/{name}", "language": "vi",
           "speaker_id": speaker_ref, "transcript": text,
           "sample_rate": 24000, "channels": 1, "encoding": "PCM16",
           "duration_s": 2.0, "source_id": "SRC_" + name,
           "source_sha256": seed_sha * 64, "audio_sha256": sha256_file(
               os.path.join(p.canonical_dir, name)),
           "segment_start_sample_source": 0,
           "segment_end_sample_source": 48000, "rights_record_id": "rr",
           "quality_evidence_id": "qe_" + name, "quality_policy_version": "v1",
           "quality_gate": "ACCEPT", "split": split, "release_id": "R",
           "rights_status": "REDISTRIBUTION_APPROVED",
           "redistribution_permitted": True,
           "source_uri": f"hf://datasets/org/A@{SHA}/{name}",
           "source_revision": SHA, "original_file_id": name,
           "source_speaker_ref": speaker_ref, "source_license": "CC-BY-4.0",
           "language_verified": True, "transcript_verified": True}
    with open(os.path.join(p.root, "BATCH_ACCEPTED.jsonl"), "w") as fh:
        fh.write(json.dumps(row) + "\n")
    return batch


def _real_wav_bytes(tmp_path, name, seed):
    d = tmp_path / "wavsrc"
    d.mkdir(exist_ok=True)
    p = d / name
    synth.write_wav(str(p), synth.speechlike(24000, 2.0, seed=seed), 24000)
    with open(p, "rb") as fh:
        return fh.read()


# T17 (incremental) + T16 (real WAV via campaign bridge) ----------------------
def test_incremental_campaign_export_keeps_old_rows(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pub", [{"repo": f"org/A@{SHA}"}],
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    ds = tmp_path / "Ornix-Datasets"
    state = tmp_path / "state"

    b1 = _campaign_batch(store, jid, "one.wav",
                         _real_wav_bytes(tmp_path, "one.wav", 1),
                         "câu một", "spk-A", "train", "1")
    r1 = export_campaign_batch(store, jid, b1.batch_id, str(ds), str(state),
                               require_redistributable=True)
    assert r1["n_rows"] == 1 and len(load_ornix_dataset(str(ds))) == 1
    first_id = load_ornix_dataset(str(ds))[0]["audio"]

    b2 = _campaign_batch(store, jid, "two.wav",
                         _real_wav_bytes(tmp_path, "two.wav", 2),
                         "câu hai", "spk-B", "validation", "2")
    r2 = export_campaign_batch(store, jid, b2.batch_id, str(ds), str(state),
                               require_redistributable=True)
    assert r2["n_rows"] == 2  # dataset total after the incremental export
    assert r2["merged"]["validation"]["added"] == 1
    rows = load_ornix_dataset(str(ds))
    assert len(rows) == 2  # old row not lost
    assert first_id in {r["audio"] for r in rows}
    assert split_stats(str(ds))["by_split"] == {"train": 1, "validation": 1,
                                                "test": 0}
    assert verify_canonical_dataset(str(ds)).ok


def test_sidecar_supplies_transcript_and_speaker(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pub", [{"repo": f"org/A@{SHA}"}],
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    b = _campaign_batch(store, jid, "x.wav",
                        _real_wav_bytes(tmp_path, "x.wav", 3),
                        "", "spk-X", "train", "3")
    # strip the verified flags -> fail closed without the sidecar
    p = qc_paths(store, jid, b.batch_id)
    rows = [json.loads(l) for l in open(os.path.join(p.root, "BATCH_ACCEPTED.jsonl"))]
    rows[0]["transcript_verified"] = False
    rows[0]["language_verified"] = False
    rows[0]["transcript"] = ""
    with open(os.path.join(p.root, "BATCH_ACCEPTED.jsonl"), "w") as fh:
        fh.write(json.dumps(rows[0]) + "\n")
    ds, state = tmp_path / "ds", tmp_path / "st"
    blocked = export_campaign_batch(store, jid, b.batch_id, str(ds), str(state))
    assert blocked["n_rows"] == 0
    side = tmp_path / "side.jsonl"
    side.write_text(json.dumps({"original_file_id": "x.wav", "text": "nhập từ sidecar",
                                "speaker_ref": "spk-X", "language": "vi-VN",
                                "transcript_verified": True,
                                "language_verified": True}) + "\n", encoding="utf-8")
    rep = export_campaign_batch(store, jid, b.batch_id, str(ds), str(state),
                                metadata_file=str(side))
    assert rep["n_rows"] == 1
    row = load_ornix_dataset(str(ds))[0]
    assert row["text"] == "nhập từ sidecar" and row["language"] == "vi"


# T18 (publish + remote verify failure blocks cleanup) ------------------------
def test_publish_dataset_and_remote_verify_failure(tmp_path, monkeypatch):
    hub = FakeHub()
    wire_hub(monkeypatch, hub)
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pub", [{"repo": f"org/A@{SHA}"}],
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    ds, state = tmp_path / "Ornix-Datasets", tmp_path / "state"
    b = _campaign_batch(store, jid, "ok.wav",
                        _real_wav_bytes(tmp_path, "ok.wav", 4),
                        "kiểm thử", "spk-K", "train", "4")
    export_campaign_batch(store, jid, b.batch_id, str(ds), str(state))
    finalize_dataset(str(ds))

    ap = tmp_path / "appr.yaml"
    ap.write_text(yaml.safe_dump(approval_for(str(ds), "main", repo_id="org/dest")),
                  encoding="utf-8")
    pub = publish_dataset(str(ds), "org/dest", str(ap), full_hash=True,
                          staging_revision="main")
    assert pub["ok"] and pub["status"] == "PUBLISHED_VERIFIED"
    assert os.path.exists(marker_path_for(str(ds)))
    assert os.path.exists(receipt_path_for(str(ds)))
    assert any(k.startswith("train/") for k in hub.remote)
    assert any(k.endswith("README.md") for k in hub.remote)

    # partial upload -> REMOTE_VERIFY_FAILED, no marker, tree intact
    class _DropOne(FakeHub):
        def upload_folder(self, folder_path, repo_id, repo_type="dataset",
                          revision=None, path_in_repo=None, commit_message=""):
            super().upload_folder(folder_path, repo_id, repo_type, revision,
                                  path_in_repo, commit_message)
            victim = next(k for k in self.remote if k.endswith("metadata.jsonl"))
            del self.remote[victim]

    hub2 = _DropOne()
    wire_hub(monkeypatch, hub2)
    ds2 = tmp_path / "ds2"
    export_campaign_batch(store, jid, b.batch_id, str(ds2), str(state))
    finalize_dataset(str(ds2))
    ap2 = tmp_path / "appr2.yaml"
    ap2.write_text(yaml.safe_dump(approval_for(str(ds2), "main", repo_id="org/dest2")),
                   encoding="utf-8")
    bad = publish_dataset(str(ds2), "org/dest2", str(ap2), staging_revision="main")
    assert bad["ok"] is False and bad["status"] == "REMOTE_VERIFY_FAILED"
    assert not os.path.exists(marker_path_for(str(ds2)))
    assert verify_canonical_dataset(str(ds2)).ok  # local tree untouched


def test_public_gate_blocks_train_only_campaign_row(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("pub", [{"repo": f"org/A@{SHA}"}],
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    b = _campaign_batch(store, jid, "t.wav",
                        _real_wav_bytes(tmp_path, "t.wav", 5),
                        "huấn luyện", "spk-T", "train", "5")
    p = qc_paths(store, jid, b.batch_id)
    row = json.loads(open(os.path.join(p.root, "BATCH_ACCEPTED.jsonl")).readline())
    row["rights_status"] = "TRAIN_ONLY"
    row["redistribution_permitted"] = False
    open(os.path.join(p.root, "BATCH_ACCEPTED.jsonl"), "w").write(
        json.dumps(row) + "\n")
    rep = export_campaign_batch(store, jid, b.batch_id, str(tmp_path / "ds"),
                                str(tmp_path / "st"),
                                require_redistributable=True)
    assert rep["n_rows"] == 0
    assert rep["blocked"][0]["reason"].startswith("RIGHT_NOT_PERMITTED")


def test_canonical_cli_export_verify_load_publish(tmp_path, sources_config, capsys):
    from ornix_dataset.cli import main

    _run_local_pipeline(tmp_path, sources_config)
    ds, state = str(tmp_path / "Ornix-Datasets"), str(tmp_path / "state")
    work = str(tmp_path / "work")

    code = main(["canonical", "export", "--run-id", "r1", "--workdir", work,
                 "--dataset", ds, "--state", state, "--allow-train-only"])
    assert code == 0
    capsys.readouterr()

    assert main(["canonical", "verify", "--dataset", ds]) == 0
    v = json.loads(capsys.readouterr().out)
    assert v["ok"] and v["n_rows"] == 4

    assert main(["canonical", "load", "--dataset", ds]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["n_rows"] == 4 and set(loaded["sample"]) == set(CANONICAL_FIELDS)

    assert main(["canonical", "publish", "--dataset", ds, "--repo-id", "org/x",
                 "--approval", str(tmp_path / "nope.yaml")]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["status"] == "DRY_RUN"

    assert main(["canonical", "export-batch", "--root", str(tmp_path / "nope"),
                 "--job", "j", "--batch", "b", "--dataset", ds,
                 "--state", state]) == 2
