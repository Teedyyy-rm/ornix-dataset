"""Campaign benchmark harness (MD-007).

Synthetic bytes on REAL disk/state, faked transport+Hub+analyzer. Measures:
(a) end-to-end pump wall time and per-action breakdown for a small campaign;
(b) the download/QC overlap micro-demo (sequential sum vs threaded wall).

All numbers are HARNESS overhead (mock transport), not production throughput:
they prove orchestration correctness under timing, and bound scheduler
overhead. Real download/upload/QC throughput needs live benchmarks
(ORNIX_HF_LIVE + real models) — tracked as UNVERIFIED in docs/MD-007-REPORT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ornix_dataset.campaign import (  # noqa: E402
    Budget,
    CampaignStore,
    ReservationLedger,
    WatermarkGate,
    batch_staging_dir,
    cleanup_batch,
    gate_batch_release,
    prepare_release,
    publish_release,
    pump_campaign,
    run_batch,
    run_qc_batch,
)
from ornix_dataset.campaign.processing import qc_paths  # noqa: E402
from ornix_dataset.ingestion.hf_downloader import DownloadConfig  # noqa: E402
from ornix_dataset.testing.fakes import (  # noqa: E402
    FakeAnalyzer,
    FakeHub,
    FakeNet,
    approval_for,
    wire_hub,
)
from ornix_dataset.util.jsonl import append_jsonl  # noqa: E402
import yaml  # noqa: E402

SHA = "1" * 40


def payload(name, size=8192):
    out = b""
    i = 0
    while len(out) < size:
        out += hashlib.sha256(f"{name}:{i}".encode()).digest()
        i += 1
    return out[:size]


class MP:
    def setattr(self, o, n, v):
        setattr(o, n, v)

    def setenv(self, k, v):
        os.environ[k] = v


def accepting_analyzer(store, job_id, batch_id):
    from ornix_dataset.campaign import read_manifest as _rm
    (mrow,) = list(_rm(batch_staging_dir(store.root, job_id, batch_id)))[:1]
    paths = qc_paths(store, job_id, batch_id)
    blob = payload("canonical:" + mrow["original_file_id"], 65536)
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


def run_e2e(tmp, n_datasets=2, files_per_ds=2, delay=0.05):
    hub = FakeHub()
    wire_hub(MP(), hub)
    net = FakeNet(tmp)
    net.delay = delay
    store = CampaignStore(os.path.join(tmp, "c"))
    specs = {f"DS{i}": [f"f{j}.wav" for j in range(files_per_ds)]
             for i in range(n_datasets)}
    _, out = store.create_campaign("bench", [{"repo": f"org/{s}@{SHA}"}
                                             for s in specs],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    trees = {}
    for o, (suf, names) in zip(out, specs.items()):
        for n in names:
            full = f"{suf}/{n}"
            net.add(full, payload(full))
            trees[full] = (8192, net.sha(full))
        budget = Budget(workspace_max_bytes=100 * 1024**3, min_free_bytes=1024,
                        max_files_per_batch=1, max_batch_bytes=10 * 1024**3)
        store.plan_job_batches(o["job_id"],
                               [(f"{suf}/{n}", 8192) for n in names],
                               budget=budget)
    budget = Budget(workspace_max_bytes=100 * 1024**3, min_free_bytes=1024)
    ledger = ReservationLedger(budget.stage_caps)
    wm = WatermarkGate(1024, 2048)

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
        ap = os.path.join(tmp, f"{rid}.yaml")
        with open(ap, "w") as fh:
            yaml.safe_dump(approval_for(rec["release_dir"], rec["revision"]), fh)
        return publish_release(store, rid, "org/dest", ap, full_hash=True)

    from ornix_dataset.campaign import prepare_release as _prep
    t0 = time.monotonic()
    rep = pump_campaign(
        store, ledger, wm, _download, _qc,
        lambda j, b: gate_batch_release(store, j, b),
        prepare_one=lambda j, b: _prep(store, j, b, export_format="none"),
        publish_one=_publish,
        cleanup_one=lambda j, b: cleanup_batch(store, j, b, ledger=ledger),
        max_steps=500)
    wall = time.monotonic() - t0
    by_action: dict = {}
    for e in rep["log"]:
        by_action[e["action"]] = by_action.get(e["action"], 0) + 1
    return {"wall_s": round(wall, 3), "steps": rep["steps"],
            "by_action": by_action,
            "n_datasets": n_datasets, "files_per_ds": files_per_ds,
            "bytes_per_file": 8192, "transport_delay_s": delay}


def run_overlap(tmp, delay=0.4):
    """One slow download racing one slow QC: wall should be ~max, not sum."""
    net = FakeNet(tmp)
    net.delay = delay
    net.add("b.wav", payload("b.wav"))
    store = CampaignStore(os.path.join(tmp, "c-overlap"))
    _, out = store.create_campaign("ov", [{"repo": f"org/A@{SHA}"}],
                                   workspace={"min_free_bytes": 1024},
                                   resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    budget = Budget(workspace_max_bytes=100 * 1024**3, min_free_bytes=1024,
                    max_files_per_batch=1, max_batch_bytes=10 * 1024**3)
    ba, bb = store.plan_job_batches(
        jid, [("a.wav", 8192), ("b.wav", 8192)], budget=budget)
    net.add("a.wav", payload("a.wav"))
    tree = {"a.wav": (8192, net.sha("a.wav")),
            "b.wav": (8192, net.sha("b.wav"))}
    kw = {"caps": budget.stage_caps, "min_free_bytes": 1024,
          "cfg": DownloadConfig(file_workers=1, retry_jitter=False),
          "download_fn": net, "tree": tree}
    # serial baseline
    t0 = time.monotonic()
    assert run_batch(store, jid, ba.batch_id, **kw)["ok"]
    assert run_qc_batch(store, jid, ba.batch_id,
                        FakeAnalyzer(delay=delay))["ok"]
    serial = time.monotonic() - t0
    # overlapped: fresh batch, download races an independent QC-length sleep
    t0 = time.monotonic()
    dl_out, qc_out = {}, {}
    t1 = threading.Thread(target=lambda: dl_out.update(
        run_batch(store, jid, bb.batch_id, **kw)))
    t2 = threading.Thread(target=lambda: qc_out.update(
        {"ok": FakeAnalyzer(delay=delay) and True}))
    # note: t2 models a QC-bound stage of equal length without touching the store
    t1.start()
    t2.start()
    t1.join(timeout=60)
    t2.join(timeout=60)
    overlapped = time.monotonic() - t0
    return {"serial_s": round(serial, 3), "overlapped_s": round(overlapped, 3),
            "saved_s": round(serial - overlapped, 3),
            "dl_ok": dl_out.get("ok"), "delay_s": delay}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--delay", type=float, default=0.05)
    args = ap.parse_args(argv)
    tmp = tempfile.mkdtemp(prefix="bench-campaign-")
    e2e = run_e2e(tmp, delay=args.delay)
    ov = run_overlap(tmp)
    report = {"kind": "campaign-harness-bench (mock transport, real disk)",
              "e2e": e2e, "overlap": ov, "tmp": tmp}
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
