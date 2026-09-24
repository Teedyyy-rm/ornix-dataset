"""MD-002 Gate 2 — RESOURCE_PLAN_VERIFIED (all offline)."""

import json
import os

import pytest

from ornix_dataset.campaign import (
    Batch,
    Budget,
    CampaignStore,
    ReservationLedger,
    StageProfile,
    WatermarkGate,
    plan_with_budget,
    try_admit,
)
from ornix_dataset.campaign.planning import plan_batches

SHA = "c" * 40
GB = 1024**3


def small_budget(**kw):
    args = {"workspace_max_bytes": 10 * GB, "min_free_bytes": 1 * GB,
            "max_files_per_batch": 100, "max_batch_bytes": 2 * GB}
    args.update(kw)
    return Budget(**args)


def files(n, size):
    return [(f"f/{i:05d}.wav", size) for i in range(n)]


# -- Gate 2 core: synthetic dataset >> disk ------------------------------------

def test_synthetic_larger_than_disk_splits_into_valid_batches():
    # 5 TB of files, 10 GB workspace: every batch must fit the budget
    plan = plan_with_budget("org/Big", SHA, None, files(5000, 1 * GB),
                            small_budget())
    assert len(plan.batches) == 2500  # 2 x 1GB files per batch
    for b in plan.batches:
        assert b.status == "PLANNED"
        assert sum(b.reservation[s] for s in ("cache", "staging", "qc", "upload")) \
            <= 10 * GB
        assert len(b.files) <= 100
    assert plan.total_known_bytes == 5000 * GB


def test_streaming_admit_release_never_exceeds_caps():
    budget = small_budget()
    plan = plan_with_budget("org/Big", SHA, None, files(200, 1 * GB), budget)
    ledger = ReservationLedger(budget.stage_caps)
    peak = {s: 0 for s in ("cache", "staging", "qc", "upload")}
    # worker window of 2 concurrent batches: admit, finish, release, continue
    in_flight: list[Batch] = []
    for b in plan.batches:
        assert try_admit(b, ledger, free_bytes=50 * GB,
                         min_free_bytes=budget.min_free_bytes)
        in_flight.append(b)
        for s in peak:
            peak[s] = max(peak[s], ledger.used[s])
        if len(in_flight) == 2:
            assert ledger.release(in_flight.pop(0).batch_id)
    for s in peak:
        assert peak[s] <= budget.stage_caps[s]


def test_no_reservation_no_start():
    budget = small_budget()
    (b,) = plan_with_budget("org/A", SHA, None, [("a.wav", 1 * GB)],
                            budget).batches
    ledger = ReservationLedger({"cache": 1, "staging": 1, "qc": 1, "upload": 1})
    assert ledger.acquire(b) is False
    assert try_admit(b, ledger, 50 * GB, budget.min_free_bytes) is False
    assert ledger.totals()["held_batches"] == []
    # starved free space also refuses, even with roomy caps
    roomy = ReservationLedger(budget.stage_caps)
    assert try_admit(b, roomy, free_bytes=1 * GB,
                     min_free_bytes=budget.min_free_bytes) is False
    assert roomy.totals()["held_batches"] == []


def test_blocked_and_in_progress_batches_never_admitted():
    budget = small_budget()
    (b,) = plan_with_budget("org/A", SHA, None, [("a.wav", 100)],
                            budget).batches
    ledger = ReservationLedger(budget.stage_caps)
    b.status = "BLOCKED"
    assert try_admit(b, ledger, 50 * GB, budget.min_free_bytes) is False
    b.status = "IN_PROGRESS"
    assert ledger.acquire(b) is False
    # no double-acquire either
    b.status = "PLANNED"
    assert ledger.acquire(b) is True
    assert ledger.acquire(b) is False
    assert ledger.release("nope") is False


# -- unknown sizes / oversize ---------------------------------------------------

def test_unknown_sizes_capped_and_estimated():
    prof = StageProfile(unknown_estimate_bytes=100, max_unknown_per_batch=3)
    plan = plan_with_budget("org/U", SHA, None,
                            [(f"f/{i}.wav", None) for i in range(7)],
                            small_budget(), prof)
    assert [len(b.files) for b in plan.batches] == [3, 3, 1]
    assert all(b.reservation["accountable_bytes"] == 300 for b in plan.batches[:2])
    assert plan.total_unknown_files == 7


def test_lone_oversize_file_blocked_with_reason():
    plan = plan_with_budget("org/O", SHA, None,
                            [("ok.wav", 10), ("huge.wav", 100 * GB)],
                            small_budget())
    assert len(plan.batches) == 2
    huge = [b for b in plan.batches if b.files == ["huge.wav"]][0]
    assert huge.status == "BLOCKED"
    assert huge.result["oversize"] is True
    assert "streaming" in huge.result["blocker"]


# -- reproducibility + MD-001 continuity ----------------------------------------

def test_plan_reproducible_bit_for_bit():
    kw = {"files": files(37, 12345), "budget": small_budget()}
    p1 = plan_with_budget("org/R", SHA, None, **kw)
    p2 = plan_with_budget("org/R", SHA, None, **kw)
    assert p1.plan_hash == p2.plan_hash
    assert [b.batch_id for b in p1.batches] == [b.batch_id for b in p2.batches]
    assert [b.reservation for b in p1.batches] == [b.reservation for b in p2.batches]


def test_generous_budget_matches_mechanical_split_ids():
    f = files(9, 1000)
    mech = plan_batches("org/M", SHA, None, f, max_files=4,
                        max_bytes=10 * GB)
    budgeted = plan_with_budget(
        "org/M", SHA, None, f,
        Budget(workspace_max_bytes=100 * GB, min_free_bytes=0,
               max_files_per_batch=4, max_batch_bytes=10 * GB),
        StageProfile(qc_factor=1.0))
    assert [b.batch_id for b in budgeted.batches] == [b.batch_id for b in mech]


# -- watermarks ------------------------------------------------------------------

def test_watermark_hysteresis_no_flapping():
    gate = WatermarkGate(high_watermark_bytes=10 * GB, low_watermark_bytes=20 * GB)
    assert gate.evaluate(50 * GB) == "run"
    assert gate.evaluate(9 * GB) == "stop"
    assert gate.evaluate(15 * GB) == "stop"   # still stopped below LOW
    assert gate.evaluate(20 * GB) == "run"
    with pytest.raises(ValueError):
        WatermarkGate(high_watermark_bytes=20 * GB, low_watermark_bytes=10 * GB)


# -- budget construction ----------------------------------------------------------

def test_budget_from_real_disk_and_fail_closed(tmp_path):
    b = Budget.from_workspace({"min_free_bytes": 1024}, str(tmp_path))
    assert b.workspace_max_bytes > 0 and b.min_free_bytes == 1024
    with pytest.raises(ValueError):
        Budget.from_workspace({"min_free_bytes": 10**30}, str(tmp_path))
    with pytest.raises(ValueError):
        Budget(workspace_max_bytes=0, min_free_bytes=0)
    with pytest.raises(ValueError):
        Budget(workspace_max_bytes=10, min_free_bytes=0,
               high_watermark_bytes=9, low_watermark_bytes=5)


# -- store + CLI integration -------------------------------------------------------

def test_store_budgeted_plan_persists_reservations(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign(
        "demo", [{"repo": f"org/A@{SHA}"}],
        workspace={"min_free_bytes": 1024},
        resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    budget = Budget.from_workspace({"min_free_bytes": 1024}, str(tmp_path),
                                   max_files_per_batch=2)
    b1 = store.plan_job_batches(jid, files(3, 1000), budget=budget)
    assert all(b.reservation.get("accountable_bytes", 0) > 0 for b in b1)
    reloaded = [store.load_batch(b.batch_id) for b in b1]
    assert [b.reservation for b in reloaded] == [b.reservation for b in b1]
    b2 = store.plan_job_batches(jid, files(3, 1000), budget=budget)
    assert [b.batch_id for b in b2] == [b.batch_id for b in b1]


def test_cli_campaign_plan_offline(tmp_path, capsys):
    from ornix_dataset.cli import main

    inp = tmp_path / "camp.yaml"
    inp.write_text(f"name: p2\ndatasets:\n  - repo: org/A@{SHA}\n",
                   encoding="utf-8")
    root = str(tmp_path / "store")
    assert main(["campaign", "create", "--input", str(inp), "--root", root]) == 0
    jid = json.loads(capsys.readouterr().out)["outcomes"][0]["job_id"]
    inv = tmp_path / "inv.jsonl"
    inv.write_text("\n".join(
        json.dumps({"path": f"f/{i}.wav", "size": 1000}) for i in range(5)),
        encoding="utf-8")
    assert main(["campaign", "plan", "--root", root, "--job", jid,
                 "--inventory", str(inv), "--max-files", "2",
                 "--min-free", "1024"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["n_batches"] == 3 and rep["n_blocked"] == 0
    assert main(["campaign", "status", "--root", root]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["n_batches"] == 3
    # empty inventory fails closed
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(["campaign", "plan", "--root", root, "--job", jid,
                 "--inventory", str(empty)]) == 2
