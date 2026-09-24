"""G3 destination preflight + G4 report/dry-run (all offline)."""

import json
import os

from ornix_dataset.campaign import (
    Budget,
    CampaignStore,
    campaign_preflight,
    campaign_report,
    check_destination,
    dry_run_plan,
    expected_upload_bytes,
    oversized_files,
    prepare_release,
    repo_current_bytes,
)
from ornix_dataset.testing.fakes import FakeHub
from test_publish_campaign import ready_batch

SHA = "q" * 40
GB = 1024**3


class TreeHub(FakeHub):
    """FakeHub + realistic list_repo_tree sizes."""

    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes  # path -> bytes

    def list_repo_tree(self, repo_id, revision=None, repo_type="dataset",
                       recursive=True):
        from types import SimpleNamespace
        return [SimpleNamespace(type="file", path=p, size=s, lfs={})
                for p, s in self.sizes.items()]


# -- G3: destination preflight ----------------------------------------------------

def test_repo_current_bytes_sums_files():
    hub = TreeHub({"a.bin": 100, "sub/b.bin": 250})
    assert repo_current_bytes(hub, "org/d") == {"bytes": 350, "n_files": 2}


def test_check_destination_quota_exceeded_fails_closed():
    hub = TreeHub({"existing.bin": 90 * GB})
    pf = check_destination(hub, {"repo_id": "org/d", "max_bytes": 100 * GB},
                           projected_bytes=20 * GB)
    assert pf.ok is False
    assert any("DESTINATION_QUOTA_EXCEEDED" in r for r in pf.reasons)
    assert pf.current_bytes == 90 * GB and pf.projected_bytes == 110 * GB


def test_check_destination_within_quota_ok():
    hub = TreeHub({"existing.bin": 10 * GB})
    pf = check_destination(hub, {"repo_id": "org/d", "max_bytes": 100 * GB,
                                 "redistribution_confirmed": True},
                           projected_bytes=20 * GB)
    assert pf.ok is True and not pf.reasons


def test_check_destination_unreachable_fails_closed():
    hub = TreeHub({})
    hub.missing_repos.add("org/nope")
    pf = check_destination(hub, {"repo_id": "org/nope"})
    assert pf.ok is False and any("DESTINATION_UNREACHABLE" in r for r in pf.reasons)


def test_check_destination_missing_quota_warns_not_passes():
    hub = TreeHub({"x": 1})
    pf = check_destination(hub, {"repo_id": "org/d"})
    assert pf.ok is True  # may proceed; not silently
    assert any("QUOTA_NOT_DECLARED" in w for w in pf.warnings)
    assert any("REDISTRIBUTION_NOT_CONFIRMED" in w for w in pf.warnings)


def test_check_destination_missing_repo_id():
    pf = check_destination(FakeHub(), {})
    assert pf.ok is False and pf.reasons == ["NO_DESTINATION_REPO"]


def test_campaign_preflight_uses_declared_quota(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign(
        "pf", [{"repo": f"org/A@{SHA}"}],
        destination={"repo_id": "org/d", "max_bytes": 10 * GB,
                     "redistribution_confirmed": True},
        resolver=lambda r, v: SHA)
    jid = out[0]["job_id"]
    budget = Budget(workspace_max_bytes=100 * GB, min_free_bytes=1024)
    store.plan_job_batches(jid, [("a.wav", 5 * GB)], budget=budget)
    hub = TreeHub({"existing.bin": 9 * GB})
    rep = campaign_preflight(store, api=hub)
    assert rep["ok"] is False
    assert any("DESTINATION_QUOTA_EXCEEDED" in r for r in rep["reasons"])
    assert rep["projected_bytes"] == expected_upload_bytes(store)


def test_oversized_files_flags_above_hard_limit(tmp_path):
    d = tmp_path / "rel"
    d.mkdir()
    (d / "ok.bin").write_bytes(b"x" * 10)
    big = d / "big.bin"
    with open(big, "wb") as fh:
        fh.truncate(2 * GB)
    # tiny limit stands in for the 500 GB hard cap (can't allocate 500 GB)
    bad = oversized_files(str(d), limit=GB)
    assert [b["path"] for b in bad] == ["big.bin"]


def test_prepare_release_blocks_oversized_via_hard_limit(tmp_path, monkeypatch):
    store, jid, batch = ready_batch(tmp_path)
    import ornix_dataset.campaign.publishing as P
    monkeypatch.setattr(P, "oversized_files",
                        lambda d, limit=None: [{"path": "audio/a.wav",
                                                "bytes": 600 * GB}])
    rep = prepare_release(store, jid, batch.batch_id, export_format="none")
    assert rep["ok"] is False
    assert any(b.startswith("HUB_FILE_HARD_LIMIT") for b in rep["blockers"])


# -- G4: report + dry-run ---------------------------------------------------------

def test_campaign_report_aggregates_state(tmp_path):
    store, jid, batch = ready_batch(tmp_path)
    rep = campaign_report(store)
    assert rep["campaign_id"] == "pub"
    assert rep["n_jobs"] == 1 and rep["n_batches"] == 1
    assert rep["by_job_status"] == {"BATCHED": 1}
    assert rep["by_batch_status"].get("RELEASE_READY") == 1
    assert rep["jobs"][0]["batches"][0]["next_action"] == "prepare-release"
    assert "generated_utc" in rep


def test_campaign_report_unpublished_release_not_verified(tmp_path):
    store, jid, batch = ready_batch(tmp_path)
    prepare_release(store, jid, batch.batch_id, export_format="none")
    rep = campaign_report(store)
    assert rep["n_releases"] == 1
    assert rep["releases"][0]["published_verified"] is False


def test_dry_run_plan_is_read_only(tmp_path):
    store, jid, batch = ready_batch(tmp_path)
    before = store.load_batch(batch.batch_id).to_dict()
    plan = dry_run_plan(store)
    assert plan["dry_run"] is True and plan["n_actions"] == 1
    assert plan["actions"][0]["next_action"] == "prepare-release"
    after = store.load_batch(batch.batch_id).to_dict()
    assert before == after  # nothing mutated


def test_cli_preflight_and_report(tmp_path, capsys):
    from ornix_dataset.cli import main

    root = str(tmp_path / "store")
    assert main(["campaign", "preflight", "--root", root]) == 2
    assert main(["campaign", "report", "--root", root]) == 2


def test_cli_report_writes_artifact_and_dry_run(tmp_path, capsys):
    from ornix_dataset.cli import main

    store, jid, batch = ready_batch(tmp_path)
    out = tmp_path / "report.json"
    assert main(["campaign", "report", "--root", store.root,
                 "--out", str(out)]) == 0
    capsys.readouterr()
    data = json.load(open(out, encoding="utf-8"))
    assert data["n_batches"] == 1
    assert main(["campaign", "report", "--root", store.root,
                 "--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["dry_run"] is True
