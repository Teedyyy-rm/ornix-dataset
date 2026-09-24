"""CLI end-to-end (spec §7): ingest -> qc -> release build -> verify -> publish
dry-run, driven through the packaged entry point. Fully offline."""

import json
import os

import pytest

from ornix_dataset.cli import main


def _run(argv, capsys):
    code = main(argv)
    out = capsys.readouterr().out
    payload = json.loads(out) if out.strip() else {}
    return code, payload


def test_cli_full_flow(tmp_path, sources_config, capsys):
    workdir = str(tmp_path / "work")
    run_id = "cli1"

    code, _ = _run(["sources", "validate", "--config", sources_config, "--dry-run"], capsys)
    assert code == 0

    code, ing = _run(["ingest", "--config", sources_config, "--run-id", run_id,
                      "--workdir", workdir], capsys)
    assert code == 0 and ing["ingested"] == 5

    code, qc = _run(["qc", "--run-id", run_id, "--policy",
                     "configs/quality_policy.pilot.yaml", "--workdir", workdir], capsys)
    assert code == 0 and qc["accepted"] == 4

    rel_out = str(tmp_path / "rel")
    code, rb = _run(["release", "build", "--run-id", run_id, "--release-id", "cli-rel-1",
                     "--out", rel_out, "--workdir", workdir], capsys)
    assert code == 0 and rb["ready"] is True

    code, ver = _run(["release", "verify", "--release-dir", rel_out], capsys)
    assert code == 0 and ver["ok"] is True and "release_digest" in ver

    code, pub = _run(["publish", "--release-dir", rel_out, "--repo-id", "org/ornix-vi"], capsys)
    assert code == 0 and pub["status"] == "DRY_RUN"
    assert not os.path.exists(os.path.join(rel_out, "PUBLISHED_VERIFIED.json"))


def test_cli_ingest_idempotent(tmp_path, sources_config, capsys):
    workdir = str(tmp_path / "work")
    _run(["ingest", "--config", sources_config, "--run-id", "r", "--workdir", workdir], capsys)
    code, second = _run(["ingest", "--config", sources_config, "--run-id", "r",
                         "--workdir", workdir], capsys)
    assert code == 0 and second["ingested"] == 0


def test_cli_qc_requires_ingest(tmp_path, capsys):
    workdir = str(tmp_path / "empty")
    code, payload = _run(["qc", "--run-id", "nope", "--policy",
                          "configs/quality_policy.pilot.yaml", "--workdir", workdir], capsys)
    assert code == 2 and "error" in payload
