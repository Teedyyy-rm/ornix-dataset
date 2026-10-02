"""Fan-out release chunking tests (spec: incremental publish).

Pure, offline: only the deterministic split and the approval binding are
exercised here; the release build itself is covered by the release tests.
"""

import os

import pytest

from ornix_dataset.campaign.fanout import DEFAULT_CHUNK, chunk_accepted, write_approvals


def _row(i, split="train"):
    return {"audio_id": f"ornix_vi_{i:012d}", "split": split,
            "source_sha256": f"{i:064d}", "source_license": "MIT",
            "rights_status": "REDISTRIBUTION_APPROVED", "source_id": "SRC_1",
            "audio": f"audio/SEG_{i}.wav"}


def test_default_chunk_is_100():
    assert DEFAULT_CHUNK == 100


def test_chunk_splits_at_size():
    rows = [_row(i) for i in range(250)]
    chunks = chunk_accepted(rows, 100)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_chunk_is_deterministic():
    rows = [_row(i) for i in range(250)]
    a = [[r["audio_id"] for r in c] for c in chunk_accepted(rows, 100)]
    b = [[r["audio_id"] for r in c] for c in chunk_accepted(list(reversed(rows)), 100)]
    assert a == b


def test_chunk_order_is_split_then_id():
    rows = [_row(2, "validation"), _row(1, "train"), _row(3, "train")]
    flat = [r["audio_id"] for c in chunk_accepted(rows, 10) for r in c]
    assert flat == sorted(flat, key=lambda x: (
        "train" if x.endswith("000000000001") or x.endswith("000000000003") else "validation",
        x))


def test_no_rows_no_chunks():
    assert chunk_accepted([], 100) == []


def test_invalid_chunk_size_rejected():
    with pytest.raises(ValueError):
        chunk_accepted([_row(1)], 0)


def test_rows_are_not_duplicated_or_dropped():
    rows = [_row(i) for i in range(1000)]
    flat = [r["audio_id"] for c in chunk_accepted(rows, 100) for r in c]
    assert len(flat) == len(set(flat)) == 1000


def test_approvals_only_for_ready_and_bound_to_digest(tmp_path):
    releases = [
        {"release_id": "r0", "ready": True, "digest": "d" * 64, "n_rows": 100},
        {"release_id": "r1", "ready": False, "blockers": ["X"]},
        {"release_id": "r2", "ready": True, "digest": "e" * 64, "n_rows": 7},
    ]
    out = write_approvals(
        releases, repo_id="org/ds", revision_of=lambda rid: "rev-" + rid,
        operator_id="op", policy_version="pol-v1",
        expires_utc="2099-01-01T00:00:00Z", max_bytes=1024,
        approval_dir=str(tmp_path))
    assert [a["release_id"] for a in out] == ["r0", "r2"]
    import yaml
    rec = yaml.safe_load(open(out[0]["approval"]))
    assert rec["release_digest"] == "d" * 64
    assert rec["repo_id"] == "org/ds"
    assert rec["revision"] == "rev-r0"
    assert rec["license_ack"] is True
    # a blocked chunk never gets an approval file
    assert not os.path.exists(str(tmp_path / "r1.approval.yaml"))