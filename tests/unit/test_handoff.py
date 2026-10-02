"""Ornix-TTS handoff manifest tests (spec: stable inter-project contract).

Offline, CPU-only. Covers the happy path, determinism, the exact schema, and
every fail-closed validation rule; also asserts the two existing JSONL
contracts (``metadata.jsonl``, ``release_manifest.jsonl``) are untouched.
"""

import json
import os

import pytest

from fixtures import synth

from ornix_dataset.canonical import (
    HANDOFF_FIELDS,
    HANDOFF_NAME,
    HandoffError,
    SampleInput,
    normalize_samples,
    write_handoff_manifest,
)
from ornix_dataset.canonical.layout import SPLITS, read_split_rows
from ornix_dataset.canonical.schema import CANONICAL_FIELDS
from ornix_dataset.util.hashing import sha256_file


def _sample(root, name="a.wav", *, text="xin chào thế giới", lang="vi",
            speaker_ref="speaker_01", scope="hf://datasets/org/A@rev1",
            sid="SRC_1", ssha="a" * 64, rights="REDISTRIBUTION_APPROVED",
            red=True, split="train", license_id="CC-BY-4.0", dur=2.0, seed=1,
            sr=24000, sub=""):
    src_dir = root / "src" / sub
    src_dir.mkdir(parents=True, exist_ok=True)
    wav = src_dir / name
    synth.write_wav(str(wav), synth.speechlike(sr, dur, seed=seed), sr)
    return SampleInput(
        wav_path=str(wav), text=text, language=lang, speaker_ref=speaker_ref,
        source_scope=scope, source_id=sid, source_sha256=ssha,
        seg_start=0, seg_end=int(sr * dur), split=split,
        rights_status=rights, redistribution_permitted=red,
        transcript_verified=True, language_verified=True,
        source_uri=scope, source_revision="rev1", original_file_id=name,
        audio_sha256=sha256_file(str(wav)), source_license=license_id)


def _export(root, samples, **kw):
    return normalize_samples(samples, str(root / "ds"), str(root / "state"), **kw)


def _handoff_path(root):
    return os.path.join(str(root / "ds"), HANDOFF_NAME)


def _read(root):
    with open(_handoff_path(root), "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- happy path & schema ------------------------------------------------------

def test_happy_path_creates_manifest(tmp_path):
    rep = _export(tmp_path, [_sample(tmp_path)])
    assert rep["handoff"]["written"] is True
    assert rep["handoff"]["n_rows"] == 1
    assert os.path.isfile(_handoff_path(tmp_path))
    rows = _read(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(HANDOFF_FIELDS)
    assert row["transcript"] == "xin chào thế giới"
    assert row["language"] == "vi"
    assert row["split"] == "train"
    assert row["license"] == "CC-BY-4.0"
    assert row["consent"] == "redistribution-approved"
    assert row["speaker_id"].startswith("spk_")


def test_audio_path_relative_to_release_root_and_exists(tmp_path):
    _export(tmp_path, [_sample(tmp_path)])
    row = _read(tmp_path)[0]
    assert row["audio"].startswith("train/audio/")
    assert not row["audio"].startswith("/")
    assert ".." not in row["audio"]
    assert os.path.isfile(os.path.join(str(tmp_path / "ds"), *row["audio"].split("/")))


def test_id_is_stable_ornix_identity(tmp_path):
    _export(tmp_path, [_sample(tmp_path)])
    row = _read(tmp_path)[0]
    # the handoff id is the stable Ornix identity, never the filename
    from ornix_dataset.canonical.schema import ORNIX_ID_RE

    assert ORNIX_ID_RE.match(row["id"])
    assert row["id"] in row["audio"]


def test_unique_ids_across_many_rows(tmp_path):
    samples = [_sample(tmp_path, name=f"a{i}.wav", sid=f"SRC_{i}",
                       ssha=str(i) * 64, seed=i) for i in range(5)]
    _export(tmp_path, samples)
    ids = [r["id"] for r in _read(tmp_path)]
    assert len(ids) == len(set(ids)) == 5


def test_multiple_splits(tmp_path):
    samples = [
        _sample(tmp_path, name="tr.wav", split="train", sid="S1", ssha="1" * 64,
                speaker_ref="spk_tr", seed=1),
        _sample(tmp_path, name="va.wav", split="validation", sid="S2", ssha="2" * 64,
                speaker_ref="spk_va", seed=2),
        _sample(tmp_path, name="te.wav", split="test", sid="S3", ssha="3" * 64,
                speaker_ref="spk_te", seed=3),
    ]
    _export(tmp_path, samples)
    rows = _read(tmp_path)
    assert {r["split"] for r in rows} == set(SPLITS)
    for r in rows:
        assert r["audio"].startswith(r["split"] + "/")


# --- determinism --------------------------------------------------------------

def test_output_is_byte_identical_on_reexport(tmp_path):
    _export(tmp_path, [_sample(tmp_path, name=f"a{i}.wav", sid=f"S{i}",
                               ssha=str(i) * 64, seed=i) for i in range(4)])
    first = open(_handoff_path(tmp_path), "rb").read()
    _export(tmp_path, [_sample(tmp_path, name=f"a{i}.wav", sid=f"S{i}",
                               ssha=str(i) * 64, seed=i) for i in range(4)])
    second = open(_handoff_path(tmp_path), "rb").read()
    assert first == second


def test_stable_ordering_sorted_by_split_then_id(tmp_path):
    samples = [
        _sample(tmp_path, name="te.wav", split="test", sid="S3", ssha="3" * 64,
                speaker_ref="spk_te", seed=3),
        _sample(tmp_path, name="tr.wav", split="train", sid="S1", ssha="1" * 64,
                speaker_ref="spk_tr", seed=1),
        _sample(tmp_path, name="va.wav", split="validation", sid="S2", ssha="2" * 64,
                speaker_ref="spk_va", seed=2),
    ]
    _export(tmp_path, samples)
    rows = _read(tmp_path)
    keys = [(r["split"], r["id"]) for r in rows]
    assert keys == sorted(keys)


def test_reexport_does_not_duplicate_rows(tmp_path):
    s = _sample(tmp_path)
    _export(tmp_path, [s])
    n1 = len(_read(tmp_path))
    _export(tmp_path, [s])
    assert len(_read(tmp_path)) == n1 == 1


# --- internal contracts untouched ---------------------------------------------

def test_metadata_jsonl_schema_unchanged(tmp_path):
    _export(tmp_path, [_sample(tmp_path)])
    rows = read_split_rows(str(tmp_path / "ds"), "train")
    assert len(rows) == 1
    for row in rows.values():
        assert set(row.to_dict()) == set(CANONICAL_FIELDS)


def test_release_manifest_jsonl_not_written_by_handoff(tmp_path):
    # the handoff exporter writes only manifest.jsonl; it never creates or
    # rewrites release_manifest.jsonl
    _export(tmp_path, [_sample(tmp_path)])
    assert not os.path.exists(os.path.join(str(tmp_path / "ds"),
                                           "release_manifest.jsonl"))


# --- validation (fail-closed) -------------------------------------------------

def test_unknown_license_rejected(tmp_path):
    _export(tmp_path, [_sample(tmp_path, license_id="UNKNOWN")])
    rep = _export(tmp_path, [_sample(tmp_path, license_id="UNKNOWN",
                                      name="b.wav", sid="S2", ssha="2" * 64)])
    assert rep["handoff"]["written"] is False
    assert "license" in rep["handoff"]["reason"]


@pytest.mark.parametrize("bad", ["UNKNOWN", "UNLICENSED", "NONE", "NULL", ""])
def test_blank_or_unknown_license_sentinels_rejected(tmp_path, bad):
    rep = _export(tmp_path, [_sample(tmp_path, license_id=bad)])
    assert rep["handoff"]["written"] is False


def test_repo_operator_declared_sentinel_rejected(tmp_path):
    # exactly the sentinel configs/ornix_campaign*.yaml uses for a dataset with no
    # Hub license that the operator merely asserts redistribution rights for
    rep = _export(tmp_path, [_sample(tmp_path,
                                     license_id="UNSPECIFIED-OPERATOR-DECLARED")])
    assert rep["handoff"]["written"] is False
    assert "license is unknown" in rep["handoff"]["reason"]


def test_train_only_rights_block_handoff(tmp_path):
    rep = _export(tmp_path, [_sample(tmp_path, rights="TRAIN_ONLY", red=False)],
                  require_redistributable=False)
    assert rep["handoff"]["written"] is False
    assert "rights" in rep["handoff"]["reason"]


def test_missing_audio_file_rejected(tmp_path):
    _export(tmp_path, [_sample(tmp_path)])
    # remove the copied audio; re-export must now fail closed
    row = _read(tmp_path)[0]
    os.remove(os.path.join(str(tmp_path / "ds"), *row["audio"].split("/")))
    with pytest.raises(HandoffError):
        write_handoff_manifest(str(tmp_path / "ds"), str(tmp_path / "state"))


def test_speaker_across_splits_rejected(tmp_path):
    # same speaker ref in two splits -> speaker straddles splits
    _export(tmp_path, [
        _sample(tmp_path, name="tr.wav", split="train", sid="S1", ssha="1" * 64,
                speaker_ref="spk_same", seed=1),
        _sample(tmp_path, name="va.wav", split="validation", sid="S2", ssha="2" * 64,
                speaker_ref="spk_same", seed=2),
    ], require_redistributable=True)
    with pytest.raises(HandoffError):
        write_handoff_manifest(str(tmp_path / "ds"), str(tmp_path / "state"))


# --- row-level validators -----------------------------------------------------

def _valid_row(**over):
    row = {
        "id": "Ornix_0000001",
        "audio": "train/audio/01/Ornix_0000001.wav",
        "transcript": "xin chào",
        "language": "vi",
        "speaker_id": "spk_" + "a" * 32,
        "split": "train",
        "license": "CC-BY-4.0",
        "consent": "redistribution-approved",
    }
    row.update(over)
    return row


@pytest.mark.parametrize("bad_split", ["dev", "TRAIN", "", "holdout"])
def test_invalid_split_rejected(bad_split):
    from ornix_dataset.canonical.handoff import validate_handoff_row

    with pytest.raises(HandoffError):
        validate_handoff_row(_valid_row(split=bad_split))


@pytest.mark.parametrize("bad_path", [
    "/abs/train/audio/01/x.wav",
    "../train/audio/01/x.wav",
    "train/../train/audio/01/x.wav",
    "./train/audio/01/x.wav",
    "https://x/train/audio/01/x.wav",
    "train\\audio\\x.wav",
    "elsewhere/audio/x.wav",
])
def test_unsafe_audio_path_rejected(bad_path):
    from ornix_dataset.canonical.handoff import validate_handoff_row

    with pytest.raises(HandoffError):
        validate_handoff_row(_valid_row(audio=bad_path))


@pytest.mark.parametrize("field", ["transcript", "language", "speaker_id", "id"])
def test_blank_required_field_rejected(field):
    from ornix_dataset.canonical.handoff import validate_handoff_row

    with pytest.raises(HandoffError):
        validate_handoff_row(_valid_row(**{field: ""}))


@pytest.mark.parametrize("bad", ["UNKNOWN", "UNLICENSED", "NONE", "NULL", ""])
def test_unknown_license_row_rejected(bad):
    from ornix_dataset.canonical.handoff import validate_handoff_row

    with pytest.raises(HandoffError):
        validate_handoff_row(_valid_row(license=bad))


@pytest.mark.parametrize("bad", ["UNKNOWN", "UNSPECIFIED", "NONE", ""])
def test_unknown_consent_row_rejected(bad):
    from ornix_dataset.canonical.handoff import validate_handoff_row

    with pytest.raises(HandoffError):
        validate_handoff_row(_valid_row(consent=bad))


def test_missing_audio_file_row_rejected(tmp_path):
    from ornix_dataset.canonical.handoff import validate_handoff_row

    with pytest.raises(HandoffError):
        validate_handoff_row(_valid_row(), dataset_dir=str(tmp_path))


# --- atomicity ----------------------------------------------------------------

def test_no_temp_file_left_behind(tmp_path):
    _export(tmp_path, [_sample(tmp_path)])
    leftovers = [f for f in os.listdir(str(tmp_path / "ds")) if ".tmp-" in f]
    assert leftovers == []