"""Canonical normalization unit tests (T1-T15), fully offline with real WAVs."""

import json
import os
import re

import pytest

from fixtures import synth

from ornix_dataset.canonical import (
    CANONICAL_FIELDS,
    ORNIX_ID_RE,
    SampleInput,
    SchemaError,
    internal_sample_key,
    load_ornix_dataset,
    normalize_language,
    normalize_samples,
    open_state,
    read_all_rows,
    read_wav_int16,
    verify_canonical_dataset,
)
from ornix_dataset.canonical.layout import SPLITS, metadata_path
from ornix_dataset.canonical.schema import CanonicalRow
from ornix_dataset.util.hashing import sha256_file

WAV_RE = re.compile(r"^ornix_[0-9a-f]{32}\.wav$")


def make_sample(root, name="a.wav", *, text="xin chào thế giới", lang="vi",
                speaker_ref="speaker_01", scope="hf://datasets/org/A@rev1",
                sid="SRC_1", ssha="a" * 64, rights="REDISTRIBUTION_APPROVED",
                red=True, split="train", tverify=True, lverify=True,
                dur=2.0, seed=1, sr=24000, original=None, audio_sha=None, sub=""):
    src_dir = root / "src" / sub
    src_dir.mkdir(parents=True, exist_ok=True)
    wav = src_dir / name
    synth.write_wav(str(wav), synth.speechlike(sr, dur, seed=seed), sr)
    return SampleInput(
        wav_path=str(wav), text=text, language=lang, speaker_ref=speaker_ref,
        source_scope=scope, source_id=sid, source_sha256=ssha,
        seg_start=0, seg_end=int(sr * dur), split=split,
        rights_status=rights, redistribution_permitted=red,
        transcript_verified=tverify, language_verified=lverify,
        source_uri=scope, source_revision="rev1",
        original_file_id=original if original is not None else name,
        audio_sha256=audio_sha if audio_sha is not None else sha256_file(str(wav)))


def normalize(root, samples, **kw):
    return normalize_samples(samples, str(root / "ds"), str(root / "state"), **kw)


# T1 ---------------------------------------------------------------------------
def test_t1_metadata_exactly_six_fields_and_types(tmp_path):
    rep = normalize(tmp_path, [make_sample(tmp_path)])
    rows = load_ornix_dataset(str(tmp_path / "ds"))
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(CANONICAL_FIELDS)
    assert isinstance(row["duration"], float)
    for k in ("audio", "text", "file_name", "speaker", "language"):
        assert isinstance(row[k], str)
    # rejects any extra/absent field
    with pytest.raises(SchemaError):
        CanonicalRow.from_dict({**row, "source_license": "x"})
    with pytest.raises(SchemaError):
        CanonicalRow.from_dict({k: v for k, v in row.items() if k != "text"})


# T2 / T3 ----------------------------------------------------------------------
def test_t2_t3_output_names_are_opaque_ornix(tmp_path):
    s = make_sample(tmp_path, name="my_source_dataset_clip_001.wav", seed=3)
    normalize(tmp_path, [s])
    files = []
    for dp, _d, fs in os.walk(str(tmp_path / "ds")):
        files += [os.path.basename(f) for f in fs if f.endswith(".wav")]
    assert files and all(WAV_RE.match(f) for f in files)
    joined = " ".join(files)
    assert "my_source_dataset_clip_001" not in joined
    assert not any("A" == f for f in files)


# T4 ---------------------------------------------------------------------------
def test_t4_same_filename_two_datasets_no_collision(tmp_path):
    a = make_sample(tmp_path, name="a.wav", sub="A", scope="hf://datasets/org/A@r1",
                    sid="SRC_A", ssha="a" * 64, seed=1, original="a.wav")
    b = make_sample(tmp_path, name="a.wav", sub="B", scope="hf://datasets/org/B@r1",
                    sid="SRC_B", ssha="b" * 64, seed=2, original="a.wav")
    rep = normalize(tmp_path, [a, b])
    rows = load_ornix_dataset(str(tmp_path / "ds"))
    assert rep["n_rows"] == 2, rep["blocked"]
    assert len(rows) == 2
    assert rows[0]["audio"] != rows[1]["audio"]


# T5 / T6 ----------------------------------------------------------------------
def test_t5_retry_resume_keeps_same_id(tmp_path):
    s = make_sample(tmp_path, seed=5)
    r1 = normalize(tmp_path, [s])
    id1 = load_ornix_dataset(str(tmp_path / "ds"))[0]["audio"]
    r2 = normalize(tmp_path, [s])
    id2 = load_ornix_dataset(str(tmp_path / "ds"))[0]["audio"]
    assert id1 == id2 and len(load_ornix_dataset(str(tmp_path / "ds"))) == 1
    assert r2["n_reused_files"] == 1


def test_t6_out_of_order_no_duplicate_ids_or_rows(tmp_path):
    a = make_sample(tmp_path, name="a.wav", sid="S_A", ssha="a" * 64, seed=1)
    b = make_sample(tmp_path, name="b.wav", sid="S_B", ssha="b" * 64, seed=2)
    normalize(tmp_path, [a, b])
    ids = {r["audio"] for r in load_ornix_dataset(str(tmp_path / "ds"))}
    normalize(tmp_path, [b, a])  # reversed order
    rows = load_ornix_dataset(str(tmp_path / "ds"))
    assert {r["audio"] for r in rows} == ids and len(rows) == 2


# T7 ---------------------------------------------------------------------------
def test_t7_speaker_mapping_stable_and_no_false_merge(tmp_path):
    a1 = make_sample(tmp_path, name="a1.wav", scope="org/A@r1", sid="S1",
                     ssha="1" * 64, seed=1)
    a2 = make_sample(tmp_path, name="a2.wav", scope="org/A@r1", sid="S2",
                     ssha="2" * 64, seed=2)
    b1 = make_sample(tmp_path, name="b1.wav", scope="org/B@r1", sid="S3",
                     ssha="3" * 64, seed=3)
    normalize(tmp_path, [a1, a2, b1])
    with open_state(str(tmp_path / "state")) as st:
        spk_by_key = {k: v["speaker_id"] for k, v in st.data["speakers"].items()}
    a_key = next(k for k in spk_by_key if "org/A@r1" in k)
    b_key = next(k for k in spk_by_key if "org/B@r1" in k)
    # same dataset+ref -> one stable speaker; different dataset same ref -> distinct
    assert len(spk_by_key) == 2
    assert spk_by_key[a_key] != spk_by_key[b_key]
    rows = load_ornix_dataset(str(tmp_path / "ds"))
    speakers = {r["speaker"] for r in rows}
    assert speakers == {spk_by_key[a_key], spk_by_key[b_key]}


def test_t7_unknown_speaker_never_merged(tmp_path):
    u1 = make_sample(tmp_path, name="u1.wav", speaker_ref=None, sid="U1",
                     ssha="1" * 64, seed=1)
    u2 = make_sample(tmp_path, name="u2.wav", speaker_ref=None, sid="U2",
                     ssha="2" * 64, seed=2)
    normalize(tmp_path, [u1, u2])
    speakers = {r["speaker"] for r in load_ornix_dataset(str(tmp_path / "ds"))}
    assert len(speakers) == 2


# T8 ---------------------------------------------------------------------------
def test_t8_duration_matches_verified_wav(tmp_path):
    s = make_sample(tmp_path, dur=3.0, seed=8)
    normalize(tmp_path, [s])
    row = load_ornix_dataset(str(tmp_path / "ds"))[0]
    ds = tmp_path / "ds"
    samples, sr = read_wav_int16(str(ds / "train" / row["audio"]))
    assert abs(row["duration"] - samples.size / sr) < 1e-6
    assert row["duration"] == 3.0


# T9 / T15 ---------------------------------------------------------------------
def test_t9_unverified_transcript_excluded(tmp_path):
    bad = make_sample(tmp_path, name="bad.wav", text="", tverify=False, ssha="a" * 64)
    good = make_sample(tmp_path, name="good.wav", sid="S2", ssha="b" * 64, seed=2)
    rep = normalize(tmp_path, [bad, good])
    rows = load_ornix_dataset(str(tmp_path / "ds"))
    assert len(rows) == 1 and rows[0]["text"]
    assert any("TRANSCRIPT_UNVERIFIED" in b["reason"] for b in rep["blocked"])


def test_t9_language_never_blind_default(tmp_path):
    no_lang = make_sample(tmp_path, name="nl.wav", lang=None, lverify=False,
                          ssha="a" * 64)
    rep = normalize(tmp_path, [no_lang])
    assert rep["n_rows"] == 0
    assert any("LANGUAGE_UNVERIFIED" in b["reason"] for b in rep["blocked"])


def test_t15_rights_gate_not_bypassed_by_rename(tmp_path):
    forbidden = make_sample(tmp_path, name="x.wav", rights="LICENSE_REVIEW",
                            red=False, ssha="a" * 64)
    train_only = make_sample(tmp_path, name="y.wav", rights="TRAIN_ONLY",
                             red=False, sid="S2", ssha="b" * 64, seed=2)
    rep = normalize(tmp_path, [forbidden, train_only])  # public target
    assert rep["n_rows"] == 0
    assert {b["reason"].split(":")[0] for b in rep["blocked"]} == \
        {"RIGHT_NOT_PERMITTED", "REDISTRIBUTION_FLAG_FALSE"} or \
        any("PERMITTED" in b["reason"] or "REDISTRIBUTION_FLAG_FALSE" in b["reason"]
            for b in rep["blocked"])


def test_train_only_allowed_when_opted_in(tmp_path):
    s = make_sample(tmp_path, rights="TRAIN_ONLY", red=False, ssha="a" * 64)
    rep = normalize(tmp_path, [s], require_redistributable=False)
    assert rep["n_rows"] == 1


# T10 / T11 --------------------------------------------------------------------
def test_t10_t11_paths_present_no_orphan_dangling_duplicate(tmp_path):
    samples = [make_sample(tmp_path, name=f"c{i}.wav", sid=f"S{i}",
                           ssha=f"{i}" * 64, seed=i) for i in range(1, 4)]
    normalize(tmp_path, samples)
    res = verify_canonical_dataset(str(tmp_path / "ds"))
    assert res.ok, res.errors
    for r in load_ornix_dataset(str(tmp_path / "ds")):
        assert r["audio"] == r["file_name"]
        assert os.path.exists(str(tmp_path / "ds" / "train" / r["audio"]))


def test_orphan_and_dangling_detected(tmp_path):
    normalize(tmp_path, [make_sample(tmp_path)])
    ds = tmp_path / "ds"
    # orphan: a stray wav with no metadata row
    stray = ds / "train" / "audio" / "zz" / f"ornix_{'f' * 32}.wav"
    stray.parent.mkdir(parents=True, exist_ok=True)
    synth.write_wav(str(stray), synth.speechlike(24000, 1.0, seed=9), 24000)
    res = verify_canonical_dataset(str(ds))
    assert not res.ok and any("ORPHAN_AUDIO" in e for e in res.errors)
    stray.unlink()
    # dangling: metadata points at a missing file
    row = load_ornix_dataset(str(ds))[0]
    os.remove(str(ds / "train" / row["audio"]))
    res2 = verify_canonical_dataset(str(ds))
    assert not res2.ok and any("DANGLING_METADATA" in e for e in res2.errors)


# T12 --------------------------------------------------------------------------
def test_t12_split_preserved_and_leakage_safe(tmp_path):
    a1 = make_sample(tmp_path, name="a1.wav", scope="org/A@r1", sid="S1",
                     ssha="1" * 64, split="train", seed=1)
    a2 = make_sample(tmp_path, name="a2.wav", scope="org/A@r1", sid="S2",
                     ssha="2" * 64, split="train", seed=2)
    b1 = make_sample(tmp_path, name="b1.wav", scope="org/B@r1", sid="S3",
                     ssha="3" * 64, split="validation", seed=3)
    normalize(tmp_path, [a1, a2, b1])
    rows = read_all_rows(str(tmp_path / "ds"))
    assert len(rows["train"]) == 2 and len(rows["validation"]) == 1
    # every speaker lives in exactly one split
    seen = {}
    for split in SPLITS:
        for r in rows[split].values():
            seen.setdefault(r.speaker, set()).add(split)
    assert all(len(v) == 1 for v in seen.values())
    assert verify_canonical_dataset(str(tmp_path / "ds")).ok


# T13 --------------------------------------------------------------------------
def test_t13_jsonl_unicode_and_float_duration(tmp_path):
    s = make_sample(tmp_path, text="Xin chào, tôi tên là Ornix.")
    normalize(tmp_path, [s])
    raw = open(metadata_path(str(tmp_path / "ds"), "train"), "rb").read()
    assert "Xin chào, tôi tên là Ornix.".encode("utf-8") in raw
    line = raw.splitlines()[0].decode("utf-8")
    assert '"duration":' in line
    assert isinstance(json.loads(line)["duration"], float)


# T14 --------------------------------------------------------------------------
def test_t14_provenance_preserved_internally_and_not_public(tmp_path):
    s = make_sample(tmp_path, name="secret_source_name.wav", original="secret_source_name.wav")
    normalize(tmp_path, [s])
    row = load_ornix_dataset(str(tmp_path / "ds"))[0]
    assert set(row) == set(CANONICAL_FIELDS)
    assert "secret_source_name" not in json.dumps(row)
    with open_state(str(tmp_path / "state")) as st:
        assert len(st.data["provenance"]) == 1
        prov = next(iter(st.data["provenance"].values()))
        assert prov["source_id"] == "SRC_1"
        assert prov["original_file_id"] == "secret_source_name.wav"
        assert prov["source_sha256"] == "a" * 64
        assert prov["rights_status"] == "REDISTRIBUTION_APPROVED"
        assert prov["audio_sha256"] == sha256_file(s.wav_path)


def test_normalize_language():
    assert normalize_language("vi-VN") == "vi"
    assert normalize_language("en_US") == "en"
    assert normalize_language("") is None and normalize_language(None) is None


def test_internal_key_is_stable_and_name_free():
    k = internal_sample_key("SRC_1", "a" * 64, 0, 48000, "rev1")
    assert internal_sample_key("SRC_1", "a" * 64, 0, 48000, "rev1") == k
    assert "a.wav" not in k
    assert k != internal_sample_key("SRC_1", "a" * 64, 0, 48001, "rev1")


def test_unknown_split_rejected_fail_closed(tmp_path):
    bad = make_sample(tmp_path, name="dev.wav", split="dev", ssha="a" * 64)
    rep = normalize(tmp_path, [bad])
    assert rep["n_rows"] == 0
    assert rep["blocked"][0]["reason"] == "UNKNOWN_SPLIT:dev"
    assert not os.path.exists(str(tmp_path / "ds" / "dev"))


def test_resume_recreates_missing_file_without_new_id(tmp_path):
    # identity is committed before bytes: dropping the output must not allocate a
    # second id on the next run (crash/resume safety)
    s = make_sample(tmp_path, seed=11)
    normalize(tmp_path, [s])
    row = load_ornix_dataset(str(tmp_path / "ds"))[0]
    wav = tmp_path / "ds" / "train" / row["audio"]
    os.remove(str(wav))
    normalize(tmp_path, [s])
    rows = load_ornix_dataset(str(tmp_path / "ds"))
    assert len(rows) == 1 and rows[0]["audio"] == row["audio"]
    assert os.path.exists(str(tmp_path / "ds" / "train" / rows[0]["audio"]))
    assert verify_canonical_dataset(str(tmp_path / "ds")).ok
