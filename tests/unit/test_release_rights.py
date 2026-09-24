"""Rights-safe release contract (spec §0.2.6, §6.3): TRAIN_ONLY never publishes."""

import pytest

from ornix_dataset.contracts.release import ReleaseRow
from ornix_dataset.exporters.manifest import build_release


def _row(**over):
    base = dict(
        audio_id="ornix_vi_abc123", audio="audio/SEG_x.wav", language="vi",
        speaker_id="spk0", transcript="xin chao", sample_rate=24000, channels=1,
        encoding="PCM_S16LE", duration_s=3.0, source_id="SRC_1",
        source_sha256="a" * 64, audio_sha256="b" * 64,
        segment_start_sample_source=0, segment_end_sample_source=72000,
        rights_record_id="RIGHTS_1", quality_evidence_id="SEG_x",
        quality_policy_version="v1", quality_gate="ACCEPT", split="train",
        release_id="rel1",
    )
    base.update(over)
    return ReleaseRow(**base)


def test_train_only_row_validates_for_train_target():
    row = _row(rights_status="TRAIN_ONLY", redistribution_permitted=False,
               source_license="CC-BY-NC-4.0")
    row.validate(release_target="train_only")  # ok


def test_train_only_row_rejected_for_public_target():
    row = _row(rights_status="TRAIN_ONLY", redistribution_permitted=False)
    with pytest.raises(ValueError):
        row.validate(release_target="public")


def test_approved_row_validates_for_public_target():
    row = _row(rights_status="REDISTRIBUTION_APPROVED", redistribution_permitted=True,
               source_license="CC-BY-4.0")
    row.validate(release_target="public")  # ok


def test_approved_status_without_permission_flag_still_blocks_public():
    row = _row(rights_status="REDISTRIBUTION_APPROVED", redistribution_permitted=False)
    with pytest.raises(ValueError):
        row.validate(release_target="public")


def test_build_release_public_blocks_train_only(tmp_path):
    import os
    from fixtures import synth
    from ornix_dataset.util.hashing import sha256_file

    src_dir = tmp_path / "canonical"
    src_dir.mkdir()
    wav = src_dir / "SEG_x.wav"
    synth.write_wav(str(wav), synth.speechlike(24000, 2.0, seed=1), 24000)
    sha = sha256_file(str(wav))
    row = _row(audio="audio/SEG_x.wav", audio_sha256=sha, source_sha256="a" * 64,
               rights_status="TRAIN_ONLY", redistribution_permitted=False).to_dict()

    out = tmp_path / "rel_public"
    art = build_release("rel1", [row], str(src_dir), str(out),
                        release_target="public", export_format="parquet")
    assert not art.ready
    assert any("ROW_INVALID" in b or "NON_REDISTRIBUTABLE" in b for b in art.blockers)
    assert not os.path.exists(os.path.join(str(out), "RELEASE_READY.json"))


def test_build_release_train_only_target_allows_train_only(tmp_path):
    import os
    from fixtures import synth
    from ornix_dataset.util.hashing import sha256_file

    src_dir = tmp_path / "canonical"
    src_dir.mkdir()
    wav = src_dir / "SEG_x.wav"
    synth.write_wav(str(wav), synth.speechlike(24000, 2.0, seed=1), 24000)
    sha = sha256_file(str(wav))
    row = _row(audio="audio/SEG_x.wav", audio_sha256=sha, source_sha256="a" * 64,
               rights_status="TRAIN_ONLY", redistribution_permitted=False).to_dict()

    out = tmp_path / "rel_train"
    art = build_release("rel1", [row], str(src_dir), str(out),
                        release_target="train_only", export_format="parquet")
    assert art.ready, art.blockers
    assert os.path.exists(os.path.join(str(out), "RELEASE_READY.json"))
