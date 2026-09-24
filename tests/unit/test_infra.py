"""Permission gate + contracts + render + ops tests."""

import os

import numpy as np
import pytest

from fixtures import synth
from ornix_dataset.contracts.enums import RightsStatus
from ornix_dataset.contracts.release import ReleaseRow
from ornix_dataset.ingestion.permissions import PermissionGate
from ornix_dataset.dsp import decode_to_float, render_canonical_wav
from ornix_dataset.ops import Checkpoint, ContentCache, cache_key, takedown_plan


def test_t012_unknown_rights_quarantine():
    gate = PermissionGate({})
    d = gate.evaluate("S1", "file:///x/y.wav")
    assert d.rights_status == RightsStatus.LICENSE_REVIEW
    assert not d.redistribution_permitted


def test_redistribution_flag_without_approval_downgraded():
    gate = PermissionGate({"S1": {"rights_status": "TRAIN_ONLY",
                                  "redistribution_permitted": True}})
    d = gate.evaluate("S1", "file:///x")
    assert d.rights_status == RightsStatus.LICENSE_REVIEW


def test_approved_source_redistributable():
    gate = PermissionGate({"S1": {"rights_status": "REDISTRIBUTION_APPROVED",
                                  "redistribution_permitted": True}})
    d = gate.evaluate("S1", "file:///x")
    assert d.rights_status == RightsStatus.REDISTRIBUTION_APPROVED and d.redistribution_permitted


def test_license_from_declaration_propagates():
    # A per-source rights declaration carries the license so the published card /
    # RELEASE_READY records real attribution terms instead of UNKNOWN.
    gate = PermissionGate({"S1": {"rights_status": "REDISTRIBUTION_APPROVED",
                                  "redistribution_permitted": True,
                                  "source_license": "CC-BY-NC-SA-4.0"}})
    assert gate.evaluate("S1", "file:///x").source_license == "CC-BY-NC-SA-4.0"


def test_license_per_file_overrides_declaration():
    gate = PermissionGate({"S1": {"rights_status": "REDISTRIBUTION_APPROVED",
                                  "redistribution_permitted": True,
                                  "source_license": "CC-BY-NC-SA-4.0"}})
    d = gate.evaluate("S1", "file:///x", declared_license="CC-BY-4.0")
    assert d.source_license == "CC-BY-4.0"


def test_license_unknown_without_declaration():
    assert PermissionGate({}).evaluate("S1", "file:///x").source_license == "UNKNOWN"


def test_release_row_validate_rejects_absolute_path():
    row = ReleaseRow(audio_id="a", audio="/abs/x.wav", language="vi", speaker_id="s",
                     transcript="t", sample_rate=24000, channels=1, encoding="PCM_S16LE",
                     duration_s=2.0, source_id="S", source_sha256="a" * 64,
                     audio_sha256="b" * 64, segment_start_sample_source=0,
                     segment_end_sample_source=1, rights_record_id="R",
                     quality_evidence_id="E", quality_policy_version="v1",
                     quality_gate="ACCEPT", split="train", release_id="rel")
    with pytest.raises(ValueError):
        row.validate()


def test_render_canonical_roundtrip(tmp_path):
    p = str(tmp_path / "in.wav")
    synth.write_wav(p, synth.speechlike(44100, 2.0), 44100)
    buf, _ = decode_to_float(p)
    out = str(tmp_path / "out.wav")
    sha, recipe = render_canonical_wav(buf, out)
    buf2, _ = decode_to_float(out)
    assert buf2.sample_rate == 24000 and buf2.channels == 1
    assert abs(buf2.duration_s - 2.0) < 0.02
    assert recipe.resample_method == "scipy-resample_poly"
    assert len(sha) == 64


def test_cache_key_invalidates_on_weight_change():
    k1 = cache_key("s", {"vad": "w1"}, "cfg", "p1")
    k2 = cache_key("s", {"vad": "w2"}, "cfg", "p1")
    assert k1 != k2


def test_content_cache_roundtrip(tmp_path):
    c = ContentCache(str(tmp_path / "cache"))
    c.put("k", {"v": 1})
    assert c.get("k") == {"v": 1}
    c.invalidate("k")
    assert c.get("k") is None


def test_checkpoint_idempotent_replay(tmp_path):
    cp = Checkpoint(str(tmp_path / "cp.jsonl"), "run1")
    cp.mark("item1", "ACCEPTED")
    cp.mark("item2", "REJECTED")
    prog = cp.load()
    assert cp.is_done("item1", prog) and cp.is_done("item2", prog)


def test_takedown_plan_non_destructive():
    plan = takedown_plan(["S1"], "rel-1", "rel-2")
    assert plan["action"] == "QUARANTINE_AND_REISSUE"
    assert "preserved" in plan["invariant"]
