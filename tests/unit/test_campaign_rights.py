"""Campaign QC must apply the job's operator-declared rights.

Regression: manifest_to_source_record left SourceRecord at its fail-closed
defaults (UNKNOWN / redistribute=False), so the required ``rights_ok`` check
rejected every row of a legitimately-licensed corpus and the whole campaign
produced 0 ACCEPT -> "no-accepted-rows" at release.
"""

import os
from types import SimpleNamespace

import pytest

from ornix_dataset.contracts.enums import DecisionState, RightsStatus
from ornix_dataset.campaign.processing import _job_rights_fields, manifest_to_source_record

SHA = "c" * 40
ROW = {"source_uri": "hf://datasets/org/A@" + SHA + "/a.wav",
       "original_file_id": "a.wav", "sha256": "9" * 64, "size": 10,
       "staged_path": "a.wav"}


def _job(rights):
    return SimpleNamespace(pinned_sha=SHA, rights=rights)


APPROVED = {"status": "REDISTRIBUTION_APPROVED", "redistribute": True,
            "commercial": "false", "license": "CC-BY-NC-SA-4.0"}


def test_rights_approved_maps_to_record():
    f = _job_rights_fields(_job(APPROVED))
    assert f["rights_status"] == RightsStatus.REDISTRIBUTION_APPROVED
    assert f["redistribution_permitted"] is True
    assert f["commercial_training_permitted"] == "false"
    assert f["source_license"] == "CC-BY-NC-SA-4.0"


def test_approved_status_without_flag_is_not_approval():
    f = _job_rights_fields(_job({"status": "REDISTRIBUTION_APPROVED",
                                 "redistribute": False}))
    assert f["redistribution_permitted"] is False


def test_train_only_keeps_status_but_blocks_redistribution():
    f = _job_rights_fields(_job({"status": "TRAIN_ONLY", "redistribute": True}))
    assert f["rights_status"] == RightsStatus.TRAIN_ONLY
    assert f["redistribution_permitted"] is False


def test_forbidden_and_missing_are_fail_closed():
    f = _job_rights_fields(_job({"status": "FORBIDDEN", "redistribute": True}))
    assert f["rights_status"] == RightsStatus.FORBIDDEN
    assert f["redistribution_permitted"] is False
    assert _job_rights_fields(_job({})) == {}
    assert _job_rights_fields(_job(None)) == {}
    bad = _job_rights_fields(_job({"status": "MADE-UP-STATUS",
                                  "redistribute": True}))
    assert bad["rights_status"] == RightsStatus.UNKNOWN
    assert bad["redistribution_permitted"] is False


def test_source_record_carries_rights():
    rec = manifest_to_source_record(ROW, _job(APPROVED), "/staging")
    assert rec.rights_status == RightsStatus.REDISTRIBUTION_APPROVED
    assert rec.redistribution_permitted is True
    assert rec.source_license == "CC-BY-NC-SA-4.0"
    # undeclared job keeps the fail-closed default
    rec2 = manifest_to_source_record(ROW, _job({}), "/staging")
    assert rec2.rights_status == RightsStatus.UNKNOWN
    assert rec2.redistribution_permitted is False


def test_rights_ok_passes_with_declaration_and_fails_without():
    """End-to-end through the real policy engine."""
    from ornix_dataset.contracts.quality import QualityEvidence
    from ornix_dataset.curation.policy import PolicyConfig, PolicyEngine
    from ornix_dataset.contracts.source import SourceRecord

    def _ev():
        return QualityEvidence(
            segment_id="S1", source_sha256="1" * 64, interval_start_sample=0,
            interval_end_sample=24000 * 5, analysis_sample_rate=24000,
            speech_ratio=0.9, clipping_ratio=0.0, noise_events=[])

    engine = PolicyEngine(PolicyConfig(
        policy_version="p", release_target="train_only",
        required_checks=["rights_ok"]))
    ok_rec = manifest_to_source_record(ROW, _job(APPROVED), "/staging")
    bad_rec = manifest_to_source_record(ROW, _job({}), "/staging")
    assert engine.decide(_ev(), ok_rec).decision == DecisionState.ACCEPT
    assert engine.decide(_ev(), bad_rec).decision == DecisionState.LICENSE_REVIEW
