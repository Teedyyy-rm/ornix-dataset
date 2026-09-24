"""Bridges from verified pipeline/campaign records to canonical samples.

The canonical layer consumes ``SampleInput`` (verified WAV + transcript +
language + speaker evidence + rights). This module maps the existing contracts
onto it without inventing data:

- ``sample_from_accepted_row``: a local-run or campaign accepted release row.
- ``export_run``: all accepted rows of a local run.
- ``export_campaign_batch``: the accepted rows of one campaign batch, optionally
  enriched by a JSONL sidecar keyed by ``original_file_id`` (the campaign
  downloader does not itself carry source transcript/speaker/language).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..pipeline import RunPaths
from ..util.jsonl import load_jsonl, read_jsonl
from .normalize import SampleInput, normalize_samples


def scope_of(uri: str, revision: str) -> str:
    """Dataset-level speaker scope: strip the per-file part of a source URI.

    Speakers only merge within the same source *dataset*+revision, never across
    two datasets (spec §5). ``hf://datasets/repo@rev/path`` ->
    ``hf://datasets/repo@rev``; ``file:///root/file.wav`` -> ``file:///root``.
    """
    if not uri:
        return f"rev:{revision}"
    if uri.startswith("hf://datasets/"):
        head = uri[len("hf://datasets/"):].split("/", 1)[0]
        return (f"hf://datasets/{head}" if "@" in head
                else f"hf://datasets/{head}@{revision}")
    if uri.startswith("file://"):
        return "file://" + os.path.dirname(uri[len("file://"):])
    return uri


def _sidecar_index(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(path):
        key = rec.get("original_file_id") or rec.get("file") or rec.get("path")
        if key:
            out[key] = rec
    return out


def sample_from_accepted_row(row: Dict[str, Any], canonical_dir: str,
                             sidecar: Optional[Dict[str, Any]] = None
                             ) -> SampleInput:
    """Map one verified accepted release row to a ``SampleInput``.

    Provenance fields are taken from the row when present; transcript/speaker/
    language are only trusted when their verified flags are set (a sidecar may
    supply them for the campaign path).
    """
    sidecar = sidecar or {}
    original = row.get("original_file_id") or sidecar.get("original_file_id", "")
    audio = os.path.basename(row.get("audio", ""))
    wav_path = os.path.join(canonical_dir, audio)
    return SampleInput(
        wav_path=wav_path,
        text=sidecar.get("text", row.get("transcript", "")),
        language=sidecar.get("language", row.get("language")),
        speaker_ref=sidecar.get("speaker_ref", row.get("source_speaker_ref")),
        source_scope=scope_of(row.get("source_uri", ""),
                              row.get("source_revision", "")),
        source_id=row.get("source_id", ""),
        source_sha256=row.get("source_sha256", ""),
        seg_start=int(row.get("segment_start_sample_source", 0) or 0),
        seg_end=int(row.get("segment_end_sample_source", 0) or 0),
        split=row.get("split") or "train",
        rights_status=row.get("rights_status", "UNKNOWN"),
        redistribution_permitted=bool(row.get("redistribution_permitted", False)),
        transcript_verified=bool(sidecar.get("transcript_verified",
                                             row.get("transcript_verified", False))),
        language_verified=bool(sidecar.get("language_verified",
                                           row.get("language_verified", False))),
        source_uri=row.get("source_uri", ""),
        source_revision=row.get("source_revision", ""),
        original_file_id=original,
        audio_sha256=row.get("audio_sha256", ""),
        quality_evidence_id=row.get("quality_evidence_id", ""),
        quality_policy_version=row.get("quality_policy_version", ""),
        source_license=row.get("source_license", "UNKNOWN"))


def export_run(workdir: str, run_id: str, dataset_dir: str, state_dir: str,
               require_redistributable: bool = True,
               changelog: str = "", **kwargs: Any) -> Dict[str, Any]:
    """Export every accepted row of a local run into the canonical dataset."""
    paths = RunPaths.create(workdir, run_id)
    rows = list(read_jsonl(paths.accepted))
    samples = [sample_from_accepted_row(r, paths.canonical_dir) for r in rows]
    return normalize_samples(samples, dataset_dir, state_dir,
                             require_redistributable=require_redistributable,
                             changelog=changelog, **kwargs)


def export_campaign_batch(store: Any, job_id: str, batch_id: str,
                          dataset_dir: str, state_dir: str,
                          metadata_file: Optional[str] = None,
                          require_redistributable: bool = True,
                          changelog: str = "", **kwargs: Any) -> Dict[str, Any]:
    """Export one campaign batch's accepted rows into the canonical dataset."""
    from ..campaign.processing import qc_paths

    paths = qc_paths(store, job_id, batch_id)
    rows = load_jsonl(os.path.join(paths.root, "BATCH_ACCEPTED.jsonl"))
    side = _sidecar_index(metadata_file)
    samples = [sample_from_accepted_row(r, paths.canonical_dir,
                                        side.get(r.get("original_file_id")))
               for r in rows]
    return normalize_samples(samples, dataset_dir, state_dir,
                             require_redistributable=require_redistributable,
                             changelog=changelog, **kwargs)
