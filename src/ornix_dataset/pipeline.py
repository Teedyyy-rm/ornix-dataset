"""End-to-end pipeline orchestration (spec §2, §8, §9).

Wires the phase modules for a local run: ingest -> technical gate -> VAD/detectors
-> deterministic policy -> conditional segmentation + canonical render + re-QC ->
dedup + leakage-safe split -> accepted manifest. Release build/verify/publish are
separate, gated steps. Fail-closed throughout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .audit.events import AuditLog
from .contracts.enums import DecisionState, IngestStatus, MeasurementStatus
from .contracts.quality import NoiseEvent, QualityEvidence
from .contracts.release import ReleaseRow
from .contracts.source import SourceRecord
from .curation.policy import PolicyConfig, PolicyEngine
from .curation.segmentation import plan_segments
from .curation.transcript import transcript_match_status
from .detectors import DetectorKind, registry
from .detectors.windowing import intersect_intervals, total_duration
from .dsp.audio import AudioBuffer
from .dsp.decode import decode_to_float
from .dsp.features import signal_stats
from .dsp.render import render_canonical_wav
from .dsp.technical import TechnicalThresholds, run_technical_validation
from .util.hashing import sha256_json, short_id
from .util.io import write_json
from .util.jsonl import append_jsonl, write_jsonl
from .util.timeutil import utc_now_iso


@dataclass
class RunPaths:
    root: str
    source_manifest: str = ""
    evidence: str = ""
    accepted: str = ""
    review: str = ""
    audit: str = ""
    canonical_dir: str = ""

    @classmethod
    def create(cls, workdir: str, run_id: str) -> "RunPaths":
        root = os.path.join(workdir, "runs", run_id)
        os.makedirs(root, exist_ok=True)
        canonical = os.path.join(root, "canonical")
        os.makedirs(canonical, exist_ok=True)
        return cls(root=root,
                   source_manifest=os.path.join(root, "source_manifest.jsonl"),
                   evidence=os.path.join(root, "evidence.jsonl"),
                   accepted=os.path.join(root, "accepted_manifest.jsonl"),
                   review=os.path.join(root, "review_queue.jsonl"),
                   audit=os.path.join(root, "audit.jsonl"),
                   canonical_dir=canonical)


@dataclass
class AnalyzeResult:
    evidences: List[QualityEvidence] = field(default_factory=list)
    accepted: List[ReleaseRow] = field(default_factory=list)


class OrnixPipeline:
    def __init__(self, workdir: str, policy: PolicyConfig,
                 vad_name: str = "energy", noise_name: str = "dsp",
                 tech: Optional[TechnicalThresholds] = None):
        self.workdir = workdir
        self.policy = policy
        self.engine = PolicyEngine(policy)
        self.vad = registry.create(DetectorKind.VAD, vad_name)
        self.noise = registry.create(DetectorKind.NOISE, noise_name)
        self.tech = tech or TechnicalThresholds()
        self.availability = {"music": False, "quality": False, "speaker": False}

    def analyze_source(self, src: SourceRecord, paths: RunPaths,
                       audit: AuditLog) -> AnalyzeResult:
        result = AnalyzeResult()
        if src.ingest_status != IngestStatus.INGESTED or not src.staged_path:
            return result
        report = run_technical_validation(src.staged_path, self.tech,
                                          declared_duration_s=src.source_duration_s)
        audit.emit("technical", src.source_id, decision=report.decision,
                   reasons=report.reason_codes)
        if not report.valid:
            ev = self._reject_tech_evidence(src, report)
            append_jsonl(paths.evidence, ev.to_dict())
            result.evidences.append(ev)
            return result
        buf, _ = decode_to_float(src.staged_path, mono=True)
        self._analyze_segments(src, buf, paths, audit, result)
        return result

    def _reject_tech_evidence(self, src: SourceRecord, report) -> QualityEvidence:
        return QualityEvidence(
            segment_id="SEG_" + short_id(src.source_sha256, "tech"),
            source_sha256=src.source_sha256, interval_start_sample=0,
            interval_end_sample=0, analysis_sample_rate=16000,
            decision=DecisionState.REJECT_TECH, reason_codes=report.reason_codes,
            policy_version=self.policy.policy_version, timestamp_utc=utc_now_iso(),
        )

    def _analyze_segments(self, src, buf: AudioBuffer, paths, audit, result) -> None:
        sr = buf.sample_rate
        speech = self.vad.infer(buf.samples, sr)
        events = self.noise.infer(buf.samples, sr, speech)
        exclude = [[e.start_s, e.end_s] for e in events
                   if e.severity.value in ("N3",) or e.label.value == "MUSIC_ONLY"]
        plan = plan_segments(speech, exclude,
                             max_duration_s=self.tech.max_duration_s,
                             min_duration_s=self.tech.min_duration_s)
        if not plan.intervals_s:
            plan.intervals_s = [[0.0, min(buf.duration_s, self.tech.max_duration_s)]]
        for idx, (s, e) in enumerate(plan.intervals_s):
            self._process_segment(src, buf, s, e, idx, speech, paths, audit, result)

    def _process_segment(self, src, buf, s, e, idx, speech, paths, audit, result) -> None:
        sr = buf.sample_rate
        a, b = int(s * sr), int(e * sr)
        seg = buf.samples[a:b]
        if seg.size == 0:
            return
        seg_buf = AudioBuffer(seg, sr)
        seg_id = "SEG_" + short_id(src.source_sha256, idx, round(s, 3), round(e, 3))
        out_wav = os.path.join(paths.canonical_dir, f"{seg_id}.wav")
        try:
            audio_sha, recipe = render_canonical_wav(seg_buf, out_wav)
        except RuntimeError as exc:
            audit.emit("render_failed", seg_id, reason=str(exc))
            return
        ev = self._build_evidence(src, seg_buf, seg_id, a, b, s, e, speech, recipe)
        decision = self.engine.decide(ev, src, self.availability)
        ev.decision = decision.decision
        ev.reason_codes = decision.reason_codes
        ev.observed_checks = decision.observed_checks
        ev.required_checks = decision.required_checks
        append_jsonl(paths.evidence, ev.to_dict())
        result.evidences.append(ev)
        audit.emit("decision", seg_id, decision=decision.decision.value,
                   reasons=decision.reason_codes)
        if decision.decision == DecisionState.ACCEPT:
            row = self._release_row(src, ev, seg_id, out_wav, audio_sha, a, b, seg_buf)
            append_jsonl(paths.accepted, row.to_dict())
            result.accepted.append(row)

    def _build_evidence(self, src, seg_buf, seg_id, a, b, s, e, speech, recipe) -> QualityEvidence:
        sr = seg_buf.sample_rate
        stats = signal_stats(seg_buf.samples)
        seg_speech = intersect_intervals(speech, [[s, e]])
        seg_dur = max(1e-9, e - s)
        speech_ratio = min(1.0, total_duration(seg_speech) / seg_dur)
        seg_events = self.noise.infer(seg_buf.samples, sr, [[0.0, seg_dur]])
        tstatus = transcript_match_status(
            src.source_transcript, verified=bool(getattr(src, "_transcript_verified", False)))
        proc_sha = sha256_json(recipe.to_dict())
        return QualityEvidence(
            segment_id=seg_id, source_sha256=src.source_sha256,
            interval_start_sample=a, interval_end_sample=b, analysis_sample_rate=sr,
            vad_intervals=[[round(x, 4), round(y, 4)] for x, y in seg_speech],
            noise_events=seg_events,
            snr_status=MeasurementStatus.UNKNOWN,
            clipping_ratio=stats.clipping_ratio, max_clipped_run=stats.max_clipped_run,
            speech_ratio=round(speech_ratio, 4),
            quality_status=MeasurementStatus.UNKNOWN,
            speaker_status=(MeasurementStatus.UNKNOWN if self.policy.single_speaker_required
                            else MeasurementStatus.NOT_APPLICABLE),
            transcript_match_status=tstatus, calibration_domain="UNKNOWN",
            policy_version=self.policy.policy_version, processing_sha256=proc_sha,
            timestamp_utc=utc_now_iso(),
        )

    def _release_row(self, src, ev, seg_id, out_wav, audio_sha, a, b, seg_buf) -> ReleaseRow:
        audio_id = "ornix_" + (src.source_language or "vi") + "_" + short_id(audio_sha, length=12)
        return ReleaseRow(
            audio_id=audio_id, audio=f"audio/{seg_id}.wav",
            language=src.source_language or "vi",
            speaker_id=src.source_speaker_ref or ("anon_" + short_id(src.source_sha256, length=10)),
            transcript=src.source_transcript or "",
            sample_rate=24000, channels=1, encoding="PCM_S16LE",
            duration_s=round(seg_buf.duration_s, 6),
            source_id=src.source_id, source_sha256=src.source_sha256, audio_sha256=audio_sha,
            segment_start_sample_source=a, segment_end_sample_source=b,
            rights_record_id="RIGHTS_" + short_id(src.source_id, length=10),
            quality_evidence_id=seg_id, quality_policy_version=self.policy.policy_version,
            quality_gate="ACCEPT", split="train", release_id="pending",
        )

    def ingest_and_stage(self, adapter, paths: RunPaths, audit: AuditLog) -> List[SourceRecord]:
        import shutil

        from .util.jsonl import read_jsonl

        staging = os.path.join(paths.root, "staging")
        os.makedirs(staging, exist_ok=True)
        records: List[SourceRecord] = []
        seen_sha: Dict[str, str] = {}
        # idempotent / incremental: preload already-ingested source hashes
        for prev in read_jsonl(paths.source_manifest):
            seen_sha.setdefault(prev.get("source_sha256"), prev.get("source_id"))
        for rec in adapter.scan():
            if rec.source_sha256 in seen_sha:
                audit.emit("ingest_skip_duplicate", rec.source_id,
                           duplicate_of=seen_sha[rec.source_sha256])
                continue
            seen_sha[rec.source_sha256] = rec.source_id
            if rec.ingest_status == IngestStatus.INGESTED and rec.source_uri.startswith("file://"):
                src_path = rec.source_uri[len("file://"):]
                ext = os.path.splitext(src_path)[1] or ".wav"
                dst = os.path.join(staging, f"{rec.source_id}{ext}")
                if not os.path.exists(dst):
                    shutil.copyfile(src_path, dst)
                    os.chmod(dst, 0o444)  # immutable staging
                rec.staged_path = dst
            append_jsonl(paths.source_manifest, rec.to_dict())
            audit.emit("ingest", rec.source_id, status=rec.ingest_status.value,
                       rights=rec.rights_status.value)
            records.append(rec)
        return records

    def run(self, adapter, paths: RunPaths, audit: AuditLog) -> AnalyzeResult:
        records = self.ingest_and_stage(adapter, paths, audit)
        combined = AnalyzeResult()
        for rec in records:
            res = self.analyze_source(rec, paths, audit)
            combined.evidences.extend(res.evidences)
            combined.accepted.extend(res.accepted)
        self._finalize_splits(combined, paths, audit)
        return combined

    def _finalize_splits(self, combined: AnalyzeResult, paths: RunPaths, audit: AuditLog) -> None:
        from .curation.split import assign_splits, split_leakage

        if not combined.accepted:
            return
        # group by speaker (fallback source) to prevent leakage
        group_keys = {row.audio_id: (row.speaker_id or row.source_id)
                      for row in combined.accepted}
        assignment = assign_splits(group_keys)
        leak = split_leakage(group_keys, assignment)
        if leak:  # fail-closed: never emit a leaking split
            raise RuntimeError(f"split leakage detected for groups: {leak}")
        rows = []
        for row in combined.accepted:
            row.split = assignment[row.audio_id]
            rows.append(row.to_dict())
        write_jsonl(paths.accepted, rows)
        audit.emit("splits", "run", n_accepted=len(rows),
                   distribution={s: sum(1 for r in rows if r["split"] == s)
                                 for s in set(assignment.values())})
