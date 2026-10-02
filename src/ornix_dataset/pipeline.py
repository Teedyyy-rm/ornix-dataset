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
from .curation.segmentation import low_energy_points, plan_segments
from .curation.transcript import transcript_match_status
from .detectors import DetectorKind, registry
from .detectors.windowing import intersect_intervals, total_duration
from .dsp.audio import AudioBuffer
from .dsp.admission import AdmissionConfig, AdmissionReport, assess_source
from .dsp.decode import DecodeError, decode_to_float, wav_only_enabled
from .dsp.features import signal_stats
from .dsp.render import RenderVerificationError, render_canonical_wav
from .dsp.technical import (TechnicalReport, TechnicalThresholds, probe_info_for,
                            run_technical_validation_on_buffer)
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


def no_split_enabled() -> bool:
    """True when ORNIX_NO_SPLIT=1: discard splittable sources whole.

    Long files (>max) are already dropped by the technical gate; this covers
    the remainder — files the segment planner would cut or trim. No salvaged
    parts, no transcript-stripped rows in the release.
    """
    v = os.environ.get("ORNIX_NO_SPLIT", "0").strip().lower()
    return v not in ("0", "false", "no", "off", "")


def slice_noise_events(events: List[NoiseEvent], s: float,
                       e: float) -> List[NoiseEvent]:
    """Restrict full-clip noise events to segment [s, e) in segment coords.

    Fail-closed bias: any event merely overlapping the segment is kept (clipped
    to bounds), never dropped on boundary ambiguity — policy gates over these
    events can only stay as strict or get stricter, never looser. This lets the
    pipeline run the DSP/music detectors once per source clip instead of once
    per segment.
    """
    out: List[NoiseEvent] = []
    for ev in events:
        if ev.end_s <= s or ev.start_s >= e:
            continue
        out.append(NoiseEvent(
            label=ev.label,
            start_s=round(max(ev.start_s, s) - s, 4),
            end_s=round(min(ev.end_s, e) - s, 4),
            overlaps_speech=ev.overlaps_speech,
            severity=ev.severity,
            confidence=ev.confidence,
            score=ev.score,
            detector=ev.detector,
            model_revision=ev.model_revision,
        ))
    return out


class OrnixPipeline:
    def __init__(self, workdir: str, policy: PolicyConfig,
                 vad_name: str = "energy", noise_name: str = "dsp",
                 tech: Optional[TechnicalThresholds] = None,
                 admission: Optional[AdmissionConfig] = None,
                 detectors: Optional[Any] = None):
        self.workdir = workdir
        self.policy = policy
        self.engine = PolicyEngine(policy)
        self.tech = tech or TechnicalThresholds()
        self.admission_cfg = admission or AdmissionConfig()
        if detectors is not None:
            self.vad = detectors.vad
            self.noise = detectors.noise
            self.music = detectors.music
            self.quality = detectors.quality
            self.speaker = detectors.speaker
            self.availability = dict(detectors.availability)
        else:
            self.vad = registry.create(DetectorKind.VAD, vad_name)
            self.noise = registry.create(DetectorKind.NOISE, noise_name)
            self.music = None
            self.quality = registry.create(DetectorKind.QUALITY, "dnsmos")
            self.speaker = registry.create(DetectorKind.SPEAKER, "pyannote")
            self.availability = {
                "music": bool(self.noise.info().detects_music and self.noise.available),
                "quality": bool(self.quality.available),
                "speaker": bool(self.speaker.available),
            }

    @staticmethod
    def _ingest_probe_info(src: SourceRecord) -> Optional[Dict[str, Any]]:
        """Rebuild an ffprobe-shaped dict from ingest-time provenance.

        Ingest already probed every staged file (ffprobe subprocess). Reusing it
        saves ~0.5s/clip here; run_technical_validation still guards with an
        on-disk size check and falls back to a fresh ffprobe on any mismatch.
        Returns None when ingest recorded no usable probe (same as before).
        """
        if src.source_codec is None or not src.source_bytes:
            return None
        return {
            "codec_name": src.source_codec,
            "sample_rate": src.source_sample_rate,
            "channels": src.source_channels,
            "bits_per_raw_sample": None,
            "sample_fmt": None,
            "duration_s": src.source_duration_s,
            "format_name": src.source_container,
            "size_bytes": src.source_bytes,
        }

    def analyze_source(self, src: SourceRecord, paths: RunPaths,
                       audit: AuditLog) -> AnalyzeResult:
        result = AnalyzeResult()
        # quarantined / errored / not-yet-ingested sources are not analyzable, but a
        # record that claims INGESTED with no staged bytes is a real fault, not a
        # thing to silently drop — emit an ERROR evidence + audit event (no-loss).
        if src.ingest_status != IngestStatus.INGESTED:
            return result
        if not src.staged_path or not os.path.exists(src.staged_path):
            ev = self._error_evidence(src, "STAGED_PATH_MISSING")
            append_jsonl(paths.evidence, ev.to_dict())
            result.evidences.append(ev)
            audit.emit("analyze_error", src.source_id, reason="STAGED_PATH_MISSING",
                       staged_path=src.staged_path or "")
            return result
        # wav-only fast path: skip non-WAV before any ffprobe/decode/convert.
        if wav_only_enabled() and os.path.splitext(src.staged_path)[1].lower() != ".wav":
            report = TechnicalReport(False, "REJECT_TECH", ["NON_WAV_SKIPPED"])
            audit.emit("technical", src.source_id, decision=report.decision,
                       reasons=report.reason_codes)
            ev = self._reject_tech_evidence(src, report)
            append_jsonl(paths.evidence, ev.to_dict())
            result.evidences.append(ev)
            return result
        # Decode ONCE (mono=False) and reuse the buffer for both the technical
        # gate and the downstream mono analysis — previously this decoded the
        # file twice (once stereo in the gate, once mono here).
        try:
            info = probe_info_for(src.staged_path,
                                  self._ingest_probe_info(src))
        except DecodeError as e:
            report = TechnicalReport(False, "ERROR", [f"PROBE_FAILED:{e}"])
            audit.emit("technical", src.source_id, decision=report.decision,
                       reasons=report.reason_codes)
            ev = self._reject_tech_evidence(src, report)
            append_jsonl(paths.evidence, ev.to_dict())
            result.evidences.append(ev)
            return result
        try:
            buf_stereo, _decoder = decode_to_float(src.staged_path, mono=False)
        except DecodeError as e:
            report = TechnicalReport(
                False, "REJECT_TECH", [f"DECODE_FAILED:{e}"],
                measured_sample_rate=info.get("sample_rate"),
                codec_name=info.get("codec_name"),
                format_name=info.get("format_name"))
            audit.emit("technical", src.source_id, decision=report.decision,
                       reasons=report.reason_codes)
            ev = self._reject_tech_evidence(src, report)
            append_jsonl(paths.evidence, ev.to_dict())
            result.evidences.append(ev)
            return result
        report = run_technical_validation_on_buffer(
            buf_stereo, info, self.tech,
            declared_duration_s=src.source_duration_s)
        audit.emit("technical", src.source_id, decision=report.decision,
                   reasons=report.reason_codes)
        if not report.valid:
            ev = self._reject_tech_evidence(src, report)
            append_jsonl(paths.evidence, ev.to_dict())
            result.evidences.append(ev)
            return result
        # Level 1 — source admission: decide clean-HQ eligibility from *measured*
        # evidence before spending any canonicalization. A rejected source stays
        # auditable in evidence (no-loss) but never reaches accepted/release.
        admission = assess_source(report, self.admission_cfg)
        audit.emit("admission", src.source_id,
                   rate_class=admission.source_rate_class,
                   action=admission.canonicalization_action,
                   admitted=admission.admitted, reasons=admission.reason_codes)
        if not admission.admitted:
            ev = self._reject_admission_evidence(src, report, admission)
            append_jsonl(paths.evidence, ev.to_dict())
            result.evidences.append(ev)
            return result
        # Reuse the single decode above: to_mono() is a no-op (no copy) when
        # the source is already mono, and a cheap mean-downmix otherwise.
        buf = buf_stereo.to_mono()
        self._analyze_segments(src, buf, paths, audit, result, admission)
        return result

    def _reject_admission_evidence(self, src: SourceRecord, report,
                                   admission: AdmissionReport) -> QualityEvidence:
        return QualityEvidence(
            segment_id="SEG_" + short_id(src.source_sha256, "admission"),
            source_sha256=src.source_sha256, interval_start_sample=0,
            interval_end_sample=0, analysis_sample_rate=16000,
            decision=DecisionState.REJECT_TECH, reason_codes=admission.reason_codes,
            source_container=admission.source_container,
            source_lossy=admission.source_lossy,
            source_rate_class=admission.source_rate_class,
            canonicalization_action=admission.canonicalization_action,
            effective_bandwidth_hz=admission.effective_bandwidth_hz,
            low_bandwidth_suspected=admission.low_bandwidth_suspected,
            policy_version=self.policy.policy_version, timestamp_utc=utc_now_iso(),
        )

    def _reject_tech_evidence(self, src: SourceRecord, report) -> QualityEvidence:
        return QualityEvidence(
            segment_id="SEG_" + short_id(src.source_sha256, "tech"),
            source_sha256=src.source_sha256, interval_start_sample=0,
            interval_end_sample=0, analysis_sample_rate=16000,
            decision=DecisionState.REJECT_TECH, reason_codes=report.reason_codes,
            policy_version=self.policy.policy_version, timestamp_utc=utc_now_iso(),
        )

    def _error_evidence(self, src: SourceRecord, reason: str) -> QualityEvidence:
        return QualityEvidence(
            segment_id="SEG_" + short_id(src.source_sha256, "error", reason),
            source_sha256=src.source_sha256, interval_start_sample=0,
            interval_end_sample=0, analysis_sample_rate=16000,
            decision=DecisionState.ERROR, reason_codes=[reason],
            policy_version=self.policy.policy_version, timestamp_utc=utc_now_iso(),
        )

    def _detect_noise(self, samples, sr, speech_intervals) -> List[NoiseEvent]:
        """DSP noise (always) + optional music-gate detector, merged."""
        events = list(self.noise.infer(samples, sr, speech_intervals))
        if self.music is not None and self.music.available:
            events.extend(self.music.infer(samples, sr, speech_intervals))
        return events

    def _batch_quality(self, buf: AudioBuffer,
                       intervals: List[List[float]]) -> Optional[Dict[int, Dict[str, Any]]]:
        """Score all segment DNSMOS inputs in one session.run.

        Returns idx -> infer()-shaped dict. None when the adapter is missing /
        unavailable or batching fails — _build_evidence then falls back to the
        original per-segment infer(), so failure semantics never change.
        """
        if self.quality is None or not self.quality.available:
            return None
        batch_fn = getattr(self.quality, "infer_batch", None)
        if batch_fn is None:
            return None
        sr = buf.sample_rate
        idxs: List[int] = []
        items: List[Any] = []
        for idx, (s, e) in enumerate(intervals):
            seg = buf.samples[int(s * sr):int(e * sr)]
            if seg.size == 0:
                continue
            idxs.append(idx)
            items.append((seg, sr))
        if not items:
            return None
        try:
            res = batch_fn(items)
        except Exception:
            return None
        return dict(zip(idxs, res))

    def _reject_short_source(self, src, buf: AudioBuffer, paths, audit, result,
                              admission: Optional[AdmissionReport] = None) -> None:
        """Fast REJECT for sources shorter than the policy minimum.

        Duration is already measured (technical gate + decode); a source below
        policy.min_duration_s can never ACCEPT (duration_ok is a required
        check), so running VAD/noise/DNSMOS/render on it only burns seconds
        for an identical REJECT. The evidence still goes through the real
        PolicyEngine, so decision + reason codes match the full path exactly.
        """
        sr = buf.sample_rate
        seg_id = "SEG_" + short_id(src.source_sha256, "short")
        ev = QualityEvidence(
            segment_id=seg_id, source_sha256=src.source_sha256,
            interval_start_sample=0, interval_end_sample=buf.n_samples,
            analysis_sample_rate=sr,
            transcript_match_status=transcript_match_status(
                src.source_transcript,
                verified=bool(getattr(src, "_transcript_verified", False))),
            source_container=getattr(admission, "source_container", None),
            source_lossy=getattr(admission, "source_lossy", None),
            source_rate_class=getattr(admission, "source_rate_class", None),
            canonicalization_action=getattr(admission, "canonicalization_action", None),
            effective_bandwidth_hz=getattr(admission, "effective_bandwidth_hz", None),
            low_bandwidth_suspected=bool(getattr(admission, "low_bandwidth_suspected", False)),
            policy_version=self.policy.policy_version, timestamp_utc=utc_now_iso(),
        )
        decision = self.engine.decide(ev, src, self.availability)
        ev.decision = decision.decision
        ev.reason_codes = decision.reason_codes
        ev.observed_checks = decision.observed_checks
        ev.required_checks = decision.required_checks
        append_jsonl(paths.evidence, ev.to_dict())
        result.evidences.append(ev)
        audit.emit("decision", seg_id, decision=decision.decision.value,
                   reasons=decision.reason_codes)

    def _reject_split_source(self, src, buf: AudioBuffer, paths, audit, result,
                               admission: Optional[AdmissionReport] = None,
                               n_intervals: int = 0) -> None:
        """Discard a source that would need segmentation (ORNIX_NO_SPLIT=1).

        Long/noisy files are dropped whole instead of salvaged in parts, so
        no sub-segment with a stripped transcript ever enters the release.
        The evidence still goes through the real PolicyEngine for check
        bookkeeping, but the decision is forced to REJECT — no required
        check covers splitting, so the engine alone could still ACCEPT.
        """
        sr = buf.sample_rate
        seg_id = "SEG_" + short_id(src.source_sha256, "nosplit")
        ev = QualityEvidence(
            segment_id=seg_id, source_sha256=src.source_sha256,
            interval_start_sample=0, interval_end_sample=buf.n_samples,
            analysis_sample_rate=sr,
            transcript_match_status=transcript_match_status(
                src.source_transcript,
                verified=bool(getattr(src, "_transcript_verified", False))),
            source_container=getattr(admission, "source_container", None),
            source_lossy=getattr(admission, "source_lossy", None),
            source_rate_class=getattr(admission, "source_rate_class", None),
            canonicalization_action=getattr(admission, "canonicalization_action", None),
            effective_bandwidth_hz=getattr(admission, "effective_bandwidth_hz", None),
            low_bandwidth_suspected=bool(getattr(admission, "low_bandwidth_suspected", False)),
            policy_version=self.policy.policy_version, timestamp_utc=utc_now_iso(),
        )
        decision = self.engine.decide(ev, src, self.availability)
        reasons = list(decision.reason_codes)
        if "REQUIRES_SEGMENTATION" not in reasons:
            reasons.append("REQUIRES_SEGMENTATION")
        ev.decision = DecisionState.REJECT
        ev.reason_codes = reasons
        ev.observed_checks = decision.observed_checks
        ev.required_checks = decision.required_checks
        append_jsonl(paths.evidence, ev.to_dict())
        result.evidences.append(ev)
        audit.emit("decision", seg_id, decision="REJECT",
                    reasons=reasons, n_intervals=n_intervals)

    def _analyze_segments(self, src, buf: AudioBuffer, paths, audit, result,
                           admission: Optional[AdmissionReport] = None) -> None:
        sr = buf.sample_rate
        if buf.duration_s < self.policy.min_duration_s:
            self._reject_short_source(src, buf, paths, audit, result, admission)
            return
        speech = self.vad.infer(buf.samples, sr)
        events = self._detect_noise(buf.samples, sr, speech)
        exclude = [[e.start_s, e.end_s] for e in events
                   if e.severity.value in ("N3",) or e.label.value == "MUSIC_ONLY"]
        # low-energy minima as candidate cut boundaries (never cut mid-syllable)
        silence_points = low_energy_points(buf.samples, sr)
        plan = plan_segments(speech, exclude,
                             max_duration_s=self.tech.max_duration_s,
                             min_duration_s=self.tech.min_duration_s,
                             silence_points=silence_points)
        if not plan.intervals_s:
            plan.intervals_s = [[0.0, min(buf.duration_s, self.tech.max_duration_s)]]
        segmented = len(plan.intervals_s) > 1 or (
            plan.intervals_s and total_duration(plan.intervals_s) < buf.duration_s - 0.2)
        if no_split_enabled() and segmented:
            self._reject_split_source(src, buf, paths, audit, result, admission,
                                      n_intervals=len(plan.intervals_s))
            return
        quality_batch = self._batch_quality(buf, plan.intervals_s)
        for idx, (s, e) in enumerate(plan.intervals_s):
            uncertain = idx in getattr(plan, "uncertain_indices", set())
            self._process_segment(src, buf, s, e, idx, speech, paths, audit, result,
                                   segmented=segmented, boundary_uncertain=uncertain,
                                   admission=admission, full_events=events,
                                   quality_batch=quality_batch)

    def _process_segment(self, src, buf, s, e, idx, speech, paths, audit, result,
                          segmented: bool = False, boundary_uncertain: bool = False,
                          admission: Optional[AdmissionReport] = None,
                          full_events: Optional[List[NoiseEvent]] = None,
                          quality_batch: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
        sr = buf.sample_rate
        a, b = int(s * sr), int(e * sr)
        seg = buf.samples[a:b]
        if seg.size == 0:
            return
        seg_buf = AudioBuffer(seg, sr)
        seg_id = "SEG_" + short_id(src.source_sha256, idx, round(s, 3), round(e, 3))
        out_wav = os.path.join(paths.canonical_dir, f"{seg_id}.wav")
        try:
            audio_sha, recipe = render_canonical_wav(seg_buf, out_wav, admission=admission)
        except (RuntimeError, RenderVerificationError) as exc:
            # fail-closed + no-loss: a render/verify failure is an auditable ERROR
            # evidence row, never a silently dropped segment (invariants I10/I12).
            audit.emit("render_failed", seg_id, reason=str(exc))
            ev = self._error_evidence(src, f"CANONICAL_RENDER_FAILED:{exc}")
            ev.segment_id = seg_id
            append_jsonl(paths.evidence, ev.to_dict())
            result.evidences.append(ev)
            return
        # a sub-segment cannot inherit the whole-source transcript verbatim
        transcript_valid = not (segmented or boundary_uncertain)
        ev = self._build_evidence(src, seg_buf, seg_id, a, b, s, e, speech, recipe,
                                   transcript_valid=transcript_valid,
                                   admission=admission, canonical_sha256=audio_sha,
                                   full_events=full_events,
                                   seg_idx=idx, quality_batch=quality_batch)
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
            row = self._release_row(src, ev, seg_id, out_wav, audio_sha, a, b, seg_buf,
                                    transcript_valid=transcript_valid)
            append_jsonl(paths.accepted, row.to_dict())
            result.accepted.append(row)

    def _build_evidence(self, src, seg_buf, seg_id, a, b, s, e, speech, recipe,
                         transcript_valid: bool = True,
                         admission: Optional[AdmissionReport] = None,
                         canonical_sha256: Optional[str] = None,
                         full_events: Optional[List[NoiseEvent]] = None,
                         seg_idx: Optional[int] = None,
                         quality_batch: Optional[Dict[int, Dict[str, Any]]] = None) -> QualityEvidence:
        sr = seg_buf.sample_rate
        stats = signal_stats(seg_buf.samples)
        seg_speech = intersect_intervals(speech, [[s, e]])
        seg_dur = max(1e-9, e - s)
        speech_ratio = min(1.0, total_duration(seg_speech) / seg_dur)
        if full_events is not None:
            seg_events = slice_noise_events(full_events, s, e)
        else:
            seg_events = self._detect_noise(seg_buf.samples, sr, [[0.0, seg_dur]])
        # transcript: only trust the source transcript for a whole-source clip
        if transcript_valid:
            tstatus = transcript_match_status(
                src.source_transcript,
                verified=bool(getattr(src, "_transcript_verified", False)))
        else:
            tstatus = MeasurementStatus.UNKNOWN  # needs forced alignment on the sub-segment

        # real quality (DNSMOS) evidence when the adapter is available
        sig = bak = ovrl = None
        quality_status = MeasurementStatus.UNKNOWN
        if quality_batch is not None and seg_idx in quality_batch:
            q = quality_batch[seg_idx]
            sig, bak, ovrl = q.get("sig"), q.get("bak"), q.get("ovrl")
            quality_status = MeasurementStatus(q.get("status", "UNKNOWN"))
        elif self.quality is not None and self.quality.available:
            q = self.quality.infer(seg_buf.samples, sr)
            sig, bak, ovrl = q.get("sig"), q.get("bak"), q.get("ovrl")
            quality_status = MeasurementStatus(q.get("status", "UNKNOWN"))

        # real speaker/overlap (pyannote) evidence when available
        speaker_overlap: List[List[float]] = []
        if self.policy.single_speaker_required:
            speaker_status = MeasurementStatus.UNKNOWN
            if self.speaker is not None and self.speaker.available:
                sp = self.speaker.infer(seg_buf.samples, sr)
                speaker_overlap = sp.get("overlap_intervals", []) or []
                speaker_status = MeasurementStatus(sp.get("status", "UNKNOWN"))
        else:
            speaker_status = MeasurementStatus.NOT_APPLICABLE

        proc_sha = sha256_json(recipe.to_dict())
        return QualityEvidence(
            segment_id=seg_id, source_sha256=src.source_sha256,
            interval_start_sample=a, interval_end_sample=b, analysis_sample_rate=sr,
            vad_intervals=[[round(x, 4), round(y, 4)] for x, y in seg_speech],
            noise_events=seg_events,
            snr_status=MeasurementStatus.UNKNOWN,
            clipping_ratio=stats.clipping_ratio, max_clipped_run=stats.max_clipped_run,
            speech_ratio=round(speech_ratio, 4),
            sig=sig, bak=bak, ovrl=ovrl,
            quality_status=quality_status,
            speaker_overlap_intervals=speaker_overlap,
            speaker_status=speaker_status,
            transcript_match_status=tstatus, calibration_domain="UNKNOWN",
            source_container=getattr(admission, "source_container", None),
            source_lossy=getattr(admission, "source_lossy", None),
            source_rate_class=getattr(admission, "source_rate_class", None),
            canonicalization_action=recipe.canonicalization_action,
            effective_bandwidth_hz=getattr(admission, "effective_bandwidth_hz", None),
            low_bandwidth_suspected=bool(getattr(admission, "low_bandwidth_suspected", False)),
            canonical_sha256=canonical_sha256,
            canonical_verify_status=recipe.canonical_verify_status,
            policy_version=self.policy.policy_version, processing_sha256=proc_sha,
            timestamp_utc=utc_now_iso(),
        )

    def _release_row(self, src, ev, seg_id, out_wav, audio_sha, a, b, seg_buf,
                     transcript_valid: bool = True) -> ReleaseRow:
        audio_id = "ornix_" + (src.source_language or "vi") + "_" + short_id(audio_sha, length=12)
        # never publish a whole-source transcript against a sub-segment
        transcript = (src.source_transcript or "") if transcript_valid else ""
        return ReleaseRow(
            audio_id=audio_id, audio=f"audio/{seg_id}.wav",
            language=src.source_language or "vi",
            speaker_id=src.source_speaker_ref or ("anon_" + short_id(src.source_sha256, length=10)),
            transcript=transcript,
            sample_rate=seg_buf.sample_rate, channels=1, encoding="PCM_S16LE",
            duration_s=round(seg_buf.duration_s, 6),
            source_id=src.source_id, source_sha256=src.source_sha256, audio_sha256=audio_sha,
            segment_start_sample_source=a, segment_end_sample_source=b,
            rights_record_id="RIGHTS_" + short_id(src.source_id, length=10),
            source_license=src.source_license,
            rights_status=src.rights_status.value,
            redistribution_permitted=bool(src.redistribution_permitted),
            quality_evidence_id=seg_id, quality_policy_version=self.policy.policy_version,
            quality_gate="ACCEPT", split="train", release_id="pending",
            source_uri=src.source_uri, source_revision=src.source_revision,
            original_file_id=src.original_file_id,
            source_speaker_ref=src.source_speaker_ref,
            language_verified=bool(src.source_language),
            transcript_verified=bool(transcript_valid and src.source_transcript),
        )

    def ingest_and_stage(self, adapter, paths: RunPaths, audit: AuditLog) -> List[SourceRecord]:
        from .ingestion.hf import HfSourceAdapter

        # HF batch path: bounded concurrent downloader. file_workers=1 keeps
        # the exact legacy sequential behavior (back-compatible, T16).
        if isinstance(adapter, HfSourceAdapter) and \
                adapter.download_cfg.effective().file_workers > 1:
            return self.ingest_and_stage_hf(adapter, paths, audit)
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
            if rec.ingest_status == IngestStatus.INGESTED:
                if rec.source_uri.startswith("file://"):
                    src_path = rec.source_uri[len("file://"):]
                    ext = os.path.splitext(src_path)[1] or ".wav"
                    dst = os.path.join(staging, f"{rec.source_id}{ext}")
                    if not os.path.exists(dst):
                        shutil.copyfile(src_path, dst)
                        os.chmod(dst, 0o444)  # immutable staging
                    rec.staged_path = dst
                elif rec.source_uri.startswith("hf://") and hasattr(adapter, "materialize"):
                    # download the pinned-commit blob, verify content sha, stage immutably
                    try:
                        adapter.materialize(rec, staging)
                        audit.emit("materialize", rec.source_id, staged=bool(rec.staged_path))
                    except Exception as exc:  # fail-closed: no hidden skip
                        rec.ingest_status = IngestStatus.ERROR
                        rec.reason_codes = list(rec.reason_codes) + [f"MATERIALIZE_FAILED:{exc}"]
                        audit.emit("materialize_failed", rec.source_id, reason=str(exc))
            append_jsonl(paths.source_manifest, rec.to_dict())
            audit.emit("ingest", rec.source_id, status=rec.ingest_status.value,
                       rights=rec.rights_status.value)
            records.append(rec)
        return records

    def ingest_and_stage_hf(self, adapter, paths: RunPaths,
                            audit: AuditLog) -> List[SourceRecord]:
        """Bounded-concurrent HF ingest (file_workers > 1).

        Manifest order == scan/inventory order regardless of completion order
        (single ordered writer), so downstream sees the same sequence as the
        legacy sequential path. Records for quarantined sources are kept in
        the manifest (as legacy) but never downloaded (PermissionGate first).
        """
        import os

        from .ingestion.hf import HfSourceAdapter
        from .ingestion.hf_downloader import (
            HfBatchDownloader,
            apply_env,
            build_inventory,
            classify_profile,
            effective_env_snapshot,
            split_allowed,
        )
        from .ops.checkpoint import Checkpoint
        from .util.jsonl import read_jsonl

        staging = os.path.join(paths.root, "staging")
        os.makedirs(staging, exist_ok=True)
        records: List[SourceRecord] = []
        seen_sha: Dict[str, str] = {}
        for prev in read_jsonl(paths.source_manifest):
            seen_sha.setdefault(prev.get("source_sha256"), prev.get("source_id"))

        scanned = list(adapter.scan())  # resolves + pins the commit SHA once
        fresh: List[SourceRecord] = []
        for r in scanned:
            if r.source_sha256 in seen_sha:
                audit.emit("ingest_skip_duplicate", r.source_id,
                           duplicate_of=seen_sha[r.source_sha256])
                continue
            seen_sha[r.source_sha256] = r.source_id
            fresh.append(r)
        eligible = [r for r in fresh if r.ingest_status == IngestStatus.INGESTED]
        splits = adapter.download_cfg.allow_splits
        by_path: Dict[str, SourceRecord] = {}
        for r in eligible:
            if not split_allowed(r.source_split, splits):
                audit.emit("ingest_skip_split", r.source_id, split=r.source_split)
                continue
            by_path[r.original_file_id] = r
        # quarantined/error records stay in the manifest (legacy parity) but
        # are never downloaded.
        by_ids = {id(r) for r in by_path.values()}
        for r in fresh:
            if id(r) not in by_ids:
                append_jsonl(paths.source_manifest, r.to_dict())
                audit.emit("ingest", r.source_id, status=r.ingest_status.value,
                           rights=r.rights_status.value)
                records.append(r)
        if not by_path:
            return records

        shas = {r.source_revision for r in by_path.values()}
        if len(shas) != 1:
            raise RuntimeError(f"mixed pinned revisions in one run: {shas}")
        pinned = next(iter(shas))
        cfg = adapter.download_cfg.effective()
        for warning in apply_env(cfg):
            audit.emit("download_env_warning", adapter.repo_id, warning=warning)
        inventory = build_inventory(
            adapter._api(), adapter.repo_id, pinned,
            allow=cfg.allow_patterns, ignore=cfg.ignore_patterns,
            allow_empty=cfg.allow_empty_inventory)
        inventory = [it for it in inventory if it.path_in_repo in by_path]
        if not inventory and cfg.allow_empty_inventory is not True:
            # No-loss: eligible records exist but nothing is downloadable
            # (patterns match nothing, or a shard-only repo with no loose
            # audio). Vanishing them silently would lose SourceRecords, so
            # fail closed. Shard-only repos are a documented FOLLOW_UP:
            # stage the shard as an artifact instead, out of scope here.
            missing = sorted(by_path)[:5]
            raise RuntimeError(
                f"0 downloadable files for {len(by_path)} eligible record(s) "
                f"(allow={cfg.allow_patterns} ignore={cfg.ignore_patterns}); "
                f"fail-closed, e.g. {missing}")
        if not inventory:
            # Explicit opt-in to empty: keep records auditable in the manifest
            # (no-loss) with a reason; downstream QC flags the missing bytes.
            for r in by_path.values():
                r.reason_codes = list(r.reason_codes) + ["INVENTORY_EMPTY_SKIPPED"]
                append_jsonl(paths.source_manifest, r.to_dict())
                audit.emit("ingest", r.source_id, status=r.ingest_status.value,
                           rights=r.rights_status.value,
                           reason="INVENTORY_EMPTY_SKIPPED")
                records.append(r)
            return records
        journal = Checkpoint(os.path.join(paths.root, "download_journal.jsonl"),
                             f"dl-{pinned[:12]}")
        # crash recovery without re-download: staged files from a previous
        # interrupted run are re-verified (size+sha), never trusted by existence.
        skip: Dict[str, str] = {}
        for it in inventory:
            rec = by_path[it.path_in_repo]
            ext = os.path.splitext(rec.original_file_id)[1] or ".bin"
            cand = os.path.join(staging, f"{rec.source_id}{ext}")
            if os.path.exists(cand):
                skip[f"{adapter.repo_id}@{pinned}/{it.path_in_repo}"] = cand
        dl = HfBatchDownloader(adapter.repo_id, pinned, staging, cfg=cfg,
                               journal=journal)
        audit.emit("download_start", adapter.repo_id, revision=pinned,
                   n_files=len(inventory),
                   profile=classify_profile(inventory, cfg),
                   env=effective_env_snapshot())
        results = dl.run(inventory,
                         staged_name=lambda it: f"{by_path[it.path_in_repo].source_id}"
                         f"{os.path.splitext(by_path[it.path_in_repo].original_file_id)[1] or '.bin'}",
                         skip_staged=skip)
        for res in results:  # inventory order (single ordered writer)
            rec = by_path[res.path_in_repo]
            rec.staged_path = res.staged_path
            HfSourceAdapter._patch_probe_provenance(rec, res.staged_path)
            append_jsonl(paths.source_manifest, rec.to_dict())
            audit.emit("ingest", rec.source_id, status=rec.ingest_status.value,
                       rights=rec.rights_status.value,
                       staged=os.path.basename(res.staged_path),
                       sha256=res.sha256, verify=res.verify_method,
                       from_cache=res.from_cache, reused=res.reused_staged)
            records.append(rec)
            seen_sha[rec.source_sha256] = rec.source_id
        write_json(os.path.join(paths.root, "download_metrics.json"),
                   {**dl.metrics.to_dict(),
                    "repo_id": adapter.repo_id, "revision": pinned})
        audit.emit("download_done", adapter.repo_id, **dl.metrics.to_dict())
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
