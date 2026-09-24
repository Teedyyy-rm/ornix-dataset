"""Deterministic, versioned policy engine (spec §5.3, Phase 5, tests T-001..T-010).

Pure function: same (evidence, source, policy, availability) => same decision.
Fail-closed:
  * a required check that cannot be evaluated (detector UNAVAILABLE / metric
    UNKNOWN) yields REVIEW, never ACCEPT;
  * a confirmed hard failure yields REJECT;
  * missing/unknown redistribution rights yields LICENSE_REVIEW for public target.
Thresholds live in a versioned config and only change via a new version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts.enums import DecisionState, MeasurementStatus, RightsStatus, Severity
from ..contracts.quality import QualityEvidence
from ..contracts.source import SourceRecord

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"


@dataclass
class PolicyConfig:
    policy_version: str
    required_checks: List[str]
    min_speech_ratio: float = 0.5
    max_clip_ratio: float = 0.01
    hiss_overlap_max_severity: str = "N1"  # allow up to this severity overlapping speech
    music_overlap_forbidden: bool = True
    single_speaker_required: bool = False
    release_target: str = "public"  # public | train_only
    accept_severities: List[str] = field(default_factory=lambda: ["N0", "N1"])
    # Optional perceptual-quality floors (DNSMOS P.835). Left unset by default:
    # the worker never auto-picks a publish threshold (spec §0.1) — an operator
    # binds these via calibration. When set, a score below the floor is a hard FAIL.
    min_quality_sig: Optional[float] = None
    min_quality_bak: Optional[float] = None
    min_quality_ovrl: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    decision: DecisionState
    reason_codes: List[str]
    observed_checks: Dict[str, str]
    required_checks: List[str]


_SEV_ORDER = {s.value: i for i, s in enumerate([Severity.N0, Severity.N1, Severity.N2,
                                                Severity.N3])}


class PolicyEngine:
    def __init__(self, config: PolicyConfig):
        self.config = config

    def decide(self, ev: QualityEvidence, source: SourceRecord,
               availability: Optional[Dict[str, bool]] = None) -> PolicyDecision:
        avail = availability or {}
        checks: Dict[str, str] = {}
        reasons: List[str] = []

        checks["has_speech"] = self._has_speech(ev, reasons)
        checks["no_clipping"] = self._no_clipping(ev, reasons)
        checks["no_confirmed_music_overlap"] = self._music(ev, avail, reasons)
        checks["no_severe_noise_overlap"] = self._noise_overlap(ev, reasons)
        checks["no_interfering_speech"] = self._interfering(ev, avail, reasons)
        checks["quality_ok"] = self._quality(ev, avail, reasons)
        checks["transcript_ok"] = self._transcript(ev, reasons)
        checks["rights_ok"] = self._rights(source, reasons)

        required = self.config.required_checks
        decision = self._combine(checks, required, source, reasons)
        return PolicyDecision(decision, reasons, checks, list(required))

    # --- individual checks -------------------------------------------------
    def _has_speech(self, ev, reasons) -> str:
        if ev.speech_ratio is None:
            reasons.append("SPEECH_RATIO_UNKNOWN")
            return UNKNOWN
        if ev.speech_ratio < self.config.min_speech_ratio:
            reasons.append(f"NO_SPEECH:ratio={ev.speech_ratio:.2f}")
            return FAIL
        return PASS

    def _no_clipping(self, ev, reasons) -> str:
        for e in ev.noise_events:
            if e.label.value == "CLIPPING_DISTORTION" and e.severity == Severity.N3:
                reasons.append("CLIPPING_CONFIRMED_N3")
                return FAIL
        if ev.clipping_ratio is not None and ev.clipping_ratio > self.config.max_clip_ratio:
            reasons.append(f"CLIPPING_RATIO:{ev.clipping_ratio:.4f}")
            return FAIL
        return PASS

    def _music(self, ev, avail, reasons) -> str:
        if not self.config.music_overlap_forbidden:
            return PASS
        if not avail.get("music", False):
            reasons.append("MUSIC_GATE_UNAVAILABLE")
            return UNKNOWN  # cannot confirm absence of music -> fail-closed
        for e in ev.noise_events:
            if e.label.value in ("MUSIC_BACKGROUND",) and e.overlaps_speech:
                if _SEV_ORDER.get(e.severity.value, 3) >= _SEV_ORDER["N2"]:
                    reasons.append("MUSIC_OVERLAP_CONFIRMED")
                    return FAIL
        return PASS

    def _noise_overlap(self, ev, reasons) -> str:
        cap = _SEV_ORDER.get(self.config.hiss_overlap_max_severity, 1)
        worst = None
        for e in ev.noise_events:
            if e.overlaps_speech and e.label.value in ("HISS_STATIC", "HUM_BUZZ",
                                                       "ENV_CONTINUOUS", "REVERB_ECHO"):
                lvl = _SEV_ORDER.get(e.severity.value, 3)
                worst = lvl if worst is None else max(worst, lvl)
        if worst is None:
            return PASS
        if worst > cap:
            reasons.append(f"NOISE_OVERLAP_SEVERITY>{self.config.hiss_overlap_max_severity}")
            return FAIL
        return PASS

    def _interfering(self, ev, avail, reasons) -> str:
        if not self.config.single_speaker_required:
            return PASS
        if ev.speaker_status == MeasurementStatus.UNKNOWN or not avail.get("speaker", False):
            reasons.append("SPEAKER_GATE_UNAVAILABLE")
            return UNKNOWN
        if ev.speaker_overlap_intervals:
            reasons.append("INTERFERING_SPEECH_CONFIRMED")
            return FAIL
        return PASS

    def _quality(self, ev, avail, reasons) -> str:
        if not avail.get("quality", False) or ev.quality_status != MeasurementStatus.OK:
            reasons.append("QUALITY_MODEL_UNAVAILABLE")
            return UNKNOWN
        floors = (("sig", self.config.min_quality_sig, ev.sig),
                  ("bak", self.config.min_quality_bak, ev.bak),
                  ("ovrl", self.config.min_quality_ovrl, ev.ovrl))
        for name, floor, score in floors:
            if floor is None:
                continue
            if score is None:
                reasons.append(f"QUALITY_{name.upper()}_UNKNOWN")
                return UNKNOWN
            if score < floor:
                reasons.append(f"QUALITY_{name.upper()}_BELOW:{score:.2f}<{floor:.2f}")
                return FAIL
        return PASS

    def _transcript(self, ev, reasons) -> str:
        st = ev.transcript_match_status
        if st == MeasurementStatus.OK:
            return PASS
        if st == MeasurementStatus.NOT_APPLICABLE:
            return PASS
        if st == MeasurementStatus.ERROR:
            reasons.append("TRANSCRIPT_MISMATCH")
            return FAIL
        reasons.append("TRANSCRIPT_UNVERIFIED")
        return UNKNOWN

    def _rights(self, source, reasons) -> str:
        if self.config.release_target == "public":
            if source.rights_status == RightsStatus.REDISTRIBUTION_APPROVED and \
                    source.redistribution_permitted:
                return PASS
            reasons.append(f"RIGHTS_NOT_REDISTRIBUTABLE:{source.rights_status.value}")
            return FAIL
        # train_only target
        if source.rights_status in (RightsStatus.REDISTRIBUTION_APPROVED,
                                    RightsStatus.TRAIN_ONLY):
            return PASS
        reasons.append(f"RIGHTS_INSUFFICIENT:{source.rights_status.value}")
        return FAIL

    # --- combine -----------------------------------------------------------
    def _combine(self, checks, required, source, reasons) -> DecisionState:
        req_results = {c: checks.get(c, UNKNOWN) for c in required}
        # hard failures first
        if any(v == FAIL for v in req_results.values()):
            if req_results.get("rights_ok") == FAIL and \
                    all(v != FAIL for k, v in req_results.items() if k != "rights_ok"):
                return DecisionState.LICENSE_REVIEW
            return DecisionState.REJECT
        if source.rights_status in (RightsStatus.LICENSE_REVIEW, RightsStatus.UNKNOWN):
            return DecisionState.LICENSE_REVIEW
        if any(v == UNKNOWN for v in req_results.values()):
            return DecisionState.REVIEW
        return DecisionState.ACCEPT


def load_policy(path: str) -> PolicyConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if "policy_version" not in raw:
        raise ValueError("policy config missing 'policy_version'")
    if "required_checks" not in raw:
        raise ValueError("policy config missing 'required_checks'")
    known = {f for f in PolicyConfig.__dataclass_fields__ if f != "raw"}
    kwargs = {k: v for k, v in raw.items() if k in known}
    return PolicyConfig(raw=raw, **kwargs)
