"""Config loading: sources, permission gate, models lock, pipeline assembly."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .curation.policy import PolicyConfig, load_policy
from .detectors import DetectorKind, registry
from .dsp.admission import AdmissionConfig
from .dsp.technical import TechnicalThresholds
from .ingestion.hf import HfSourceAdapter
from .ingestion.local import LocalSourceAdapter
from .ingestion.permissions import PermissionGate
from .util.jsonl import read_jsonl


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_metadata(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    meta: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(path):
        key = rec.get("file") or rec.get("rfilename") or rec.get("path")
        if key:
            meta[key] = rec
    return meta


def build_gate_and_adapters(sources_config: str) -> Tuple[PermissionGate, List[Any], List[Dict]]:
    cfg = load_yaml(sources_config)
    specs = cfg.get("sources", [])
    declarations: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        rights = dict(spec.get("rights", {}))
        if spec["type"] == "local":
            rights["uri_prefix"] = "file://" + os.path.abspath(spec["root"])
        elif spec["type"] == "hf":
            rights["uri_prefix"] = f"hf://datasets/{spec['repo_id']}"
        declarations[spec["name"]] = rights
    gate = PermissionGate(declarations)
    adapters = []
    for spec in specs:
        meta = _load_metadata(spec.get("metadata_file"))
        if spec["type"] == "local":
            adapters.append(LocalSourceAdapter(
                root=spec["root"], gate=gate,
                source_revision=spec.get("source_revision", "local"),
                metadata=meta, probe=spec.get("probe", True)))
        elif spec["type"] == "hf":
            adapters.append(HfSourceAdapter(
                repo_id=spec["repo_id"], gate=gate, revision=spec.get("revision", "main"),
                metadata=meta, allow_patterns=spec.get("allow_patterns")))
        else:
            raise ValueError(f"unknown source type {spec['type']!r}")
    return gate, adapters, specs


def technical_thresholds(audio_profile: Optional[str]) -> TechnicalThresholds:
    if not audio_profile or not os.path.exists(audio_profile):
        return TechnicalThresholds()
    prof = load_yaml(audio_profile).get("technical_thresholds", {})
    known = TechnicalThresholds.__dataclass_fields__.keys()
    return TechnicalThresholds(**{k: v for k, v in prof.items() if k in known})


def source_admission_config(audio_profile: Optional[str]) -> AdmissionConfig:
    if not audio_profile or not os.path.exists(audio_profile):
        return AdmissionConfig()
    prof = load_yaml(audio_profile).get("source_admission", {}).get("clean_hq", {})
    known = AdmissionConfig.__dataclass_fields__.keys()
    return AdmissionConfig(**{k: v for k, v in prof.items() if k in known})


@dataclass
class DetectorSet:
    """Instantiated, configured detector adapters (spec §2.1, §4).

    ``noise`` (DSP hiss/hum/clipping) is always present; ``music`` is an optional
    music-gate detector (e.g. PANNs) run *in addition* to noise. ``availability``
    is derived from the adapters themselves — never asserted by config alone.
    """
    vad: Any
    noise: Any
    music: Optional[Any]
    quality: Any
    speaker: Any
    provenance: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def availability(self) -> Dict[str, bool]:
        return {
            "music": bool(self.music is not None and self.music.available),
            "quality": bool(self.quality.available),
            "speaker": bool(self.speaker.available),
        }


def build_detectors(models_lock: Optional[str]) -> DetectorSet:
    """Instantiate detectors from a models lock, passing each adapter its real
    options (model_path/weights_sha256/license flags). Missing/unverified weights
    => the adapter reports UNAVAILABLE and the dependent gate fails closed."""
    lock: Dict[str, Any] = {}
    if models_lock and os.path.exists(models_lock):
        lock = load_yaml(models_lock).get("detectors", {})

    def opts(section: str, adapter: str, kind: "DetectorKind") -> Dict[str, Any]:
        sec = lock.get(section, {}) or {}
        raw = dict(sec.get(adapter, {}) or {})
        # keep only keys the adapter constructor actually accepts (drop e.g. bare
        # `license:` documentation keys) so a lock stanza never crashes construction.
        try:
            import inspect
            factory = registry._factories[kind.value][adapter]
            params = set(inspect.signature(factory).parameters)
            return {k: v for k, v in raw.items() if k in params}
        except (KeyError, ValueError, TypeError):
            return raw

    vad_name = (lock.get("vad", {}) or {}).get("adapter", "energy")
    vad = registry.create(DetectorKind.VAD, vad_name, **opts("vad", vad_name, DetectorKind.VAD))

    # DSP noise is always available (hiss/hum/clipping); music is a separate gate.
    noise = registry.create(DetectorKind.NOISE, "dsp", **opts("noise", "dsp", DetectorKind.NOISE))
    music = None
    music_name = (lock.get("music", {}) or {}).get("adapter")
    if music_name:
        music = registry.create(DetectorKind.NOISE, music_name,
                                **opts("music", music_name, DetectorKind.NOISE))

    quality = registry.create(DetectorKind.QUALITY, "dnsmos",
                              **opts("quality", "dnsmos", DetectorKind.QUALITY))
    speaker = registry.create(DetectorKind.SPEAKER, "pyannote",
                              **opts("speaker", "pyannote", DetectorKind.SPEAKER))

    ds = DetectorSet(vad=vad, noise=noise, music=music, quality=quality, speaker=speaker)
    ds.provenance = [a.info().to_dict() for a in (vad, noise, music, quality, speaker)
                     if a is not None]
    return ds


def load_policy_config(path: str) -> PolicyConfig:
    return load_policy(path)
