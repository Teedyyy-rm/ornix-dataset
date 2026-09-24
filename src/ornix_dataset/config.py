"""Config loading: sources, permission gate, models lock, pipeline assembly."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .curation.policy import PolicyConfig, load_policy
from .detectors import DetectorKind, registry
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


def resolve_detectors(models_lock: Optional[str]) -> Tuple[str, str, Dict[str, bool]]:
    """Return (vad_name, noise_name, availability_map) from a models lock."""
    vad_name, noise_name = "energy", "dsp"
    avail = {"music": False, "quality": False, "speaker": False}
    if not models_lock or not os.path.exists(models_lock):
        return vad_name, noise_name, avail
    lock = load_yaml(models_lock).get("detectors", {})
    vad_name = lock.get("vad", {}).get("adapter", "energy")
    noise_name = lock.get("noise", {}).get("adapter", "dsp")
    # availability determined by actually instantiating adapters (fail-closed)
    if noise_name == "panns":
        opts = lock.get("noise", {}).get("panns", {})
        avail["music"] = registry.create(DetectorKind.NOISE, "panns", **opts).available
    q = lock.get("quality", {})
    if q.get("adapter") == "dnsmos" and q.get("dnsmos"):
        avail["quality"] = registry.create(DetectorKind.QUALITY, "dnsmos", **q["dnsmos"]).available
    s = lock.get("speaker", {})
    if s.get("adapter") == "pyannote" and s.get("pyannote"):
        avail["speaker"] = registry.create(DetectorKind.SPEAKER, "pyannote", **s["pyannote"]).available
    return vad_name, noise_name, avail


def load_policy_config(path: str) -> PolicyConfig:
    return load_policy(path)
