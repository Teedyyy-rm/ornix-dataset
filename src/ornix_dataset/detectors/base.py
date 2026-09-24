"""Detector base classes + registry (fail-closed availability)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..contracts.quality import NoiseEvent


class Availability(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class DetectorKind(str, Enum):
    VAD = "vad"
    NOISE = "noise"
    QUALITY = "quality"
    SPEAKER = "speaker"


@dataclass
class AdapterInfo:
    name: str
    kind: DetectorKind
    availability: Availability
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    weights_sha256: Optional[str] = None
    license: Optional[str] = None
    is_heuristic: bool = False  # True => cannot satisfy a required ML gate alone
    reason: Optional[str] = None  # why UNAVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["availability"] = self.availability.value
        return d


class _Adapter:
    kind: DetectorKind = None  # type: ignore

    def info(self) -> AdapterInfo:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return self.info().availability == Availability.AVAILABLE


class VadAdapter(_Adapter):
    kind = DetectorKind.VAD

    def infer(self, mono16k: np.ndarray, sr: int) -> List[List[float]]:
        """Return speech intervals [[start_s, end_s], ...] in analysis-view seconds."""
        raise NotImplementedError


class NoiseDetector(_Adapter):
    kind = DetectorKind.NOISE

    def infer(self, mono: np.ndarray, sr: int, speech_intervals: List[List[float]]) -> List[NoiseEvent]:
        raise NotImplementedError


class QualityAdapter(_Adapter):
    kind = DetectorKind.QUALITY

    def infer(self, mono: np.ndarray, sr: int) -> Dict[str, Any]:
        """Return {'sig','bak','ovrl','status','model_id'} — status UNKNOWN if unsupported."""
        raise NotImplementedError


class SpeakerAdapter(_Adapter):
    kind = DetectorKind.SPEAKER

    def infer(self, mono: np.ndarray, sr: int) -> Dict[str, Any]:
        """Return {'overlap_intervals','n_speakers','status'}."""
        raise NotImplementedError


@dataclass
class _Registry:
    _factories: Dict[str, Dict[str, Callable[..., _Adapter]]] = field(default_factory=dict)

    def register(self, kind: DetectorKind, name: str) -> Callable:
        def deco(factory: Callable[..., _Adapter]) -> Callable[..., _Adapter]:
            self._factories.setdefault(kind.value, {})[name] = factory
            return factory
        return deco

    def create(self, kind: DetectorKind, name: str, **kwargs: Any) -> _Adapter:
        try:
            factory = self._factories[kind.value][name]
        except KeyError:
            raise KeyError(f"no {kind.value} adapter named {name!r}; "
                           f"available: {self.names(kind)}")
        return factory(**kwargs)

    def names(self, kind: DetectorKind) -> List[str]:
        return sorted(self._factories.get(kind.value, {}).keys())


registry = _Registry()
