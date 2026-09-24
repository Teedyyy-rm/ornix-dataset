"""Pluggable detector framework (spec §2.1 contracts, §4 stack, Phase 3/4).

Every ML capability (VAD, sound-event/music, speech-quality, speaker/overlap) is
a *pluggable adapter* behind a registry so Ornix is never locked into a single
vendor. An adapter that cannot load its (licensed, checksum-verified) weights
reports ``UNAVAILABLE`` and the policy engine fails the dependent required gate —
it never silently substitutes a different model or a "clean" default.
"""

from .base import (
    Availability,
    AdapterInfo,
    DetectorKind,
    VadAdapter,
    NoiseDetector,
    QualityAdapter,
    SpeakerAdapter,
    registry,
)
from .windowing import WindowPlan, plan_windows, union_intervals, dilate_intervals

# Import adapter modules so their @registry.register decorators run.
from . import vad as _vad  # noqa: E402,F401
from . import music_noise as _music_noise  # noqa: E402,F401
from . import quality as _quality  # noqa: E402,F401
from . import speaker as _speaker  # noqa: E402,F401

__all__ = [
    "Availability",
    "AdapterInfo",
    "DetectorKind",
    "VadAdapter",
    "NoiseDetector",
    "QualityAdapter",
    "SpeakerAdapter",
    "registry",
    "WindowPlan",
    "plan_windows",
    "union_intervals",
    "dilate_intervals",
]
