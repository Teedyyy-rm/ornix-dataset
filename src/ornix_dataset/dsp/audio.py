"""In-memory audio buffer (float32, shape [n_samples] mono or [n_samples, ch])."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AudioBuffer:
    samples: np.ndarray  # float32/float64, mono 1-D or [n, channels]
    sample_rate: int

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / float(self.sample_rate) if self.sample_rate else 0.0

    def to_mono(self, policy: str = "mean") -> "AudioBuffer":
        if self.samples.ndim == 1:
            return self
        if policy == "mean":
            mono = self.samples.mean(axis=1)
        elif policy == "first":
            mono = self.samples[:, 0]
        else:
            raise ValueError(f"unknown downmix policy {policy!r}")
        return AudioBuffer(mono.astype(np.float32, copy=False), self.sample_rate)
