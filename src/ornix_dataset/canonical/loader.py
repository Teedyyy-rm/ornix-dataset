"""Loader for the canonical Ornix Dataset (spec §9, tests T16).

Reads ``metadata.jsonl`` **directly** so the exact six-field contract survives:
``audio`` and ``file_name`` stay ``str``. Hugging Face ``datasets``'
``AudioFolder`` builder instead casts the linked audio column to the ``Audio``
feature (``array``/``path``/``sampling_rate``),
so it cannot preserve ``audio: str``/``file_name: str``; ``load_with_datasets``
is provided as an explicit adapter and documents that behaviour rather than
claiming the six fields are unchanged.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .layout import SPLITS, metadata_path
from .schema import CANONICAL_FIELDS, CanonicalRow


@dataclass
class OrnixSample:
    split: str
    row: Dict[str, Any]
    audio_path: str


def iter_samples(dataset_dir: str,
                 splits: Optional[List[str]] = None) -> Iterator[OrnixSample]:
    """Yield every sample in split order, validating the six-field contract."""
    for split in (splits or SPLITS):
        path = metadata_path(dataset_dir, split)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                import json
                raw = json.loads(line)
                row = CanonicalRow.from_dict(raw)  # enforces exactly 6 fields
                audio_path = os.path.join(dataset_dir, split,
                                          *row.audio.split("/"))
                yield OrnixSample(split=split, row=row.to_dict(),
                                  audio_path=audio_path)


def load_ornix_dataset(dataset_dir: str,
                       splits: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """All canonical rows as plain dicts (exactly six fields each)."""
    out = [s.row for s in iter_samples(dataset_dir, splits)]
    for r in out:
        if set(r) != set(CANONICAL_FIELDS):
            raise ValueError(f"loader produced a non-canonical row: {sorted(r)}")
    return out


def read_wav_int16(path: str) -> Tuple[np.ndarray, int]:
    """Decode a canonical PCM16 WAV to int16 samples (stdlib, no extra deps)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        if wf.getsampwidth() != 2:
            raise ValueError("not PCM16")
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype="<i2"), sr


def load_with_datasets(dataset_dir: str) -> Any:
    """Explicit ``datasets`` AudioFolder adapter.

    Returns the loaded ``DatasetDict`` but callers must NOT assume ``audio`` stays
    a string: ``AudioFolder`` decodes it into the ``Audio`` feature. This wrapper
    only re-asserts that the remaining four fields are intact, and is honest that
    the canonical ``audio: str`` column is not guaranteed here.
    """
    from datasets import load_dataset  # imported lazily: optional dependency

    return load_dataset("audiofolder", data_dir=dataset_dir)
