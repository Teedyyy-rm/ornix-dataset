"""ONNX Runtime execution-provider selection (GPU when available, CPU fallback).

Auto-prefers CUDA when an onnxruntime-gpu build exposes it; falls back to CPU so
CPU-only environments (and the offline test suite) are unaffected. Override with
ORNIX_ORT_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider".
"""

from __future__ import annotations

import os
from typing import List


def onnx_providers() -> List[str]:
    override = os.environ.get("ORNIX_ORT_PROVIDERS")
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    providers: List[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers
