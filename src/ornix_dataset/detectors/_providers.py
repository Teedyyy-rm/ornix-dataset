"""ONNX Runtime execution-provider selection (GPU when available, CPU fallback).

Auto-prefers CUDA when an onnxruntime-gpu build exposes it; falls back to CPU so
CPU-only environments (and the offline test suite) are unaffected. Override with
ORNIX_ORT_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider".
"""

from __future__ import annotations

import os
from typing import List, Optional


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


def make_session(model_path: str, sess_options=None):
    """Create an InferenceSession preferring GPU, falling back to CPU per-model.

    Some models (e.g. Silero VAD's RNN) can fail at CUDA session init on certain
    driver/cuDNN combinations even though CUDA is otherwise usable. Rather than
    letting that take the whole detector UNAVAILABLE, retry CPU-only so the model
    still runs — other models (e.g. DNSMOS) keep using the GPU.
    """
    import onnxruntime as ort

    providers = onnx_providers()
    try:
        return ort.InferenceSession(model_path, sess_options=sess_options,
                                    providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        return ort.InferenceSession(model_path, sess_options=sess_options,
                                    providers=["CPUExecutionProvider"])
