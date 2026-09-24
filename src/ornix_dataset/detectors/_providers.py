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

    For CPU data-parallelism, ``ORNIX_ORT_INTRA_THREADS`` caps each session's
    intra-op thread pool. Small models (Silero/DNSMOS) scale poorly with many
    intra-op threads, so running many single-/few-threaded workers in parallel is
    far faster than one many-threaded session — the cap prevents N workers from
    each grabbing all cores and thrashing.
    """
    import onnxruntime as ort

    if sess_options is None:
        cap = os.environ.get("ORNIX_ORT_INTRA_THREADS")
        if cap:
            try:
                n = max(1, int(cap))
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = n
                sess_options.inter_op_num_threads = 1
            except Exception:
                sess_options = None

    providers = onnx_providers()
    try:
        return ort.InferenceSession(model_path, sess_options=sess_options,
                                    providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        return ort.InferenceSession(model_path, sess_options=sess_options,
                                    providers=["CPUExecutionProvider"])
