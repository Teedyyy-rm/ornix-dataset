"""Real ML-weight integration checks (spec §4, Phase 3).

These run ONLY when checksum-pinned weights are present in ``work/models`` (fetch
them with ``scripts/fetch_models.py``). They confirm the adapters load the real
ONNX graphs, honour the sha256 pin, and produce well-formed inference output.
They are skipped in a clean checkout so the default suite stays offline.
"""

import os

import numpy as np
import pytest

from fixtures import synth
from ornix_dataset.detectors.quality import DnsmosAdapter
from ornix_dataset.detectors.vad import SileroVadAdapter

_MODELS = "work/models"
_SILERO = os.path.join(_MODELS, "silero_vad.onnx")
_DNSMOS = os.path.join(_MODELS, "dnsmos_p835.onnx")
_SILERO_SHA = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
_DNSMOS_SHA = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"

onnxruntime = pytest.importorskip("onnxruntime")


@pytest.mark.skipif(not os.path.exists(_SILERO), reason="silero weights not provisioned")
def test_silero_loads_and_runs_real_onnx():
    v = SileroVadAdapter(model_path=_SILERO, weights_sha256=_SILERO_SHA)
    assert v.available, v.info().reason
    # the real graph runs and yields a probability per 512-sample window
    probs = v._probs(synth.speechlike(16000, 2.0, seed=1).astype(np.float32))
    assert probs.size > 0
    assert np.all((probs >= 0.0) & (probs <= 1.0))


@pytest.mark.skipif(not os.path.exists(_SILERO), reason="silero weights not provisioned")
def test_silero_sha_mismatch_fails_closed():
    v = SileroVadAdapter(model_path=_SILERO, weights_sha256="0" * 64)
    assert not v.available
    assert "sha256" in (v.info().reason or "")


def test_silero_probs_feed_v5_context_window():
    """Regression guard for the v5 ONNX contract: each step must feed 576 samples
    (64 carried-over context + 512 new). Feeding the bare 512-sample window makes
    the real model emit ~0 on genuine speech and rejects every clip as NO_SPEECH.
    Uses a stub session so it runs without provisioned weights."""

    widths = []

    class _StubSession:
        def run(self, _outputs, feeds):
            widths.append(int(feeds["input"].shape[-1]))
            return [np.array([[0.9]], dtype=np.float32), feeds["state"]]

    v = SileroVadAdapter.__new__(SileroVadAdapter)
    v._session = _StubSession()
    probs = v._probs(np.zeros(SileroVadAdapter._WIN * 3, dtype=np.float32))
    assert probs.size == 3
    assert widths == [SileroVadAdapter._WIN + SileroVadAdapter._CTX] * 3


@pytest.mark.skipif(not os.path.exists(_DNSMOS), reason="dnsmos weights not provisioned")
def test_dnsmos_discriminates_clean_from_noisy():
    q = DnsmosAdapter(model_path=_DNSMOS, weights_sha256=_DNSMOS_SHA, license_ack=True)
    assert q.available, q.info().reason
    clean = synth.speechlike(24000, 3.0, seed=1)
    noisy = clean * 0.3 + 0.5 * synth.white_noise(24000, 3.0)
    r_clean = q.infer(clean, 24000)
    r_noisy = q.infer(noisy, 24000)
    assert r_clean["status"] == "OK" and r_noisy["status"] == "OK"
    # a real no-reference MOS must score clean speech above heavy white noise
    assert r_clean["sig"] > r_noisy["sig"]


@pytest.mark.skipif(not os.path.exists(_DNSMOS), reason="dnsmos weights not provisioned")
def test_dnsmos_requires_license_ack():
    q = DnsmosAdapter(model_path=_DNSMOS, weights_sha256=_DNSMOS_SHA, license_ack=False)
    assert not q.available
    assert "license" in (q.info().reason or "")
