"""Live HF download integration (opt-in ONLY).

Runs real bytes against the Hub. Skipped unless ORNIX_HF_LIVE=1 AND network
is reachable. Bounded to ONE tiny file (README, ~520 B) of the public HF
test dataset at a pinned commit; isolated temp caches; never touches the
user's real HF cache. Not redistribution: bytes stay in a temp dir and are
used only to prove the downloader end-to-end.
"""

import json
import os
import subprocess
import sys

import pytest

LIVE = os.environ.get("ORNIX_HF_LIVE") == "1"
REPO = "hf-internal-testing/librispeech_asr_dummy"

pytestmark = pytest.mark.skipif(not LIVE, reason="needs ORNIX_HF_LIVE=1")


def test_live_single_small_file_workers_1(tmp_path):
    work = str(tmp_path)
    hf_home = os.path.join(work, "hf")
    script = (
        "import json, os, sys\n"
        f"os.environ['HF_HOME'] = {hf_home!r}\n"
        "os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'\n"
        "import sys as _s\n"
        f"_s.path.insert(0, {os.path.join('src')!r})\n"
        "from huggingface_hub import HfApi\n"
        "from ornix_dataset.ingestion.hf_downloader import (\n"
        "    HfBatchDownloader, DownloadConfig, build_inventory)\n"
        f"api = HfApi(token=os.environ.get('HF_TOKEN'))\n"
        f"pinned = api.resolve_revision({REPO!r}, revision='main',\n"
        "                              repo_type='dataset').resolved\n"
        f"inv = build_inventory(api, {REPO!r}, pinned, allow=['README*'],\n"
        "                      audio_exts={'.md'})\n"
        "assert len(inv) == 1 and inv[0].size < 10000\n"
        "cfg = DownloadConfig(file_workers=1)\n"
        f"dl = HfBatchDownloader({REPO!r}, pinned, {work!r} + '/staging', cfg=cfg)\n"
        "res = dl.run(inv)\n"
        "print(json.dumps({'pinned': pinned,\n"
        "                'sha256': res[0].sha256, 'size': res[0].size,\n"
        "                'metrics': dl.metrics.to_dict()}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=600,
                          cwd=os.path.dirname(os.path.dirname(
                              os.path.dirname(os.path.abspath(__file__)))),
                          env={**os.environ, "PYTHONPATH": "src"})
    assert proc.returncode == 0, proc.stderr[-1500:]
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(payload["pinned"]) == 40
    assert payload["size"] > 0
    assert payload["metrics"]["n_downloaded"] == 1
    assert payload["metrics"]["network_bytes"] == payload["size"]
