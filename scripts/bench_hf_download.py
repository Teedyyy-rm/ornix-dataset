"""HF download benchmark harness (read-only, bounded, isolated caches).

Measures the Ornix batch downloader against a pinned dataset revision with
per-config SUBPROCESSES (HF_* env is read at import time) and ISOLATED temp
caches on the SAME device class as production. NEVER touches the user's real
HF cache. Bounded by --max-files/--max-bytes. Writes a JSON report with full
provenance (SDK versions, hardware, Ornix + dataset commits).

Usage:
    python scripts/bench_hf_download.py --repo-id <id> --revision <sha|branch> \\
        --allow '*.parquet' --max-files 2 --max-bytes 20000000 \\
        --workers 1 4 --out work/bench/report.json [--live]
Without --live only the metadata (dry-run inventory) stage runs: zero bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def net_counters():
    """rx/tx bytes over non-loopback interfaces (Linux /proc)."""
    rx = tx = 0
    try:
        with open("/proc/net/dev") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                iface, rest = line.split(":", 1)
                iface = iface.strip()
                if iface == "lo":
                    continue
                parts = rest.split()
                rx += int(parts[0])
                tx += int(parts[8])
    except OSError:
        pass
    return rx, tx


WORKER_SRC = r"""
import json, os, sys
# ENV FIRST: huggingface_hub freezes cache paths at import time. The cfg path
# arrives via argv precisely so no SDK import precedes these assignments.
_cfg = json.load(open(sys.argv[1]))
os.environ["HF_HOME"] = _cfg["hf_home"]
os.environ["HF_HUB_CACHE"] = _cfg["hf_hub_cache"]
os.environ["HF_XET_CACHE"] = _cfg["hf_xet_cache"]
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import time
sys.path.insert(0, @@SRC@@)
from huggingface_hub import HfApi
import huggingface_hub
import hf_xet  # noqa: F401  (prove import surface)
from ornix_dataset.ingestion.hf_downloader import (
    HfBatchDownloader, DownloadConfig, build_inventory, http_env_snapshot)

cfg = _cfg

api = HfApi(token=cfg.get("token") or os.environ.get("HF_TOKEN"))
pinned = api.resolve_revision(cfg["repo_id"], revision=cfg["revision"],
                              repo_type="dataset").resolved
t0 = time.monotonic()
inv = build_inventory(api, cfg["repo_id"], pinned, allow=cfg.get("allow"),
                      ignore=cfg.get("ignore"), audio_exts=set(cfg["exts"]),
                      allow_empty=True)
t_inv = time.monotonic() - t0
inv = [i for i in inv if i.size <= cfg["max_file_bytes"]][:cfg["max_files"]]
kept = sum(i.size for i in inv)
dl_cfg = DownloadConfig.from_dict(cfg["download"]).effective()
dl = HfBatchDownloader(cfg["repo_id"], pinned, cfg["staging"], cfg=dl_cfg)
t1 = time.monotonic()
res = dl.run(inv) if cfg["live"] else []
t2 = time.monotonic()
print(json.dumps({
    "pinned": pinned, "n_inventory": len(inv), "kept_bytes": kept,
    "t_inventory_s": round(t_inv, 3), "t_run_s": round(t2 - t1, 3),
    "metrics": dl.metrics.to_dict(),
    "staged": [r.staged_path for r in res],
    "hub_version": huggingface_hub.__version__,
    "env": http_env_snapshot(),
}))
"""


def run_config(base, dl_cfg, live, label):
    work = tempfile.mkdtemp(prefix="ornix-bench-")
    hf_home = os.path.join(work, "hf")
    staging = os.path.join(work, "staging")
    cfg = dict(base, hf_home=hf_home,
               hf_hub_cache=os.path.join(hf_home, "hub"),
               hf_xet_cache=os.path.join(hf_home, "xet"),
               staging=staging, download=dl_cfg, live=live)
    cfg_path = os.path.join(work, "cfg.json")
    json.dump({k: v for k, v in cfg.items() if k != "token"}, open(cfg_path, "w"))
    # token travels via env only (never in the file)
    env = dict(os.environ)
    if cfg.get("token"):
        env["HF_TOKEN"] = cfg["token"]
    src = os.path.join(REPO_ROOT, "src")
    worker = WORKER_SRC.replace("@@SRC@@", repr(src))
    rx0, tx0 = net_counters()
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", worker, cfg_path],
                          capture_output=True, text=True, env=env, timeout=1800)
    wall = time.monotonic() - t0
    rx1, tx1 = net_counters()
    if proc.returncode != 0:
        return {"label": label, "ok": False, "wall_s": round(wall, 3),
                "stderr_tail": proc.stderr[-2000:],
                "net_rx_bytes": rx1 - rx0, "net_tx_bytes": tx1 - tx0,
                "workdir": work}
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload.update({"label": label, "ok": True, "wall_s": round(wall, 3),
                    "net_rx_bytes": rx1 - rx0, "net_tx_bytes": tx1 - tx0,
                    "workdir": work, "download_cfg": dl_cfg})
    return payload


def cpu_info():
    try:
        with open("/proc/cpuinfo") as fh:
            models = {l.split(":", 1)[1].strip() for l in fh
                      if l.startswith("model name")}
        return {"model": sorted(models)[:1], "count": os.cpu_count()}
    except OSError:
        return {"model": [platform.processor()], "count": os.cpu_count()}


def mem_info():
    try:
        out = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                if k.strip() in ("MemTotal", "MemAvailable"):
                    out[k.strip()] = int(v.split()[0]) * 1024
        return out
    except OSError:
        return {}


def disk_ceiling(path, size_mb=512):
    """Local sequential-write ceiling on the device holding `path` (MB/s)."""
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, ".bench-write.bin")
    data = os.urandom(1 << 20)
    t0 = time.monotonic()
    with open(tmp, "wb") as fh:
        for _ in range(size_mb):
            fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    dt = time.monotonic() - t0
    os.remove(tmp)
    return round(size_mb / dt, 2)


def hash_ceiling(mb=256):
    import hashlib
    data = os.urandom(1 << 20)
    t0 = time.monotonic()
    h = hashlib.sha256()
    for _ in range(mb):
        h.update(data)
    h.hexdigest()
    dt = time.monotonic() - t0
    return round(mb / dt, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ornix HF download benchmark")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--allow", nargs="*", default=None)
    ap.add_argument("--ignore", nargs="*", default=None)
    ap.add_argument("--exts", nargs="*", default=[".wav", ".flac", ".mp3",
                                                 ".ogg", ".opus", ".m4a", ".aac"])
    ap.add_argument("--max-files", type=int, default=4)
    ap.add_argument("--max-bytes", type=int, default=20_000_000)
    ap.add_argument("--max-file-bytes", type=int, default=50_000_000)
    ap.add_argument("--workers", nargs="*", type=int, default=[1, 4])
    ap.add_argument("--out", required=True)
    ap.add_argument("--live", action="store_true",
                    help="actually fetch bytes (bounded); without it: dry-run only")
    ap.add_argument("--token-env", default="HF_TOKEN")
    args = ap.parse_args(argv)

    import huggingface_hub
    try:
        import hf_xet  # noqa: F401
        xet = True
    except Exception:
        xet = False
    base = {"repo_id": args.repo_id, "revision": args.revision,
            "allow": args.allow, "ignore": args.ignore, "exts": args.exts,
            "max_files": args.max_files, "max_bytes": args.max_bytes,
            "max_file_bytes": args.max_file_bytes,
            "token": os.environ.get(args.token_env)}
    report = {
        "provenance": {
            "huggingface_hub": huggingface_hub.__version__, "hf_xet": xet,
            "python": platform.python_version(), "os": platform.platform(),
            "cpu": cpu_info(), "mem_bytes": mem_info(),
        },
        "base": {k: v for k, v in base.items() if k != "token"},
        "runs": [],
    }
    staging_probe = tempfile.mkdtemp(prefix="ornix-bench-disk-")
    report["ceilings"] = {
        "disk_write_MBps_staging_device": disk_ceiling(staging_probe),
        "sha256_ram_MBps": hash_ceiling(),
    }
    for w in args.workers:
        dl_cfg = {"file_workers": w, "min_free_disk_bytes": 1_000_000_000}
        try:
            report["runs"].append(run_config(base, dl_cfg, args.live,
                                             label=f"workers={w}"))
        except subprocess.TimeoutExpired:
            report["runs"].append({"label": f"workers={w}", "ok": False,
                                   "error": "timeout>1800s"})
        if not args.live:
            break  # one dry-run is enough
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
