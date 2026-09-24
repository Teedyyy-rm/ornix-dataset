#!/usr/bin/env python3
"""run_ornix_campaign.py — drive the full multi-dataset Ornix campaign.

For each dataset in configs/ornix_campaign.manifest.yaml this streams a bounded
page of clips (extract_hf_audio.py) into a temp corpus, runs the standard
ingest -> qc -> canonical-export chain (appending into ONE unified canonical
tree + persistent identity/speaker state), then DELETES the page before fetching
the next one. Disk-bounded, resumable (a checkpoint records each dataset's cursor),
and fail-closed (rights are declared per-dataset in the manifest; the tool never
infers them, and canonical export drops anything not redistributable).

Subcommands:
  run     drive the campaign (resumable; --only NAME to limit; --max-chunks N)
  status  print the checkpoint + unified-tree stats

This orchestrates only regenerable intermediate work; it never deletes the
canonical tree, the state dir, or the checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _yaml():
    import yaml
    return yaml


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as fh:
        return _yaml().safe_load(fh)


def free_gib(path):
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return float("inf")

def log(state_dir, event):
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
    line = json.dumps(event, ensure_ascii=False)
    with open(os.path.join(state_dir, "progress.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def load_ckpt(state_dir):
    p = os.path.join(state_dir, "campaign_state.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"datasets": {}}


def save_ckpt(state_dir, ckpt):
    p = os.path.join(state_dir, "campaign_state.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ckpt, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def run(cmd, env=None):
    """Run a subprocess, streaming nothing; return (rc, stdout, stderr)."""
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def parse_result_line(stdout):
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return None


def parse_last_json(stdout):
    """The CLI prints an indented JSON object last; grab the final {...} block."""
    buf, depth, started, blocks = [], 0, False, []
    for ch in stdout:
        if ch == "{":
            depth += 1
            started = True
        if started:
            buf.append(ch)
        if ch == "}":
            depth -= 1
            if depth == 0 and started:
                blocks.append("".join(buf))
                buf, started = [], False
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    return None

def write_sources_yaml(ds, extract_dir, sha, path):
    r = ds["rights"]
    spec = {
        "sources": [{
            "name": ds["name"],
            "type": "local",
            "root": os.path.abspath(extract_dir),
            "source_revision": f"{ds['repo']}@{sha}",
            "probe": True,
            "metadata_file": os.path.abspath(os.path.join(extract_dir, "metadata.jsonl")),
            "rights": {
                "rights_status": r["status"],
                "redistribution_permitted": bool(r["redistribute"]),
                "commercial_training_permitted": str(r.get("commercial", "false")),
                "attribution_required": True,
                "source_license": r.get("license", "UNSPECIFIED-OPERATOR-DECLARED"),
                "note": r.get("note", ""),
            },
        }]
    }
    with open(path, "w", encoding="utf-8") as fh:
        _yaml().safe_dump(spec, fh, allow_unicode=True, sort_keys=False)


def extractor_cmd(ds, defaults, out_dir, cursor, chunk_rows):
    py = sys.executable
    cmd = [py, os.path.join(HERE, "extract_hf_audio.py"),
           "--repo", ds["repo"], "--out", out_dir, "--format", ds["format"],
           "--limit", str(chunk_rows),
           "--start-shard", str(cursor.get("next_shard", 0)),
           "--start-rg", str(cursor.get("next_rg", 0)),
           "--audio-col", ds.get("audio_col", defaults.get("audio_col", "audio")),
           "--text-col", ds.get("text_col", "text"),
           "--language", ds.get("language", defaults.get("language", "vi"))]
    sp = ds.get("speaker", {}) or {}
    if sp.get("mode") == "fixed":
        cmd += ["--speaker-ref", sp["ref"]]
    elif sp.get("mode") == "column":
        cmd += ["--speaker-col", sp.get("col", "speaker")]
    if ds["format"] == "parquet":
        cmd += ["--config-dir", str(ds.get("parquet_prefix", "data"))]
    if ds["format"] == "loose":
        cmd += ["--loose-audio-prefix", ds.get("loose_audio_prefix", ""),
                "--loose-meta-kind", ds.get("loose_meta_kind", "csv"),
                "--file-col", ds.get("file_col", "file_name")]
        if ds.get("loose_meta"):
            cmd += ["--loose-meta", ds["loose_meta"]]
    return cmd

def prune_hf_cache(repo):
    """Delete a repo's cached blobs (loose/arrow downloads) to bound disk."""
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    slug = "datasets--" + repo.replace("/", "--")
    d = os.path.join(home, "hub", slug)
    shutil.rmtree(d, ignore_errors=True)


def run_chunk(ds, defaults, args, cursor, chunk_idx, env):
    """Extract + ingest + qc + canonical-export one page. Returns (result, stats)."""
    name = ds["name"]
    extract_dir = os.path.join(args.workdir, "campaign_extract", f"{name}_c{chunk_idx:05d}")
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)
    src_yaml = os.path.join(extract_dir, "_sources.yaml")
    run_id = f"{name}_c{chunk_idx:05d}"
    stats = {"written": 0, "ingested": 0, "accepted": 0, "exported": 0, "blocked": 0}
    try:
        rc, out, err = run(extractor_cmd(ds, defaults, extract_dir, cursor, args.chunk_rows), env)
        res = parse_result_line(out)
        if rc != 0 or res is None:
            return None, {"error": f"extract rc={rc}: {(err or out)[-400:]}"}
        stats["written"] = res["written"]
        if res["written"] == 0:
            return res, stats
        write_sources_yaml(ds, extract_dir, res["sha"], src_yaml)
        rc, out, err = run([sys.executable, "-m", "ornix_dataset.cli", "ingest",
                            "--config", src_yaml, "--run-id", run_id,
                            "--workdir", args.workdir], env)
        if rc != 0:
            return res, {**stats, "error": f"ingest rc={rc}: {(err or out)[-400:]}"}
        j = parse_last_json(out) or {}
        stats["ingested"] = j.get("ingested", 0)
        rc, out, err = run([sys.executable, "-m", "ornix_dataset.cli", "qc",
                            "--run-id", run_id, "--policy", args.policy,
                            "--models-lock", args.models_lock,
                            "--audio-profile", args.audio_profile,
                            "--workdir", args.workdir], env)
        if rc != 0:
            return res, {**stats, "error": f"qc rc={rc}: {(err or out)[-400:]}"}
        j = parse_last_json(out) or {}
        stats["accepted"] = j.get("accepted", j.get("n_accepted", 0))
        rc, out, err = run([sys.executable, "-m", "ornix_dataset.cli", "canonical", "export",
                            "--run-id", run_id, "--dataset", args.dataset,
                            "--state", args.state, "--workdir", args.workdir], env)
        j = parse_last_json(out) or {}
        stats["exported"] = j.get("n_rows", 0)
        stats["blocked"] = j.get("n_blocked", 0)
        if rc not in (0, 3):
            stats["error"] = f"canonical rc={rc}: {(err or out)[-400:]}"
        return res, stats
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(args.workdir, "runs", run_id), ignore_errors=True)
        if ds["format"] in ("loose", "arrow"):
            prune_hf_cache(ds["repo"])

def process_dataset(ds, defaults, args, ckpt, env):
    name = ds["name"]
    st = ckpt["datasets"].setdefault(name, {
        "status": "pending", "next_shard": 0, "next_rg": 0, "chunks": 0,
        "written": 0, "exported": 0, "blocked": 0})
    if st["status"] == "done":
        log(args.state, {"dataset": name, "event": "skip-done"})
        return
    st["status"] = "running"
    save_ckpt(args.state, ckpt)
    log(args.state, {"dataset": name, "event": "start", "cursor": [st["next_shard"], st["next_rg"]]})
    made = 0
    while True:
        fg = free_gib(args.state)
        if fg < args.min_free_gib:
            st["status"] = "paused-disk"
            log(args.state, {"dataset": name, "event": "pause-disk", "free_gib": round(fg, 1)})
            save_ckpt(args.state, ckpt)
            return
        res, stats = run_chunk(ds, defaults, args, st, st["chunks"], env)
        st["chunks"] += 1
        made += 1
        if stats.get("error") and res is None:
            st["status"] = "error"
            st["error"] = stats["error"]
            log(args.state, {"dataset": name, "event": "error", "detail": stats["error"]})
            save_ckpt(args.state, ckpt)
            return
        st["written"] += stats.get("written", 0)
        st["exported"] += stats.get("exported", 0)
        st["blocked"] += stats.get("blocked", 0)
        if res is not None:
            st["next_shard"] = res.get("next_shard", st["next_shard"])
            st["next_rg"] = res.get("next_rg", st["next_rg"])
        log(args.state, {"dataset": name, "event": "chunk", "chunk": st["chunks"],
                         "free_gib": round(free_gib(args.state), 1), **stats,
                         "cursor": [st["next_shard"], st["next_rg"]]})
        save_ckpt(args.state, ckpt)
        if res is not None and res.get("exhausted"):
            st["status"] = "done"
            log(args.state, {"dataset": name, "event": "done", "written": st["written"],
                             "exported": st["exported"], "blocked": st["blocked"]})
            save_ckpt(args.state, ckpt)
            return
        if args.max_chunks and made >= args.max_chunks:
            st["status"] = "paused-maxchunks"
            log(args.state, {"dataset": name, "event": "pause-maxchunks"})
            save_ckpt(args.state, ckpt)
            return

def cmd_run(args):
    man = load_manifest(args.manifest)
    defaults = man.get("defaults", {})
    if args.chunk_rows is None:
        args.chunk_rows = int(defaults.get("chunk_rows", 4000))
    if args.min_free_gib is None:
        args.min_free_gib = float(defaults.get("min_free_gib", 25))
    os.makedirs(args.state, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)
    env = dict(os.environ)
    try:
        import hf_transfer  # noqa: F401
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster bulk shard downloads
    except Exception:
        pass
    # Make GPU (CUDAExecutionProvider) usable when CUDA libs come from pip wheels
    # (nvidia-*-cu12): add their lib dirs to LD_LIBRARY_PATH so onnxruntime-gpu can
    # load libcublas/libcudnn/etc. Harmless on CPU-only boxes (no nvidia package).
    try:
        import glob
        import nvidia
        base = os.path.dirname(nvidia.__file__)
        libdirs = sorted({os.path.dirname(p)
                          for p in glob.glob(os.path.join(base, "*", "lib", "*.so*"))})
        if libdirs:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(libdirs + ([existing] if existing else []))
    except Exception:
        pass
    ckpt = load_ckpt(args.state)
    datasets = man["datasets"]
    if args.only:
        wanted = set(args.only.split(","))
        datasets = [d for d in datasets if d["name"] in wanted]
    log(args.state, {"event": "campaign-start", "datasets": [d["name"] for d in datasets],
                     "chunk_rows": args.chunk_rows, "dataset_dir": args.dataset})
    for ds in datasets:
        process_dataset(ds, defaults, args, ckpt, env)
    log(args.state, {"event": "campaign-end"})
    cmd_status(args)
    return 0


def cmd_status(args):
    ckpt = load_ckpt(args.state)
    rows = 0
    for split in ("train", "validation", "test"):
        mp = os.path.join(args.dataset, split, "metadata.jsonl")
        if os.path.exists(mp):
            with open(mp, "r", encoding="utf-8") as fh:
                rows += sum(1 for _ in fh)
    summary = {"canonical_rows": rows, "free_gib": round(free_gib(args.state), 1),
               "datasets": {}}
    for name, st in ckpt.get("datasets", {}).items():
        summary["datasets"][name] = {k: st.get(k) for k in
                                     ("status", "chunks", "written", "exported", "blocked", "error")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", default=os.path.join(REPO, "configs/ornix_campaign.manifest.yaml"))
    common.add_argument("--dataset", default=os.path.join(REPO, "work/Ornix-Datasets"))
    common.add_argument("--state", default=os.path.join(REPO, "work/ornix-campaign-state"))
    common.add_argument("--workdir", default=os.path.join(REPO, "work"))

    pr = sub.add_parser("run", parents=[common])
    pr.add_argument("--policy", default=os.path.join(REPO, "configs/quality_policy.pilot.yaml"))
    pr.add_argument("--models-lock", default=os.path.join(REPO, "configs/models.lock.fpt.yaml"))
    pr.add_argument("--audio-profile", default=os.path.join(REPO, "configs/audio_profile.yaml"))
    pr.add_argument("--chunk-rows", type=int, default=None)
    pr.add_argument("--min-free-gib", type=float, default=None)
    pr.add_argument("--only", default=None, help="comma-separated dataset names")
    pr.add_argument("--max-chunks", type=int, default=0, help="cap chunks per dataset (0=all)")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("status", parents=[common])
    ps.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

