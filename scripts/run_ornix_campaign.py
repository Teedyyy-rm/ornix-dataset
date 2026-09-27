#!/usr/bin/env python3
"""run_ornix_campaign.py — drive the full multi-dataset Ornix campaign.

For each dataset in configs/ornix_campaign.manifest.yaml this downloads the
WHOLE dataset (extract_hf_audio.py) into a temp corpus, runs the standard
ingest -> qc -> canonical-export chain (appending into ONE unified canonical
tree + persistent identity/speaker state), then DELETES the temp corpus.
Dataset-level resume (a checkpoint records each dataset's status), and
fail-closed (rights are declared per-dataset in the manifest; the tool never
infers them, and canonical export drops anything not redistributable).

Subcommands:
  run     drive the campaign (resumable; --only NAME to limit)
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
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC_DIR = os.path.join(REPO, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Guards shared writes (progress log + checkpoint) when datasets run on a worker
# pool. Canonical export serializes itself via an flock on the identity state, so
# it needs no lock here; only these in-process shared files do.
_IO_LOCK = threading.Lock()


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
    with _IO_LOCK:
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
    with _IO_LOCK:
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

def _has_gpu() -> bool:
    """True when a CUDA GPU is usable for inference (torch or onnxruntime)."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return True
    except Exception:
        pass
    if shutil.which("nvidia-smi") is not None:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                 text=True, timeout=10)
            if out.returncode == 0 and "GPU" in (out.stdout or ""):
                return True
        except Exception:
            pass
    return False


def resolve_parallelism(n_datasets, workers_arg, ort_arg):
    """Resolve workers/threads to saturate ~100% CPU (GPU-aware).

    0 (the default) means auto. On GPU boxes each ORT session is capped to 1
    intra-op thread (inference lives on the GPU) and dataset workers scale up
    to cpu_count; on CPU-only boxes workers = cpu // ort_threads so
    workers * threads ~= cores.
    """
    cpu = os.cpu_count() or 8
    gpu = _has_gpu()
    if ort_arg and ort_arg > 0:
        ort_threads = max(1, ort_arg)
    else:
        ort_threads = 1 if gpu else min(2, cpu)
    if workers_arg and workers_arg > 0:
        workers = max(1, workers_arg)
    elif gpu:
        workers = min(max(2, cpu), max(1, n_datasets or 1))
    else:
        workers = min(max(2, cpu // max(1, ort_threads)),
                      max(1, n_datasets or 1))
    probe_workers = max(2, (2 * cpu) // max(1, workers))
    return {"workers": workers, "ort_threads": ort_threads,
            "probe_workers": probe_workers, "cpu": cpu, "gpu": gpu}


_PIPE_CACHE = {}
_PIPE_LOCK = threading.Lock()


def get_campaign_pipeline(workdir, policy_path, models_lock, audio_profile_path):
    """Process-wide cached OrnixPipeline (model server: warm ONNX sessions).

    Silero/DNSMOS sessions load once per worker process and are reused across
    all datasets instead of reloading (+ sha re-verify) per dataset.
    """
    key = (os.path.abspath(workdir), os.path.abspath(policy_path or ""),
           os.path.abspath(models_lock) if models_lock else "",
           os.path.abspath(audio_profile_path) if audio_profile_path else "")
    with _PIPE_LOCK:
        hit = _PIPE_CACHE.get(key)
    if hit is not None:
        return hit
    from ornix_dataset.config import (load_policy_config,
                                      source_admission_config,
                                      technical_thresholds)
    from ornix_dataset.detectors.model_server import get_detectors
    from ornix_dataset.pipeline import OrnixPipeline
    policy = load_policy_config(policy_path)
    detectors = get_detectors(models_lock)
    pipe = OrnixPipeline(workdir, policy,
                         tech=technical_thresholds(audio_profile_path),
                         admission=source_admission_config(audio_profile_path),
                         detectors=detectors)
    with _PIPE_LOCK:
        _PIPE_CACHE[key] = pipe
    return pipe


def _inprocess_enabled():
    v = os.environ.get("ORNIX_CAMPAIGN_INPROCESS", "1").strip().lower()
    return v not in ("0", "false", "no", "off", "")


def run_dataset_local(ds, defaults, args, push=None):
    """In-process full dataset: extract ALL + ingest + qc + canonical-export.

    Same stage order and stats contract as run_dataset_subprocess, but every
    stage runs in this worker process, so the cached pipeline keeps model
    sessions warm across datasets.
    """
    # NOTE: every stage below is a direct object call — never stdout capture.
    # redirect_stdout is process-global and corrupts results when dataset
    # workers run on threads.
    name = ds["name"]
    extract_dir = os.path.join(args.workdir, "campaign_extract", name)
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)
    src_yaml = os.path.join(extract_dir, "_sources.yaml")
    run_id = name
    stats = {"written": 0, "ingested": 0, "accepted": 0, "exported": 0, "blocked": 0}
    try:
        import extract_hf_audio
        from ornix_dataset import cli as cli_mod
        from ornix_dataset.audit.events import AuditLog
        from ornix_dataset.canonical.bridge import sample_from_accepted_row
        from ornix_dataset.canonical.normalize import normalize_samples
        from ornix_dataset.config import build_gate_and_adapters
        from ornix_dataset.contracts.source import SourceRecord
        from ornix_dataset.curation.policy import PolicyConfig
        from ornix_dataset.pipeline import AnalyzeResult, OrnixPipeline, RunPaths
        from ornix_dataset.util.jsonl import read_jsonl
        ecmd = extractor_cmd(ds, defaults, extract_dir, args.loose_workers)
        try:
            res = extract_hf_audio.run_extraction(ecmd[2:])
        except SystemExit as e:
            return None, {"error": f"extract exit={e.code}"}
        except Exception as e:
            return None, {"error": f"extract failed: {e}"}
        stats["written"] = res["written"]
        if res["written"] == 0:
            return res, stats
        write_sources_yaml(ds, extract_dir, res["sha"], src_yaml)
        paths = RunPaths.create(args.workdir, run_id)
        audit = AuditLog(paths.audit, run_id=run_id)
        gate, adapters, _specs = build_gate_and_adapters(src_yaml)
        stub = PolicyConfig(policy_version="ingest-only",
                            required_checks=["rights_ok"])
        ing_pipe = OrnixPipeline(args.workdir, stub)
        total = 0
        for adapter in adapters:
            total += len(ing_pipe.ingest_and_stage(adapter, paths, audit))
        stats["ingested"] = total
        if total == 0 and res["written"] > 0:
            # Loud, not silent: extracted files exist but none entered QC
            # (e.g. mp3 content skipped by wav-only mode). The dataset would
            # otherwise finish "done" with zero rows and no explanation.
            log(args.state, {"dataset": name, "event": "ingest-empty",
                             "written": res["written"],
                             "note": "0 ingested; non-wav sources are skipped "
                             "when ORNIX_WAV_ONLY=1 (set 0 to transcode)"})
        # Warm the model server before QC so the first dataset pays load
        # once, not per file.
        pipe = get_campaign_pipeline(args.workdir, args.policy, args.models_lock,
                                     args.audio_profile)
        # Reset evidence/accepted for an idempotent re-run (same as cmd_qc).
        # paths/audit already exist from ingest above; keep the same AuditLog
        # so event seq numbers stay monotonic.
        for p in (paths.evidence, paths.accepted):
            if os.path.exists(p):
                os.remove(p)
        records = [SourceRecord.from_dict(r) for r in read_jsonl(paths.source_manifest)]
        every = push["every"] if push else 0
        step = every if every > 0 else len(records)
        combined = AnalyzeResult()
        n_blocked = 0
        for i in range(0, len(records), step or 1):
            sl = records[i:i + (step or len(records))]
            res_qc = cli_mod.qc_records(pipe, sl, paths, audit)
            combined.evidences.extend(res_qc.evidences)
            combined.accepted.extend(res_qc.accepted)
            if not res_qc.accepted:
                continue  # nothing new: tree unchanged, push would be NOOP
            # Deterministic per-group splits: re-finalizing over the growing
            # set never moves an earlier row, so slice exports converge with
            # the final export (no stale remote files).
            pipe._finalize_splits(combined, paths, audit)
            samples = [sample_from_accepted_row(r.to_dict(), paths.canonical_dir)
                       for r in res_qc.accepted]
            rep = normalize_samples(samples, args.dataset, args.state,
                                    require_redistributable=True)
            n_blocked += len(rep.get("blocked", []))
            stats["exported"] = stats.get("exported", 0) + len(samples) - len(
                rep.get("blocked", []))
            if push:
                _push_slice(args, push, len(sl), stats)
        stats["accepted"] = len(combined.accepted)
        stats["blocked"] = n_blocked
        pipe._finalize_splits(combined, paths, audit)
        funnel = cli_mod.write_qc_funnel(combined, paths)
        stats["qc_funnel_accept"] = funnel.get("accept_coverage")
        return res, stats
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(args.workdir, "runs", run_id), ignore_errors=True)
        if ds["format"] in ("loose", "arrow"):
            prune_hf_cache(ds["repo"])


def _push_slice(args, push, n_processed, stats):
    """Export-then-push one slice; ledger isolation means no file is re-pushed."""
    from ornix_dataset.publishing.incremental import push_tree

    rep = push_tree(args.dataset, push["repo_id"], args.state,
                    revision="main", policy_path=args.policy)
    stats["pushed"] = stats.get("pushed", 0) + rep.get("pushed", 0)
    if rep.get("status") == "NO_PUSH_NEEDED":
        return
    log(args.state, {"event": "push-slice", "processed": n_processed,
                     "pushed": rep.get("pushed"), "skipped": rep.get("skipped"),
                     "status": rep.get("status"), "reasons": rep.get("reasons"),
                     "commit": rep.get("commit_sha")})
    if not rep.get("ok"):
        stats["push_error"] = rep.get("reasons")


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


def extractor_cmd(ds, defaults, out_dir, loose_workers):
    py = sys.executable
    cmd = [py, os.path.join(HERE, "extract_hf_audio.py"),
           "--repo", ds["repo"], "--out", out_dir, "--format", ds["format"],
           "--loose-workers", str(loose_workers),
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


def run_dataset(ds, defaults, args, env, push=None):
    """Full-dataset single pass: extract ALL -> ingest -> qc -> canonical-export.

    Dispatches to the in-process runner (warm model server) or legacy
    subprocess. Returns (result, stats).
    """
    if _inprocess_enabled():
        return run_dataset_local(ds, defaults, args, push)
    return run_dataset_subprocess(ds, defaults, args, env, push)


def run_dataset_subprocess(ds, defaults, args, env, push=None):
    """Full-dataset single pass via subprocess stages. Returns (result, stats)."""
    name = ds["name"]
    extract_dir = os.path.join(args.workdir, "campaign_extract", name)
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)
    src_yaml = os.path.join(extract_dir, "_sources.yaml")
    run_id = name
    stats = {"written": 0, "ingested": 0, "accepted": 0, "exported": 0, "blocked": 0}
    try:
        rc, out, err = run(extractor_cmd(ds, defaults, extract_dir, args.loose_workers), env)
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
            stats["error"] = f"canonical rc={rc}: {out[-400:]}"
        if push and "error" not in stats:
            _push_slice(args, push, stats.get("ingested", 0), stats)
        return res, stats
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(args.workdir, "runs", run_id), ignore_errors=True)
        if ds["format"] in ("loose", "arrow"):
            prune_hf_cache(ds["repo"])

def process_dataset(ds, defaults, args, ckpt, env, push=None):
    """Run one dataset end-to-end in a single pass (dataset-level resume).

    A re-run skips datasets marked done; a failed dataset restarts from a
    clean extract dir (ingest dedups by content sha, QC evidence is rebuilt,
    and the push ledger skips bytes already verified on the Hub).
    """
    name = ds["name"]
    st = ckpt["datasets"].setdefault(name, {
        "status": "pending", "written": 0, "exported": 0, "blocked": 0,
        "pushed": 0})
    if st["status"] == "done":
        log(args.state, {"dataset": name, "event": "skip-done"})
        return
    st["status"] = "running"
    save_ckpt(args.state, ckpt)
    log(args.state, {"dataset": name, "event": "start"})
    fg = free_gib(args.state)
    if fg < args.min_free_gib:
        st["status"] = "paused-disk"
        log(args.state, {"dataset": name, "event": "pause-disk", "free_gib": round(fg, 1)})
        save_ckpt(args.state, ckpt)
        return
    res, stats = run_dataset(ds, defaults, args, env, push)
    if stats.get("error") and res is None:
        st["status"] = "error"
        st["error"] = stats["error"]
        log(args.state, {"dataset": name, "event": "error", "detail": stats["error"]})
        save_ckpt(args.state, ckpt)
        return
    st["written"] = stats.get("written", 0)
    st["exported"] = stats.get("exported", 0)
    st["blocked"] = stats.get("blocked", 0)
    st["pushed"] = stats.get("pushed", 0)
    st.pop("error", None)
    st["status"] = "done"
    log(args.state, {"dataset": name, "event": "done",
                     "free_gib": round(free_gib(args.state), 1), **stats})
    save_ckpt(args.state, ckpt)

def _canonical_rows(dataset_dir):
    rows = 0
    for split in ("train", "validation", "test"):
        mp = os.path.join(dataset_dir, split, "metadata.jsonl")
        if os.path.exists(mp):
            with open(mp, "r", encoding="utf-8") as fh:
                rows += sum(1 for _ in fh)
    return rows


def auto_publish(man, args, env):
    """Finalize the canonical tree, mint an operator approval, and push to HF.

    Gated by --publish (opt-in): the campaign operator has explicitly authorized
    publishing to their own destination repo, so we generate the approval receipt
    that binds this exact release digest and hand it to the same staged, verified
    publisher used interactively. Nothing here relaxes the rights gate — only
    rows that already passed canonical export exist in the tree.
    """
    dest = man.get("destination", {}) or {}
    repo_id = args.publish_repo or dest.get("repo_id")
    if not repo_id:
        log(args.state, {"event": "publish-skip", "reason": "no destination repo_id"})
        return
    rows = _canonical_rows(args.dataset)
    if rows == 0:
        log(args.state, {"event": "publish-skip", "reason": "canonical tree empty"})
        return
    if not os.environ.get("HF_TOKEN"):
        log(args.state, {"event": "publish-skip", "reason": "no HF_TOKEN"})
        return

    log(args.state, {"event": "publish-start", "repo_id": repo_id, "rows": rows})
    # Ensure the destination repo exists (operator-owned; exist_ok is a no-op if present).
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).create_repo(
            repo_id, repo_type="dataset", private=args.publish_private, exist_ok=True)
    except Exception as e:
        log(args.state, {"event": "publish-warn", "stage": "create_repo", "detail": str(e)[:200]})

    # Final push goes through the same ledger-isolated incremental publisher
    # (usually a NOOP after per-slice pushes) with a whole-tree exact-set
    # remote verification on top.
    from ornix_dataset.publishing.incremental import push_tree
    rep = push_tree(args.dataset, repo_id, args.state, revision="main",
                    policy_path=args.policy, full_verify=True)
    log(args.state, {"event": "publish-done", "ok": rep.get("ok"),
                     "status": rep.get("status"), "reasons": rep.get("reasons"),
                     "pushed": rep.get("pushed"),
                     "remote_commit_sha": rep.get("commit_sha")})
    return rep


def cmd_run(args):
    # Load .env once (existing env vars win) so HF_TOKEN and tuning knobs are
    # available to in-process stages — same contract as `ornix-dataset` CLI.
    try:
        from ornix_dataset.ops.env import load_dotenv
        load_dotenv()
    except Exception:
        pass
    man = load_manifest(args.manifest)
    defaults = man.get("defaults", {})
    if args.min_free_gib is None:
        args.min_free_gib = float(defaults.get("min_free_gib", 25))
    os.makedirs(args.state, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)
    datasets = man["datasets"]
    if args.only:
        wanted = set(args.only.split(","))
        datasets = [d for d in datasets if d["name"] in wanted]
    # Auto-tune to saturate the box: 0 means auto (GPU-aware). Explicit values
    # are still honored. Caps go to os.environ (in-process stages read them at
    # session creation) and are mirrored into env for legacy subprocess stages.
    par = resolve_parallelism(len(datasets), args.workers, args.ort_threads)
    args.workers = par["workers"]
    args.ort_threads = par["ort_threads"]
    if args.ingest_probe_workers and args.ingest_probe_workers > 0:
        probe_workers = max(1, args.ingest_probe_workers)
    else:
        probe_workers = par["probe_workers"]
    cap = str(max(1, args.ort_threads))
    os.environ["ORNIX_ORT_INTRA_THREADS"] = cap
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = cap
    os.environ["ORNIX_INGEST_PROBE_WORKERS"] = str(probe_workers)
    # Bulk mode: per-row fsync dominates QC wall time and every campaign write
    # is re-runnable (a crash only re-does tail rows, never corrupts). Explicit
    # ORNIX_FSYNC_APPEND=1 restores fully durable appends.
    os.environ.setdefault("ORNIX_FSYNC_APPEND", "0")
    env = dict(os.environ)
    try:
        import hf_transfer  # noqa: F401
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster bulk shard downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
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
            os.environ["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"]
    except Exception:
        pass
    # Small ONNX models scale poorly with intra-op threads, so dataset pipelines
    # run in parallel with capped per-worker pools. Canonical export serializes
    # across workers on its own flock; only checkpoint + progress log need the
    # in-process lock.
    # Incremental push config: every N QC-processed files the tree diff is
    # pushed (ledger-isolated: verified bytes are never re-pushed). Requires
    # --publish (operator pre-authorization) + HF_TOKEN, like the final push.
    push = None
    if getattr(args, "publish", False):
        dest = man.get("destination", {}) or {}
        repo_id = args.publish_repo or dest.get("repo_id")
        if not repo_id:
            log(args.state, {"event": "push-skip", "reason": "no destination repo_id"})
        elif not os.environ.get("HF_TOKEN"):
            log(args.state, {"event": "push-skip", "reason": "no HF_TOKEN"})
        else:
            push = {"repo_id": repo_id, "every": max(0, args.push_every)}
    ckpt = load_ckpt(args.state)
    log(args.state, {"event": "campaign-start", "datasets": [d["name"] for d in datasets],
                     "dataset_dir": args.dataset,
                     "workers": args.workers, "ort_threads": args.ort_threads,
                     "ingest_probe_workers": str(probe_workers),
                     "loose_workers": args.loose_workers,
                     "push_every": push["every"] if push else 0,
                     "push_repo": push["repo_id"] if push else None,
                     "cpu": par["cpu"], "gpu": par["gpu"],
                     "inprocess": _inprocess_enabled(),
                     "fsync": os.environ.get("ORNIX_FSYNC_APPEND", "1")})
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_dataset, ds, defaults, args, ckpt, env, push): ds["name"]
                     for ds in datasets}
            for f in futs:
                try:
                    f.result()
                except Exception as e:
                    log(args.state, {"event": "dataset-fail", "dataset": futs[f],
                                     "detail": str(e)[:300]})
    else:
        for ds in datasets:
            process_dataset(ds, defaults, args, ckpt, env, push)
            if getattr(args, "publish_each", False):
                try:
                    auto_publish(man, args, env)
                except Exception as e:
                    log(args.state, {"event": "publish-fail", "stage": "exception",
                                     "after": ds["name"], "detail": str(e)[:300]})
    log(args.state, {"event": "campaign-end"})
    if getattr(args, "publish", False):
        try:
            auto_publish(man, args, env)
        except Exception as e:
            log(args.state, {"event": "publish-fail", "stage": "exception", "detail": str(e)[:300]})
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
                                     ("status", "written", "exported", "blocked",
                                      "pushed", "error")}
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
    pr.add_argument("--min-free-gib", type=float, default=None)
    pr.add_argument("--only", default=None, help="comma-separated dataset names")
    pr.add_argument("--loose-workers", type=int, default=0,
                    help="concurrent downloads for loose-format datasets (0=auto)")
    pr.add_argument("--workers", type=int, default=0,
                    help="dataset pipelines in parallel; 0=auto (GPU: ~cpu_count, "
                         "CPU: ~cores/ort-threads, capped by dataset count)")
    pr.add_argument("--ort-threads", type=int, default=0,
                    help="intra-op thread cap per worker (also caps OMP/BLAS); "
                         "0=auto (1 on GPU, 2 on CPU)")
    pr.add_argument("--ingest-probe-workers", type=int, default=0,
                    help="per-dataset threads for ingest ffprobe+sha256 (0=auto ~2*cores/workers)")
    pr.add_argument("--publish", action="store_true",
                    help="push the canonical tree to HF: incremental diff-push every "
                         "--push-every processed files + final verified push")
    pr.add_argument("--push-every", type=int, default=100,
                    help="push the tree diff every N QC-processed files (0 = only "
                         "push at dataset end / campaign end)")
    pr.add_argument("--publish-each", action="store_true",
                    help="publish the (growing) canonical tree after EACH dataset completes")
    pr.add_argument("--publish-repo", default=None,
                    help="destination repo_id (default: manifest destination.repo_id)")
    pr.add_argument("--publish-private", action="store_true",
                    help="create the destination repo private if it does not exist")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("status", parents=[common])
    ps.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

