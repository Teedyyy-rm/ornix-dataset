"""Ornix Dataset CLI (spec §7). Fail-closed; publish is dry-run by default."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .audit.events import AuditLog
from .audit.reports import build_funnel_report
from .config import (
    build_detectors,
    build_gate_and_adapters,
    load_policy_config,
    source_admission_config,
    technical_thresholds,
)
from .contracts.source import SourceRecord
from .pipeline import OrnixPipeline, RunPaths
from .util.io import write_json
from .util.jsonl import read_jsonl, write_jsonl

DEFAULT_WORKDIR = "work"


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_sources_validate(args) -> int:
    gate, adapters, specs = build_gate_and_adapters(args.config)
    summary = {"sources": [], "blockers": []}
    for spec, adapter in zip(specs, adapters):
        entry = {"name": spec["name"], "type": spec["type"],
                 "declared_rights": spec.get("rights", {}).get("rights_status", "UNKNOWN")}
        if spec["type"] == "local" and args.dry_run:
            try:
                recs = list(adapter.scan())
                entry["n_files"] = len(recs)
                entry["quarantined"] = sum(1 for r in recs
                                           if r.ingest_status.value == "QUARANTINE")
            except Exception as e:
                entry["error"] = str(e)
                summary["blockers"].append(f"{spec['name']}: {e}")
        rights = spec.get("rights", {}).get("rights_status", "UNKNOWN")
        if rights in ("UNKNOWN", "LICENSE_REVIEW", "FORBIDDEN"):
            summary["blockers"].append(
                f"{spec['name']}: rights {rights} -> not publishable without operator sign-off")
        summary["sources"].append(entry)
    _print(summary)
    return 0


def _build_pipeline(args) -> OrnixPipeline:
    policy = load_policy_config(args.policy)
    detectors = build_detectors(getattr(args, "models_lock", None))
    tech = technical_thresholds(getattr(args, "audio_profile", None))
    admission = source_admission_config(getattr(args, "audio_profile", None))
    return OrnixPipeline(args.workdir, policy, tech=tech, admission=admission,
                         detectors=detectors)


def cmd_ingest(args) -> int:
    gate, adapters, specs = build_gate_and_adapters(args.config)
    paths = RunPaths.create(args.workdir, args.run_id)
    audit = AuditLog(paths.audit, run_id=args.run_id)
    # ingest needs a policy only to construct the pipeline; use a permissive stub
    from .curation.policy import PolicyConfig

    stub = PolicyConfig(policy_version="ingest-only", required_checks=["rights_ok"])
    pipe = OrnixPipeline(args.workdir, stub)
    total = 0
    for adapter in adapters:
        recs = pipe.ingest_and_stage(adapter, paths, audit)
        total += len(recs)
    _print({"run_id": args.run_id, "ingested": total,
            "source_manifest": paths.source_manifest})
    return 0


def cmd_qc(args) -> int:
    paths = RunPaths.create(args.workdir, args.run_id)
    if not os.path.exists(paths.source_manifest):
        _print({"error": "no source_manifest; run ingest first"})
        return 2
    # reset evidence/accepted for idempotent re-run
    for p in (paths.evidence, paths.accepted):
        if os.path.exists(p):
            os.remove(p)
    pipe = _build_pipeline(args)
    audit = AuditLog(paths.audit, run_id=args.run_id)
    from .pipeline import AnalyzeResult

    combined = AnalyzeResult()
    for rec in read_jsonl(paths.source_manifest):
        src = SourceRecord.from_dict(rec)
        res = pipe.analyze_source(src, paths, audit)
        combined.evidences.extend(res.evidences)
        combined.accepted.extend(res.accepted)
    pipe._finalize_splits(combined, paths, audit)
    funnel = build_funnel_report(e.to_dict() for e in combined.evidences)
    write_json(os.path.join(paths.root, "QC_FUNNEL.json"), funnel)
    _print({"run_id": args.run_id, "candidates": len(combined.evidences),
            "accepted": len(combined.accepted), "funnel": funnel})
    return 0


def cmd_review_export(args) -> int:
    paths = RunPaths.create(args.workdir, args.run_id)
    from .curation.review import build_review_queue

    queue = build_review_queue(read_jsonl(paths.evidence), paths.evidence)
    write_jsonl(paths.review, [q.to_dict() for q in queue])
    _print({"run_id": args.run_id, "review_items": len(queue), "path": paths.review})
    return 0


def cmd_curate(args) -> int:
    paths = RunPaths.create(args.workdir, args.run_id)
    accepted = list(read_jsonl(paths.accepted))
    non_accept = [e for e in read_jsonl(paths.evidence)
                  if e.get("decision") != "ACCEPT"]
    report = {"run_id": args.run_id, "accepted": len(accepted),
              "non_accepted": len(non_accept),
              "accepted_only": bool(args.accepted_only)}
    write_json(os.path.join(paths.root, "CURATE_REPORT.json"), report)
    _print(report)
    return 0


def cmd_calibrate(args) -> int:
    from .calibration.runner import run_calibration

    pipe = _build_pipeline(args)
    result = run_calibration(args.goldset, pipe, args.workdir,
                             split=args.split, run_id=args.run_id)
    paths = RunPaths.create(args.workdir, args.run_id)
    payload = {"ok": result.ok, "split": result.split, "blockers": result.blockers,
               "report": result.report, "stratification": result.stratification}
    write_json(os.path.join(paths.root, "CALIBRATION_REPORT.json"),
               {**payload, "per_clip": result.per_clip})
    _print(payload)
    # metrics only — thresholds remain operator-signed; leakage/no-clips => nonzero exit
    return 0 if result.ok else 3


def cmd_release_build(args) -> int:
    paths = RunPaths.create(args.workdir, args.run_id)
    from .exporters import build_release
    rows = list(read_jsonl(paths.accepted))
    for r in rows:
        r["release_id"] = args.release_id
    out = args.out or os.path.join(args.workdir, "releases", args.release_id)
    rights_report = {
        "licenses": sorted({r.get("source_license", "UNKNOWN") for r in rows}),
        "rights_status": sorted({r.get("rights_status", "UNKNOWN") for r in rows}),
        "n_sources": len({r.get("source_id") for r in rows}),
        "release_target": args.release_target,
    }
    quality_report = {}
    fq = os.path.join(paths.root, "QC_FUNNEL.json")
    if os.path.exists(fq):
        quality_report = json.load(open(fq, encoding="utf-8"))
    art = build_release(args.release_id, rows, paths.canonical_dir, out,
                        rights_report=rights_report, quality_report=quality_report,
                        export_format=args.format, release_target=args.release_target)
    _print({"release_dir": art.release_dir, "ready": art.ready,
            "blockers": art.blockers, "n_rows": art.n_rows})
    return 0 if art.ready else 3


def cmd_release_verify(args) -> int:
    from .exporters import verify_release
    from .publishing.approval import release_digest

    res = verify_release(args.release_dir, offline=args.offline)
    out = {"ok": res.ok, "n_rows": res.n_rows, "total_duration_s": res.total_duration_s,
           "errors": res.errors}
    try:
        out["release_digest"] = release_digest(args.release_dir)
    except Exception as e:
        out["release_digest_error"] = str(e)
    _print(out)
    return 0 if res.ok else 3


def cmd_publish(args) -> int:
    from .publishing import StagedPublisher

    pub = StagedPublisher()
    res = pub.publish(args.release_dir, args.repo_id, dry_run=args.dry_run,
                      approval_path=args.approval_file)
    _print(res.to_dict())
    return 0 if res.status.value in ("DRY_RUN", "PUBLISHED_VERIFIED") else 3


def cmd_publish_verify(args) -> int:
    try:
        from huggingface_hub import HfApi
        from .publishing.verification import remote_verify
    except Exception as e:
        _print({"error": f"huggingface_hub unavailable: {e}"})
        return 2
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    res = remote_verify(api, args.repo_id, args.commit_sha, args.release_dir)
    _print(res.to_dict())
    return 0 if res.ok else 3


def cmd_campaign_create(args) -> int:
    from .campaign import CampaignStore, default_resolver
    from .config import load_yaml

    spec = load_yaml(args.input)
    if not isinstance(spec, dict):
        _print({"error": "campaign input must be a mapping"})
        return 2
    store = CampaignStore(args.root)
    try:
        campaign, outcomes = store.create_campaign(
            spec.get("name") or spec.get("campaign") or "campaign",
            spec.get("datasets") or [],
            destination=spec.get("destination"),
            publish_approved=bool(spec.get("publish_approved", False)),
            workspace=spec.get("workspace"),
            resolver=default_resolver)
    except ValueError as e:
        _print({"error": str(e)})
        return 2
    _print({"campaign_id": campaign.campaign_id, "root": store.root,
            "outcomes": outcomes})
    return 0 if all(o.get("ok") for o in outcomes) else 3


def cmd_campaign_status(args) -> int:
    from .campaign import CampaignStore

    store = CampaignStore(args.root)
    try:
        _print(store.status())
    except FileNotFoundError:
        _print({"error": f"no campaign in {store.root}"})
        return 2
    return 0


def cmd_campaign_resume(args) -> int:
    from .campaign import CampaignStore, default_resolver

    store = CampaignStore(args.root)
    try:
        # The resolver retries pins for BLOCKED jobs (failures stay per-job
        # errors); stored pins are never mutated — drift is only reported.
        report = store.resume(resolver=default_resolver,
                              check_remote=bool(args.check_remote))
    except FileNotFoundError:
        _print({"error": f"no campaign in {store.root}"})
        return 2
    _print(report)
    return 0


def cmd_campaign_plan(args) -> int:
    from .campaign import Budget, CampaignStore, StageProfile
    from .util.jsonl import read_jsonl

    store = CampaignStore(args.root)
    try:
        campaign = store.load_campaign()
    except FileNotFoundError:
        _print({"error": f"no campaign in {store.root}"})
        return 2
    files = []
    for rec in read_jsonl(args.inventory):
        size = rec.get("size", rec.get("bytes"))
        files.append((rec["path"], size if isinstance(size, int) else None))
    if not files:
        _print({"error": "empty inventory; refusing to plan zero batches"})
        return 2
    budget = Budget.from_workspace(
        campaign.workspace, store.root,
        max_bytes=args.max_batch_workspace, min_free_bytes=args.min_free,
        max_files_per_batch=args.max_files,
        max_batch_bytes=args.max_batch_bytes)
    profile = StageProfile(
        unknown_estimate_bytes=args.unknown_estimate,
        max_unknown_per_batch=args.max_unknown)
    try:
        batches = store.plan_job_batches(args.job, files, budget=budget,
                                         profile=profile)
    except (ValueError, FileNotFoundError) as e:
        _print({"error": str(e)})
        return 2
    blocked = [b.batch_id for b in batches if b.status == "BLOCKED"]
    _print({"job": args.job, "n_batches": len(batches),
            "n_blocked": len(blocked), "blocked": blocked,
            "plan": [{"batch_id": b.batch_id, "index": b.index,
                      "n_files": len(b.files),
                      "accountable_bytes": b.reservation.get("accountable_bytes", 0),
                      "status": b.status} for b in batches]})
    return 0


def cmd_campaign_inventory(args) -> int:
    import json

    from .campaign import CampaignStore, fetch_job_inventory
    from .util.io import atomic_write_text

    store = CampaignStore(args.root)
    try:
        job = store.load_job(args.job)
    except FileNotFoundError:
        _print({"error": f"unknown job {args.job}"})
        return 2
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        entries = fetch_job_inventory(api, job, allow=args.allow,
                                      ignore=args.ignore)
    except ValueError as e:
        _print({"error": str(e)})
        return 2
    except Exception as e:
        _print({"error": f"inventory fetch failed: {e}"})
        return 3
    out = args.out or store.inventory_file(job.job_id)
    atomic_write_text(out, "".join(
        json.dumps({"path": p, "size": s, "sha": h},
                   ensure_ascii=False, sort_keys=True) + "\n"
        for p, s, h in entries))
    _print({"job": job.job_id, "repo_id": job.repo_id,
            "pinned_sha": job.pinned_sha, "n_files": len(entries),
            "inventory": out})
    return 0


def cmd_campaign_download(args) -> int:
    from .campaign import Budget, CampaignStore, run_batch
    from .ingestion.hf_downloader import DownloadConfig

    store = CampaignStore(args.root)
    try:
        campaign = store.load_campaign()
        store.load_job(args.job)
        store.load_batch(args.batch)
    except FileNotFoundError as e:
        _print({"error": f"unknown campaign/job/batch: {e}"})
        return 2
    try:
        budget = Budget.from_workspace(campaign.workspace, store.root,
                                       min_free_bytes=args.min_free)
    except ValueError as e:
        _print({"error": str(e)})
        return 2
    cfg = DownloadConfig(file_workers=args.workers)
    rep = run_batch(store, args.job, args.batch, caps=budget.stage_caps,
                    min_free_bytes=budget.min_free_bytes, cfg=cfg)
    _print(rep)
    return 0 if rep.get("ok") else 3


def cmd_campaign_qc(args) -> int:
    from .campaign import CampaignStore, run_qc_batch

    store = CampaignStore(args.root)
    try:
        store.load_job(args.job)
        store.load_batch(args.batch)
    except FileNotFoundError as e:
        _print({"error": f"unknown campaign/job/batch: {e}"})
        return 2
    pipe = _build_pipeline(args)
    rep = run_qc_batch(store, args.job, args.batch,
                       lambda src, paths, audit: pipe.analyze_source(
                           src, paths, audit),
                       workdir=args.qc_workdir)
    _print(rep)
    return 0 if rep.get("ok") else 3


def cmd_campaign_gate(args) -> int:
    from .campaign import CampaignStore, gate_batch_release

    store = CampaignStore(args.root)
    try:
        rep = gate_batch_release(store, args.job, args.batch)
    except FileNotFoundError as e:
        _print({"error": f"unknown campaign/job/batch: {e}"})
        return 2
    _print(rep)
    return 0 if rep.get("ok") else 3


def cmd_campaign_release(args) -> int:
    from .campaign import CampaignStore, prepare_release

    store = CampaignStore(args.root)
    try:
        rep = prepare_release(store, args.job, args.batch,
                              release_target=args.target,
                              export_format=args.format)
    except FileNotFoundError as e:
        _print({"error": f"unknown campaign/job/batch: {e}"})
        return 2
    _print(rep)
    return 0 if rep.get("ok") else 3


def cmd_campaign_publish(args) -> int:
    from .campaign import CampaignStore, publish_release

    store = CampaignStore(args.root)
    try:
        rep = publish_release(store, args.release, args.repo,
                              args.approval, full_hash=args.full_hash,
                              upload_retries=args.upload_retries)
    except FileNotFoundError as e:
        _print({"error": f"unknown release: {e}"})
        return 2
    _print(rep)
    return 0 if rep.get("ok") else 3


def cmd_campaign_cleanup(args) -> int:
    from .campaign import CampaignStore, CleanupPolicy, ReservationLedger, cleanup_batch

    store = CampaignStore(args.root)
    ledger = ReservationLedger({s: 10**24 for s in
                                ("cache", "staging", "qc", "upload")})
    try:
        rep = cleanup_batch(store, args.job, args.batch, ledger=ledger,
                            policy=CleanupPolicy())
    except FileNotFoundError as e:
        _print({"error": f"unknown campaign/job/batch: {e}"})
        return 2
    _print(rep)
    return 0 if rep.get("ok") else 3


def cmd_campaign_pump(args) -> int:
    from .campaign import (
        Budget,
        CampaignStore,
        ReservationLedger,
        WatermarkGate,
        cleanup_batch,
        gate_batch_release,
        prepare_release,
        publish_release,
        run_batch,
        run_qc_batch,
    )
    from .ingestion.hf_downloader import DownloadConfig

    store = CampaignStore(args.root)
    try:
        campaign = store.load_campaign()
    except FileNotFoundError:
        _print({"error": f"no campaign in {store.root}"})
        return 2
    budget = Budget.from_workspace(campaign.workspace, store.root,
                                   min_free_bytes=args.min_free)
    ledger = ReservationLedger(budget.stage_caps)
    gate_wm = WatermarkGate(budget.high_watermark_bytes,
                            budget.low_watermark_bytes)
    cfg = DownloadConfig(file_workers=args.workers)
    pipe = None
    if args.policy:
        pipe = _build_pipeline(args)

    def _download(j, b):
        return run_batch(store, j, b, caps=budget.stage_caps,
                         min_free_bytes=budget.min_free_bytes,
                         ledger=ledger, cfg=cfg)

    def _qc(j, b):
        if pipe is None:
            return {"ok": False, "reason": "no-policy-for-qc"}
        return run_qc_batch(store, j, b,
                            lambda src, paths, audit: pipe.analyze_source(
                                src, paths, audit))

    def _publish(rid):
        if not args.approval_dir or not args.repo:
            return {"ok": False, "reason": "no-approval-dir"}
        ap = os.path.join(args.approval_dir, f"{rid}.yaml")
        return publish_release(store, rid, args.repo, ap, full_hash=True)

    rep = pump_campaign(
        store, ledger, gate_wm, _download, _qc,
        lambda j, b: gate_batch_release(store, j, b),
        prepare_one=lambda j, b: prepare_release(store, j, b),
        publish_one=_publish if args.approval_dir else None,
        cleanup_one=lambda j, b: cleanup_batch(store, j, b, ledger=ledger),
        max_steps=args.max_steps)
    _print(rep)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ornix-dataset",
                                description="Ornix dataset quality & curation (fail-closed)")
    p.add_argument("--version", action="version", version=f"ornix-dataset {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def wd(sp):
        sp.add_argument("--workdir", default=DEFAULT_WORKDIR)

    sp = sub.add_parser("sources", help="source registry commands")
    ssub = sp.add_subparsers(dest="sub", required=True)
    sv = ssub.add_parser("validate", help="validate source rights (no download)")
    sv.add_argument("--config", required=True)
    sv.add_argument("--dry-run", action="store_true")
    sv.set_defaults(func=cmd_sources_validate)

    ig = sub.add_parser("ingest", help="ingest + immutable staging")
    ig.add_argument("--config", required=True)
    ig.add_argument("--run-id", required=True)
    wd(ig)
    ig.set_defaults(func=cmd_ingest)

    qc = sub.add_parser("qc", help="technical + detectors + policy decision")
    qc.add_argument("--run-id", required=True)
    qc.add_argument("--policy", required=True)
    qc.add_argument("--models-lock", default=None)
    qc.add_argument("--audio-profile", default=None)
    wd(qc)
    qc.set_defaults(func=cmd_qc)

    rv = sub.add_parser("review", help="review queue commands")
    rsub = rv.add_subparsers(dest="sub", required=True)
    re = rsub.add_parser("export")
    re.add_argument("--run-id", required=True)
    wd(re)
    re.set_defaults(func=cmd_review_export)

    cu = sub.add_parser("curate", help="accepted-only curation report")
    cu.add_argument("--run-id", required=True)
    cu.add_argument("--accepted-only", action="store_true")
    wd(cu)
    cu.set_defaults(func=cmd_curate)

    cal = sub.add_parser("calibrate", help="measure FAR/FRR on the labeled gold set")
    cal.add_argument("--goldset", required=True, help="gold set JSONL (human-labeled)")
    cal.add_argument("--policy", required=True)
    cal.add_argument("--models-lock", default=None)
    cal.add_argument("--audio-profile", default=None)
    cal.add_argument("--split", default="calibration", choices=["calibration", "heldout"])
    cal.add_argument("--run-id", default="calibration")
    wd(cal)
    cal.set_defaults(func=cmd_calibrate)

    rel = sub.add_parser("release", help="release build/verify")
    relsub = rel.add_subparsers(dest="sub", required=True)
    rb = relsub.add_parser("build")
    rb.add_argument("--run-id", required=True)
    rb.add_argument("--release-id", required=True)
    rb.add_argument("--format", choices=["parquet", "webdataset"], default="parquet")
    rb.add_argument("--release-target", choices=["train_only", "public"],
                    default="train_only",
                    help="public requires every row REDISTRIBUTION_APPROVED (fail-closed)")
    rb.add_argument("--out", default=None)
    wd(rb)
    rb.set_defaults(func=cmd_release_build)
    rvf = relsub.add_parser("verify")
    rvf.add_argument("--release-dir", required=True)
    rvf.add_argument("--offline", action="store_true", default=True)
    rvf.set_defaults(func=cmd_release_verify)

    pb = sub.add_parser("publish", help="staged HF publish (dry-run default)")
    pbsub = pb.add_subparsers(dest="sub")
    pb.add_argument("--release-dir")
    pb.add_argument("--repo-id")
    pb.add_argument("--approval-file", default=None)
    grp = pb.add_mutually_exclusive_group()
    grp.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    grp.add_argument("--execute", dest="dry_run", action="store_false")
    pb.set_defaults(func=cmd_publish)
    pv = pbsub.add_parser("verify")
    pv.add_argument("--repo-id", required=True)
    pv.add_argument("--commit-sha", required=True)
    pv.add_argument("--release-dir", required=True)
    pv.set_defaults(func=cmd_publish_verify)

    cp = sub.add_parser("campaign", help="multi-dataset campaign commands")
    cpsub = cp.add_subparsers(dest="sub", required=True)
    cc = cpsub.add_parser("create", help="create campaign + pin dataset revisions")
    cc.add_argument("--input", required=True, help="campaign YAML/JSON file")
    cc.add_argument("--root", required=True, help="campaign state directory")
    cc.set_defaults(func=cmd_campaign_create)
    cs = cpsub.add_parser("status", help="show campaign jobs and batches")
    cs.add_argument("--root", required=True)
    cs.set_defaults(func=cmd_campaign_status)
    cr = cpsub.add_parser("resume", help="reconcile state; never mutates pins")
    cr.add_argument("--root", required=True)
    cr.add_argument("--check-remote", action="store_true",
                    help="report upstream drift without changing pins")
    cr.set_defaults(func=cmd_campaign_resume)
    cp_ = cpsub.add_parser("plan", help="resource-aware batch planning for a job")
    cp_.add_argument("--root", required=True)
    cp_.add_argument("--job", required=True)
    cp_.add_argument("--inventory", required=True,
                     help="JSONL with {path, size|null} per line")
    cp_.add_argument("--max-files", type=int, default=None)
    cp_.add_argument("--max-batch-bytes", type=int, default=None)
    cp_.add_argument("--max-batch-workspace", type=int, default=None,
                     help="cap on workspace bytes (default: campaign workspace)")
    cp_.add_argument("--min-free", type=int, default=None)
    cp_.add_argument("--unknown-estimate", type=int,
                     default=512 * 1024**2)
    cp_.add_argument("--max-unknown", type=int, default=50)
    cp_.set_defaults(func=cmd_campaign_plan)
    ci = cpsub.add_parser("inventory", help="list job files at the pinned commit")
    ci.add_argument("--root", required=True)
    ci.add_argument("--job", required=True)
    ci.add_argument("--allow", nargs="*", default=None)
    ci.add_argument("--ignore", nargs="*", default=None)
    ci.add_argument("--out", default=None)
    ci.set_defaults(func=cmd_campaign_inventory)
    cd = cpsub.add_parser("download", help="download one batch into verified staging")
    cd.add_argument("--root", required=True)
    cd.add_argument("--job", required=True)
    cd.add_argument("--batch", required=True)
    cd.add_argument("--workers", type=int, default=4)
    cd.add_argument("--min-free", type=int, default=None)
    cd.set_defaults(func=cmd_campaign_download)
    cq = cpsub.add_parser("qc", help="run audio QC over a downloaded batch")
    cq.add_argument("--root", required=True)
    cq.add_argument("--job", required=True)
    cq.add_argument("--batch", required=True)
    cq.add_argument("--policy", required=True)
    cq.add_argument("--models-lock", default=None)
    cq.add_argument("--audio-profile", default=None)
    cq.add_argument("--qc-workdir", default=None)
    wd(cq)
    cq.set_defaults(func=cmd_campaign_qc)
    cg = cpsub.add_parser("gate", help="job-global release gate for a batch")
    cg.add_argument("--root", required=True)
    cg.add_argument("--job", required=True)
    cg.add_argument("--batch", required=True)
    cg.set_defaults(func=cmd_campaign_gate)
    cr_ = cpsub.add_parser("release", help="build a namespaced release dir")
    cr_.add_argument("--root", required=True)
    cr_.add_argument("--job", required=True)
    cr_.add_argument("--batch", required=True)
    cr_.add_argument("--target", choices=["train_only", "public"],
                     default="train_only")
    cr_.add_argument("--format", choices=["parquet", "webdataset", "none"],
                     default="parquet")
    cr_.set_defaults(func=cmd_campaign_release)
    cpu = cpsub.add_parser("publish", help="upload a release + remote-verify")
    cpu.add_argument("--root", required=True)
    cpu.add_argument("--release", required=True)
    cpu.add_argument("--repo", required=True)
    cpu.add_argument("--approval", required=True)
    cpu.add_argument("--full-hash", action="store_true")
    cpu.add_argument("--upload-retries", type=int, default=3)
    cpu.set_defaults(func=cmd_campaign_publish)
    cc_ = cpsub.add_parser("cleanup", help="verified cleanup of one batch")
    cc_.add_argument("--root", required=True)
    cc_.add_argument("--job", required=True)
    cc_.add_argument("--batch", required=True)
    cc_.set_defaults(func=cmd_campaign_cleanup)
    cpump = cpsub.add_parser("pump", help="automatic continuation across batches/jobs")
    cpump.add_argument("--root", required=True)
    cpump.add_argument("--workers", type=int, default=4)
    cpump.add_argument("--min-free", type=int, default=None)
    cpump.add_argument("--policy", default=None)
    cpump.add_argument("--models-lock", default=None)
    cpump.add_argument("--audio-profile", default=None)
    cpump.add_argument("--approval-dir", default=None)
    cpump.add_argument("--repo", default=None)
    cpump.add_argument("--max-steps", type=int, default=100)
    wd(cpump)
    cpump.set_defaults(func=cmd_campaign_pump)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
