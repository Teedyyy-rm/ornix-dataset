# MD-007 — Integration & Performance Report

Date: 2026-09-24 · HEAD at time of run: see `git log` · Suite: all green
(see §4). Bench raw: `work/bench/bench-campaign-20260924.json`.

## 1. Mandatory scenarios (§Phase 7) — where each is proven

| Scenario | Proof | Result |
|---|---|---|
| Dataset > workspace | `test_planner.py`: 5 TB synthetic on 10 GB budget → 2500 valid batches | PASS |
| Many small files | E2E `test_full_campaign_two_datasets_end_to_end` (mock transport, real disk) | PASS |
| Few large shards | Oversize single file → BLOCKED batch (`test_planner`, `test_downloading`) | PASS |
| QC slower than download | Overlap demo: QC-bound stage races download (`test_processing`, bench §2) | PASS |
| Upload slower than processing | Retry + resume semantics (`test_publish_campaign` interrupt test) | PASS (mock) |
| Disk nearly full | `eligibility`/admit refuse; `stopped-disk` pump path (E2E disk test) | PASS |
| HTTP 429 | Downloader retry/backoff/breaker (`test_hf_downloader` T-series, pre-existing) | PASS (unit) |
| Checksum mismatch | E2E mismatch test: bad batch stays PLANNED, good batch completes, pump bounded | PASS |
| Process crash per stage | Crash-resume E2E (kill at step 3 → resume converges); re-plan preserves progress (unit) | PASS |

## 2. Performance (harness = mock transport + real disk/state)

E2E pump, 2 datasets × 2 files × 8 KB: **wall 0.531 s / 24 steps**
(download 4, qc 4, gate 4, prepare 4, publish 4, cleanup 4, job-done 2).
Overlap micro-demo (0.4 s stage): **serial 0.832 s → overlapped 0.416 s
(~50% saved)**. These numbers bound *orchestration overhead* (sub-second for
toy scale), not production throughput.

Real-medium numbers (prior live benches, unchanged): cold Hub download
**23.3 Mbps (w=1) / 19.9 Mbps (w=4)** on 9.2 MB file; local disk
**1108 MB/s**; sha256 **1672 MB/s** (`work/bench/bench-20260924-*.json`).

## 3. Crash recovery

- Kill mid-campaign (after 3/24+ steps) → resume from same disk converges to
  all-DONE with no duplicate releases and archived manifests intact.
- Re-planning after a crash preserves batch status/result (regression test
  `test_replan_preserves_completed_progress`) — this bug was found BY the
  E2E crash test and fixed in MD-007.
- Crash mid-cleanup converges on rerun (MD-006 tests).

## 4. Correctness ledger

Full suite at MD-007: run `PYTHONPATH=src:tests .venv/bin/python -m pytest`
(116 campaign unit + 4 campaign E2E + 1 live-gated skipped + all pre-existing).
No-load-bearing failures. Fail-closed paths (unpinned, unadmitted, tampered
release, sample-only receipt, leakage, incomplete global evidence) all covered.

## 5. Explicitly UNVERIFIED (do NOT claim VERIFIED)

- Live multi-file download throughput and many-small-files profile.
- Warm-cache transport behavior.
- Real-model QC audio throughput (all QC here uses injected analyzers).
- Real Hub upload throughput and `upload_folder` re-run resume against the
  production API (resume proven against FakeHub only).
- QC-overlap fusion at production scale.

## Gate 7 — CAMPAIGN_E2E_VERIFIED (with scope note) ✅

A campaign runs multiple datasets end-to-end on real state: upload, verify,
clean intermediates, continue automatically, recover after interruption.
**Performance is VERIFIED for orchestration overhead only**; production
throughput stays PERFORMANCE_UNVERIFIED pending live benchmarks.

## Post-audit additions (G1–G4)

- **G1/G2 (commit `b39e156`):** overlapped pump (`--overlap`: download-next
  while QC-current, admitted up-front) and true per-file handoff (downloader
  restructured to a 3-stage pipeline so `on_ready` fires mid-run). Tests:
  overlap faster-than-sequential, mid-run handoff, ledger thread-safety.
- **G3 (this commit):** destination preflight (`campaign preflight`). Research
  finding: the Hub exposes **no public quota API** — quota is operator-declared
  (`destination.max_bytes`), measured against `list_repo_tree`, fail-closed;
  undeclared => warning; plus the 500 GB single-file hard limit and
  `redistribution_confirmed`. Enforced at campaign level and per-release.
- **G4 (this commit):** `campaign report` (one aggregate artifact: statuses,
  blockers, releases, cleanup footprint, `--out`) and `campaign report
  --dry-run` (read-only next-action preview, mutates nothing).
