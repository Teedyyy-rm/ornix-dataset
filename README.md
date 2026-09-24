# Ornix Dataset — Quality & Curation Toolkit

Fail-closed tooling that ingests speech audio from multiple sources, measures
technical quality and noise, and produces a **redistribution-safe, ACCEPT-only**
24 kHz mono PCM16 release for Vietnamese-first TTS. Implements the design in
[`docs/E2E-ORNIX-DATASET.md`](docs/E2E-ORNIX-DATASET.md).

The pipeline never fabricates a PASS: a missing metric, an unavailable model, an
unverified transcript, or unclear redistribution rights routes to `REVIEW` /
`LICENSE_REVIEW`, never `ACCEPT`. Nothing is uploaded without an operator
approval receipt; publishing defaults to dry-run.

## Install

```bash
uv venv
. .venv/bin/activate
uv pip install -e ".[dev]"          # core + tests
# optional capability groups (see pyproject.toml): dsp, export, hf, vad, torch, full
uv pip install -e ".[full]"         # everything the adapters can use
```

`ffprobe`/`ffmpeg` on `PATH` enable the decode fallback path. Licensed detector
weights (Silero VAD, PANNs, DNSMOS, pyannote) are **optional**; when absent the
adapters report `UNAVAILABLE` and the affected gates stay `UNKNOWN` (fail-closed).

## Pipeline stages (phase → module)

| Phase | Concern | Module |
|---|---|---|
| 1 | Ingestion + immutable staging + provenance | `ingestion/` |
| 2 | Technical WAV gate + canonical 24k renderer | `dsp/` |
| 3 | VAD, windowing, event/noise detectors | `detectors/` |
| 4 | Quality/overlap/transcript + calibration | `detectors/`, `calibration/` |
| 5 | Deterministic policy, review, segmentation, dedup, split | `curation/` |
| 6 | Release build, exporters, dataset card, verify | `exporters/` |
| 7 | HF staged publisher (approval-gated) | `publishing/` |
| 8 | Cache, checkpoint, queue, retention | `ops/` |

## End-to-end run (local, offline)

```bash
# 0. validate declared source rights — no download
ornix-dataset sources validate --config configs/sources.example.yaml --dry-run

# 1. ingest + immutable staging (idempotent; re-run adds nothing)
ornix-dataset ingest --config configs/sources.example.yaml \
    --run-id run1 --workdir work

# 2. technical gate + detectors + deterministic policy decision
ornix-dataset qc --run-id run1 --policy configs/quality_policy.pilot.yaml \
    --models-lock configs/models.lock.example.yaml \
    --audio-profile configs/audio_profile.yaml --workdir work

# 3. export the review queue (everything not ACCEPT) for human adjudication
ornix-dataset review export --run-id run1 --workdir work

# 4. build an immutable, ACCEPT-only release (RELEASE_READY.json only if clean)
ornix-dataset release build --run-id run1 --release-id ornix-vi-0.1 \
    --format parquet --workdir work

# 5. offline verification (readback 24k/mono/PCM16, sha, no secrets, manifest)
ornix-dataset release verify --release-dir work/releases/ornix-vi-0.1

# 6. publish — DRY-RUN by default; prints the upload plan, touches no network
ornix-dataset publish --release-dir work/releases/ornix-vi-0.1 \
    --repo-id Teedyyy-rm/Ornix-Datasets
```

## Publishing (operator-gated, Phase 7)

A real upload requires **all** of the following, or it stops:

1. A built release with `RELEASE_READY.json` (all rows validate, all rights
   redistributable, no absolute paths/secrets, manifest checksums match).
2. An **approval receipt** (`configs/approval.example.yaml`) whose
   `release_digest` equals `sha256(MANIFEST.sha256)` of the exact release, bound
   to the destination `repo_id`, byte quota, policy version, operator id, and a
   future `expires_utc`.
3. A write-scoped token in `$HF_TOKEN` (never read from logs).

```bash
# compute the digest the operator must sign
ornix-dataset release verify --release-dir work/releases/ornix-vi-0.1   # prints release_digest
# ...operator fills approval.yaml with that digest...
export HF_TOKEN=hf_xxx
ornix-dataset publish --release-dir work/releases/ornix-vi-0.1 \
    --repo-id Teedyyy-rm/Ornix-Datasets --approval-file approval.yaml --execute
```

Publish uploads to a staging revision, re-verifies the remote tree/hashes, and
only then writes `PUBLISHED_VERIFIED.json`. Prior history is never deleted.
Statuses: `DRY_RUN`, `BLOCKED`, `UPLOAD_IN_PROGRESS`, `REMOTE_VERIFY_FAILED`,
`PUBLISHED_VERIFIED`.

## Policies

- `configs/quality_policy.example.yaml` — strict **public** target: music gate
  required, redistribution rights required. TRAIN_ONLY sources yield 0 ACCEPT.
- `configs/quality_policy.pilot.yaml` — **train_only** pilot using DSP-only
  gates so a local corpus can flow end-to-end for infra testing. Not valid for
  public redistribution. Thresholds must be operator-signed after calibration
  (Phase 4) before any public release.

## Operations (Phase 8)

- **Cache** keys on `(source_sha256, detector_weight_sha, config_sha,
  policy_version)` — a model/config/policy change invalidates stale results.
- **Checkpoints** are append-only and idempotent; a half-finished run resumes
  without duplicating or losing rows.
- **Retention / takedown**: `ops/retention.py` produces a non-destructive
  quarantine + new-release plan for a rights revocation; it never deletes source
  bytes.

## Testing

```bash
PYTHONPATH=src:tests python -m pytest -q          # full suite
PYTHONPATH=src:tests python -m pytest tests/unit  # unit only
```

Suite maps to the spec test matrix T-001..T-016 (see the progress table in
`docs/E2E-ORNIX-DATASET.md`). Tests are fully offline and CPU-only.

## Fail-closed invariants (do not bypass)

- Source bytes are immutable; every transform emits a new artifact referencing
  `source_sha256` + processing recipe.
- `ACCEPT` requires every required gate to PASS **and** redistribution rights.
- `UNKNOWN` / `ERROR` never promote to PASS; `ERROR` is not `REJECT`.
- No implicit denoise/normalize; no ASR-substituted transcripts or inferred
  speaker IDs.
- Never upload `raw/`, `rejected/`, `review/`, `quarantine/`, tokens, or
  unclear-rights data. Gated HF access does not grant redistribution rights.

