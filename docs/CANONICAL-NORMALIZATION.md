# Canonical Metadata, Filename & Speaker Normalization

> Status: **IMPLEMENTATION**. State machine per the task:
> `AUDIT_REQUIRED → SCHEMA_AND_IDENTITY_DESIGN_VERIFIED → IMPLEMENTATION_IN_PROGRESS
> → LOCAL_EXPORT_VERIFIED → LOADER_INTEGRATION_VERIFIED → REGRESSION_VERIFIED
> → IMPLEMENTATION_VERIFIED`. Remote upload is **not** performed here:
> `REMOTE_UPLOAD_UNVERIFIED` (no token/approval in this environment).

This document explains the `ornix_dataset.canonical` subsystem that unifies many
upstream datasets into one publishable Ornix Dataset. It is additive: the
existing QC, `PermissionGate`, `SourceRecord`/`QualityEvidence`/`ReleaseRow`
contracts and the staged publisher are reused, not replaced.

## 1. Canonical training metadata (exactly 6 fields)

| field | type | rule |
|---|---|---|
| `audio` | str | repo-relative path to the canonical WAV |
| `text` | str | **verified** transcript matched to the output audio |
| `file_name` | str | same relative path (Hugging Face `AudioFolder` link column) |
| `speaker` | str | `spk_<32 hex>` (never a source name/id) |
| `duration` | float | seconds, measured from the verified output WAV |
| `language` | str | unified code (e.g. `vi`); **never a blind default** |

No provenance field (`source_id`, source dataset, original filename, license,
quality score) is allowed in the public row. Those are kept in the internal
audit state (`canonical/identity.py`).

## 2. Filename & stable identity

- Every output WAV is `ornix_<32 lowercase hex>.wav`, stored at
  `audio/<first-2-hex>/ornix_<32hex>.wav`.
- The id is **not** derived from the source filename, dataset name, download
  order, batch index, timestamp or a per-dataset counter.
- Internal key: `("ornix-canonical-state-v1", revision, source_id,
  source_sha256, segment_start_sample, segment_end_sample)` — it distinguishes
  source revision, file/shard and segment identity.
- The mapping `internal key -> ornix_id` is persisted (atomic write, `flock`
  single-writer) **before** publish; an existing mapping is reused on
  retry/resume; a new id is `uuid4` with a collision check.
- If an output path already exists, content identity is checked: identical → reuse,
  different → hard `CONTENT_IDENTITY_CONFLICT` (audio is never overwritten).

## 3. Speaker normalization

- `speaker_scope_key(scope, ref)` -> `spk_<32hex>`.
- `scope` is the **source dataset + revision** (`hf://datasets/repo@rev` or the
  local root dir), so `speaker_01` in dataset A and dataset B never merge.
- `ref` is only trusted when the source declares one; an unverified speaker gets
  a **unique per-sample** id (no false merge), stable across resume.
- Speaker identity is never inferred from filename or voice similarity.
- Real names / source identifiers are never published.

## 4. Hugging Face output layout

```
Ornix-Datasets/
    README.md
    MANIFEST.sha256
    RELEASE_READY.json
    train/metadata.jsonl
    train/audio/<2hex>/ornix_<32hex>.wav
    validation/...
    test/...
```

Empty splits are not created; the `audio` path is relative to the split's
`metadata.jsonl`. Writing is incremental: `upsert_rows` atomically merges new
rows into the existing split metadata, so exporting a later batch never drops or
duplicates earlier rows.

## 5. Generation order (verified-only)

Download+verify → existing QC → ACCEPT only → verify canonical WAV → allocate/
reuse `ornix_id` → normalize speaker → write renamed WAV → 6-field metadata →
consistency verify → existing release packaging → publish+remote-verify.

A sample with a missing/unverified transcript, unknown language, non-canonical
audio, or non-permitted rights is **excluded with an explicit reason**, never
silently accepted. The dataset tree also stays unpublished until the rights-
and-identity mappings are committed.

## 6. Hugging Face research finding (AudioFolder)

Per the official docs (datasets `audio_dataset` / hub `datasets-audio`):

- the metadata link column is `file_name` (or `*_file_name`) and must be the
  **full relative path** from the metadata file to the audio file;
- `AudioFolder` **casts the linked audio column to the `Audio` feature**
  (`array`/`path`/`sampling_rate`), so it does **not** preserve `audio: str`.

Therefore the canonical loader reads `metadata.jsonl` directly
(`loader.load_ornix_dataset`) to keep the exact six-field contract;
`loader.load_with_datasets` is an explicit adapter and documents the cast. No
claim is made that `AudioFolder` returns the six fields unchanged.

## 7. Tests

`tests/unit/test_canonical.py` (T1–T15) and
`tests/integration/test_canonical_e2e.py` (T5/T6/T16/T17/T18) cover: exact six
fields and types; `ornix_<32hex>.wav` naming; no source name in output; two
datasets with the same filename; retry/resume stability; out-of-order workers
without duplicates; stable speaker mapping and no false merge; duration equals
the verified WAV; unverified transcript/language exclusion; `audio == file_name`
with every path present; no orphan/dangling/duplicate rows; preserved leakage-
safe splits; Unicode + float JSONL; internal provenance preserved; rights not
bypassed by renaming; real-WAV loader readback; incremental export; remote
verification failure blocking cleanup; and the full existing regression suite.

## 8. Known limitations / not claimed

- No live HF upload is performed or claimed here (`REMOTE_UPLOAD_UNVERIFIED`):
  the publish path is exercised against the in-memory `FakeHub` only.
- `duration`/readback are verified on real 24 kHz mono PCM16 WAVs; other formats
  must first pass the existing canonical renderer.
- Speaker identity is only as trustworthy as the source's declared speaker ref;
  unverified speakers are deliberately over-split, never over-merged.
- The campaign downloader does not itself carry source transcript/speaker/
  language; use the JSONL sidecar (`export-batch --metadata`) to supply them so
  the QC-accepted rows can be exported.
