# MD-000 — Baseline & Architecture Audit (read-only, không sửa `src/`)

Ngày: 2026-09-24 · HEAD `07f1240` == `origin/main`, working tree sạch.
Graph: `home-obito-projects-ornix-dataset` (full, 1034 nodes / 5138 edges),
8 file nêu tên kiểm tra coverage `no_recorded_issue`.

## 1. Baseline môi trường & số liệu

- Python 3.12.3 · `huggingface_hub 1.33.0` · `hf_xet` ok/active
- RAM 15GB (avail ~7GB) · `/home` trống 623G (HDD) · `/` trống 133G
- Bench đã có (`work/bench/bench-20260924-*.json`): cold download file 9.2MB
  `w=1` 23.3 Mbps / `w=4` 19.9 Mbps · disk nội bộ 1108 MB/s · sha256 1672 MB/s
- **Thiếu, MD-007 phải đo:** throughput QC audio, throughput upload, bench
  nhiều-file-nhỏ, warm-cache transport (đã khai PERFORMANCE_UNVERIFIED).

## 2. Execution path hiện tại (CLI 8 lệnh, ingest ∥ QC tách rời)

`ingest · qc · review · curate · calibrate · release · publish · sources` —
`ingest` và `qc` là hai lệnh độc lập, **chưa tồn tại download/QC overlap**.
Điểm nối tương lai: `HfBatchDownloader._ready` (bounded `ready_queue_max`)
→ `OrnixPipeline.analyze_source()` theo từng file (Phase 4).

## 3. Audit từng module

| Module | File | Kết luận |
|---|---|---|
| Downloader | `ingestion/hf_downloader.py` | Tái dùng được: bounded scheduler, fail-closed, atomic staging, metrics tách network/hash/stage. Thiếu duy nhất **hợp đồng reservation với planner** (MD-002/003) |
| Pipeline | `pipeline.py` | `analyze_source()` per-source tái dùng tốt. `_finalize_splits` (`:523-542`) giữ **toàn bộ accepted trong RAM** + fail-closed leak → R1 xác nhận |
| Dedup | `curation/dedup.py` | `exact_duplicates` + `near_duplicate_groups` tồn tại nhưng **pipeline không gọi** → MD-004 phải nối + định global-evidence contract |
| Checkpoint | `ops/checkpoint.py` | JSONL append-only, filter `run_id`, last-wins — tái dùng được cho 1 job; đa-job cần **namespace path (campaign,job,batch) + fsync** (MD-001) |
| Publisher | `publishing/hf.py` | Dry-run default + approval + preflight tốt. 3 lỗi đã biết chờ MD-005: `staging_revision` hằng số (`:62`), marker ghi **trong** release_dir (`:111`), gộp lỗi publish/verify (`:80-82`) |
| Remote verify | `publishing/verification.py` | Exact-set + readback đúng; mặc định `sample=3` (`:32`) → bắt buộc `full_hash=true` cho release sẽ cleanup (Gate 5) |
| PermissionGate | `ingestion/permissions.py` | Fail-closed, khai báo显式, prefix-match — giữ nguyên; campaign phải mang declarations theo từng job |
| SourceRecord | `contracts/source.py` | Append-only, identity bất biến (`source_id/uri/revision/original_file_id/sha256`) — nền no-loss/no-dup tốt; idempotency manifest đa-batch phải chứng minh ở MD-001 |

## 4. Dependency map (rút gọn)

```
MD-001 campaign store ──▶ MD-002 planner (reservation contract)
   │ batch_id + ckpt paths        │ byte budget / watermarks
   ▼                              ▼
MD-003 downloader adapter ◀──▶ MD-004 orchestrator (ready-Q → analyze_source;
   (đã có, chỉ nối)         dedup wiring; PROCESSED vs RELEASE_READY)
                                   │
MD-005 publisher (namespace/   ◀──┘
 full_hash/status/marker)          │ receipt
                                   ▼
                        MD-006 cleanup (ownership/retention) ──▶ MD-007 E2E+bench
```

## 5. Root causes / thay đổi cần thiết theo module (cho các phase sau)

1. `pipeline.py`: nối dedup + finalization không giữ toàn bộ audio trong RAM (MD-004).
2. `ops/checkpoint.py`: namespace + fsync/fsync-dir (MD-001).
3. `publishing/hf.py`: namespace, tách status, marker ngoài release_dir (MD-005).
4. `publishing/verification.py`: `full_hash` bắt buộc cho cleanup-path (Gate 5).
5. Mới: campaign/job/batch store + `batch_id` ổn định (MD-001).
6. Mới: resource planner + reservation contract với downloader (MD-002).
7. Rủi ro ngoài code: quota repo HF đích (chỉ có preflight `repo_info`), 3 dataset MVP phải có quyền redistribute hợp lệ (R2).

## Gate 0 — AUDIT_VERIFIED ✅

Bằng chứng vấn đề đã có (mục 3–5), thành phần tái dùng đã rõ. Không sửa mã nguồn
trong phase này (2 file docs mới là output audit). Đề xuất mở MD-001.
