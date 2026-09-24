# Kế hoạch xây dựng Ornix Multi-Dataset Automation

> Nguồn: plan gốc do operator cung cấp (2026-09-24) + 5 hiệu đính từ review
> đối chiếu code `main` @ `07f1240`. Các hiệu đính được đánh dấu **[REVIEW-EDIT]**.
> Quy ước: mỗi phase có acceptance, evidence và commit riêng; một phase chỉ mở
> khi dependency bắt buộc của phase trước đã được chứng minh.

## 1. Mục tiêu tổng thể

Tự động xử lý hàng loạt dataset từ Hugging Face không cần thao tác thủ công
từng dataset. Người dùng nạp một danh sách dataset, khai báo repo HF đích,
ngân sách tài nguyên và cấp quyền; hệ thống tự điều phối đến khi hoàn tất
hoặc gặp blocker cần vận hành xử lý.

```
Multi-Dataset Campaign: Dataset A · Dataset B · Dataset C · Dataset D
Resource-Aware Scheduler → Download (Batch B) ∥ Process (Batch A) ∥ Upload (Batch trước)
→ Remote Verify → Safe Cleanup → Next Batch
```

Bốn yêu cầu đồng thời: tự động hóa · tiết kiệm lưu trữ (giải phóng trung gian
an toàn) · hiệu năng (download/QC/upload chồng lấp trong ngân sách) · bảo toàn
dữ liệu (không mất SourceRecord, không upload chưa xác minh, không xóa khi chưa
đủ điều kiện cleanup).

## 2. Quyết định kiến trúc (v1)

Một campaign, nhiều dataset, **một dataset xử lý chính tại một thời điểm**;
bên trong dataset đó download/QC/upload chạy đồng thời trên các batch khác nhau.

| Thành phần     | Quyết định                                              |
|----------------|---------------------------------------------------------|
| Điều phối      | Campaign → Dataset Job → Batch                          |
| Download       | Official `huggingface_hub` + `hf_xet`                   |
| Xử lý audio    | Tái sử dụng pipeline Ornix hiện có                      |
| Upload         | Mở rộng `StagedPublisher`                               |
| Lưu trữ đích   | Repository Hugging Face được cấu hình                   |
| Tiến độ        | Checkpoint bền vững, hỗ trợ resume                      |
| Cleanup        | Chỉ xóa file trung gian được phép xóa                   |
| Mặc định       | Dry-run và resource preflight trước execution           |

Batch processing và batch publishing là hai khái niệm riêng: chỉ upload
incremental khi mọi điều kiện xuất bản áp dụng cho batch đã hoàn tất; nếu chưa,
giữ provisional hoặc đổi kế hoạch phân batch — không làm yếu quality gate.
Scheduler Ornix không tự điều chỉnh concurrency nội bộ Xet khi chưa có benchmark.

## 3. Lộ trình 8 phase

### Phase 0 — Baseline & Architecture Audit (MD-000, read-only)
Audit downloader, `OrnixPipeline`, `Checkpoint`, `StagedPublisher`,
PermissionGate, release verification; phân biệt CLI ingest/QC độc lập với
download/QC overlap; kiểm tra source identity, dedup, manifest idempotency;
đo download/upload/QC/RAM/disk thực tế; xác định quality gate toàn-dataset.
**Gate 0 — AUDIT_VERIFIED.** Không sửa mã nguồn.
**[REVIEW-EDIT 5]** Output bắt buộc gồm file benchmark số liệu (kế thừa
`work/bench/*.json`) để MD-002/MD-007 có baseline so sánh.

### Phase 1 — Multi-Dataset Campaign & Job Queue (MD-001 ← MD-000)
Ba cấp Campaign → Dataset Job → Batch; input YAML/JSON; tạo/resume campaign;
phát hiện job trùng; pin exact commit SHA; chuẩn hóa URL về (repo, revision),
không dùng URL thô làm filesystem path. Tái dùng `Checkpoint`/SourceRecord/
manifest, chỉ bổ sung state store khi không đáp ứng đa-job.
**[REVIEW-EDIT 1]** Chốt ngay: `batch_id` ổn định = hash(repo_id, revision,
file-set) + convention checkpoint path theo `(campaign, job, batch)` — input
cho MD-006 ownership, tránh retrofit.
**Gate 1 — CAMPAIGN_VERIFIED.**

### Phase 2 — Resource-Aware Batch Planner (MD-002 ← MD-001)
Chia dataset thành batch vừa ngân sách: tổng bytes đã biết/chưa biết, cache,
staging, QC/canonical, upload package, headroom; kiểm soát cả disk thực tế lẫn
bytes đã reserve, ready-depth, unknown-size bytes; file đơn vượt ngân sách bị
chặn hoặc sang streaming đã kiểm chứng; high/low watermark dừng/tiếp nhận việc.
Hợp đồng: **planner giữ reservation, downloader chỉ enforce** (tránh
double-accounting với `_ByteBudget`/`_check_disk` sẵn có của `HfBatchDownloader`).
**Gate 2 — RESOURCE_PLAN_VERIFIED** (synthetic dataset > ổ đĩa vẫn chia batch
hợp lệ; không task nào chạy khi thiếu reservation).

### Phase 3 — High-Speed Downloader & Verified Staging (MD-003 ← MD-002)
`HfBatchDownloader` đã tồn tại (`71455f9`, gaps đóng ở `07f1240`, tái sử dụng):
HF cache + Xet chính thức, lọc trước tải, bounded scheduler, tách concurrency
file-level/Xet-internal, cancel/retry/checkpoint/resume, SHA256 + đối chiếu
remote checksum tương thích, atomic staging (tmp+fsync+replace+0o444), bàn giao
từng file verified qua bounded ready queue, giữ source identity, không duplicate
SourceRecord.
**[REVIEW-EDIT — tái định nghĩa]** MD-003 là "adapter + contract với planner
(MD-002)", không phải "xây downloader", để không chặn cứng giả tạo.
**Gate 3 — DOWNLOAD_VERIFIED.**

### Phase 4 — Processing Orchestrator & Batch Finalization (MD-004 ← MD-003)
Worker nối scheduler với `OrnixPipeline.analyze_source()` + release packaging +
quality gates hiện tại; tái dùng extractor cho shard Parquet/TAR, chưa hỗ trợ
thì ghi blocker (không đẩy shard thô vào QC). Hai cấp: `BATCH_PROCESSED`
(có outcome + evidence, chưa chắc release được) và `RELEASE_READY` (mọi kiểm tra
cấp batch/dataset/campaign xong, manifest/output/quyền hợp lệ).
**[REVIEW-EDIT 2]** Deliverable thêm "global-dedup evidence contract": dedup
(`curation/dedup.py`) hiện **chưa nối vào pipeline** — phải quyết dedup chạy ở
đâu trong đường batch, state xuyên batch giữ ở đâu. Gate 4 thêm test "batch bị
chặn release-ready khi thiếu global evidence" (dedup, leakage, policy toàn cục).

### Phase 5 — Incremental Publisher & Remote Verification (MD-005 ← MD-004)
Mở rộng `StagedPublisher` + PermissionGate + approval + `remote_verify`.
**[REVIEW-EDIT 3]** (a) Namespace per-(dataset, batch, release) là deliverable
**bắt buộc** (hiện `staging_revision` là hằng số đơn — xung đột mục tiêu nhiều
release chung repo). (b) Tách status `UPLOAD_FAILED` vs `REMOTE_VERIFY_FAILED`
(hiện gộp trong một `except`). (c) Ghi `PUBLISHED_VERIFIED.json` **ra ngoài**
release_dir (ghi vào trong làm bẩn release đã verify theo logic exact-set).
(d) "Re-run `upload_folder` = resume" là **experiment phải benchmark**, không
phải sự thật mặc định.
**[REVIEW-EDIT 4 — buộc chặt Phase 5→6]** Release nào cleanup sẽ xóa local thì
`remote_verify` phải chạy `full_hash=true`; `sample=3` chỉ cho pre-check.
**Gate 5 — PUBLISH_VERIFIED.**

### Phase 6 — Verified Cleanup & Automatic Continuation (MD-006 ← MD-005)
Cleanup theo loại + ownership (staging, QC tmp, canonical, upload pkg,
checkpoint, audit evidence, shared HF cache mỗi thứ một chính sách); kiểm tra
receipt, active workers, retention, review requirements, downstream deps; cập
nhật headroom + reservation; tự tiếp tục batch/dataset kế tiếp. Thiếu bằng
chứng thì giữ dữ liệu, không tự xóa để ép scheduler.
**Gate 6 — CLEANUP_VERIFIED** (upload lỗi / verify lỗi / crash giữa cleanup đều
không mất dữ liệu, không xóa nhầm job khác).

### Phase 7 — E2E Integration & Performance Tuning (MD-007 ← MD-001…006, P1)
Kịch bản bắt buộc: dataset > workspace, nhiều file nhỏ, vài shard lớn, QC chậm
hơn download, upload chậm hơn processing, disk gần đầy, HTTP 429, checksum
mismatch, crash từng giai đoạn. Benchmark riêng + overlap; so total time, peak
RAM/disk, retry, cache-hit, throughput (cache-hit bytes không tính thành
network). Chỉnh concurrency sau correctness, từ benchmark; không auto-bật Xet
high-perf mọi máy.
**Gate 7 — CAMPAIGN_E2E_VERIFIED.**

## 4. Thứ tự giao việc

MD-000 → MD-001 → MD-002 → MD-003 → MD-004 → MD-005 → MD-006 → MD-007,
mỗi task acceptance + evidence + commit riêng. Cấm cleanup thật trước khi điều
kiện verification + retention đã kiểm thử. Cho phép **MD-005 nghiên cứu sớm
(namespace format, upload-resume experiment) song song từ MD-002** vì là input
cho MD-001/MD-006.

## 5. MVP

Một dataset chính tại một thời điểm, nhiều batch ở các công đoạn khác nhau.
Scenario: nạp A+B+C → inventory + pin + chia batch → A tải/QC/đóng gói overlap
→ upload + remote verify → xóa trung gian đủ điều kiện (giữ review/checkpoint)
→ tự tiếp tục B, C → báo cáo campaign + blockers. Chưa cần: dashboard,
phân tán đa máy, đa-dataset QC nặng GPU, tự viết HTTP stack, auto-tune hàng
chục tham số.

## 6. Rủi ro (đã xác minh mức độ với code)

- **R1 — Gate toàn cục vs dataset > ổ đĩa (ĐÃ XÁC NHẬN, nặng hơn mô tả gốc):**
  `_finalize_splits` giữ toàn bộ accepted trong RAM; dedup chưa nối pipeline.
  Phải giữ đủ metadata/evidence cho finalization không cần toàn bộ audio, nếu
  không planner trả blocker thay vì xóa dữ liệu không tái tạo được.
- **R2 — HF đích thiếu quota/quyền:** preflight hiện chỉ `repo_info`, chưa check
  quota. Xác minh storage quota + repo policy + quyền redistribute trước khi
  campaign tự động chạy.
- **R3 — Cleanup mất khả năng recovery:** sau xóa local vẫn cần checkpoint,
  metadata, evidence, remote refs để kiểm chứng; dữ liệu chưa publish/review
  không chung chính sách với release đã xác minh.
