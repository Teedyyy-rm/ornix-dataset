# HF High-Speed Downloader — vận hành

Downloader tải nhiều file đồng thời qua transport chính thức
(`huggingface_hub` + `hf_xet`). Ornix chỉ điều khiển **tầng 1**
(file-level concurrency); **tầng 2** (luồng/range trong từng file) do Xet
điều khiển qua biến môi trường.

## 1. Quy tắc vàng: biến môi trường phải set TRƯỚC khi start process

`huggingface_hub` đọc mọi biến `HF_*` **lúc import**. Set sau import = vô tác
dụng. Downloader phát hiện và cảnh báo (`download_env_warning` trong audit)
khi giá trị hiệu lực khác cấu hình, nhưng cách đúng là export trước:

```bash
export HF_TOKEN=...                       # bắt buộc: tránh rate-limit anonymous
export HF_HOME=/mnt/ssd/hf-cache          # cache chính thức, KHÔNG sửa tay
# export HF_XET_HIGH_PERFORMANCE=1        # CHỈ khi RAM >= 64 GB (mặc định tắt)
# export HF_XET_FIXED_DOWNLOAD_CONCURRENCY=8  # CHỈ khi benchmark chứng minh có lợi
# export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1  # nếu cache nằm trên HDD
exec python -m ornix_dataset.cli ingest --config configs/sources.yaml --run-id r1
```

Không bao giờ dùng `HF_HUB_ENABLE_HF_TRANSFER` (deprecated, vô tác dụng mới).

## 2. Cấu hình (`configs/sources.yaml → sources[].download`)

| Trường | Mặc định | Nghĩa |
|---|---|---|
| `file_workers` | 1 (= cũ) | số file đồng thời; hard cap 32 |
| `verify_workers` | 4 | pool hash + stage |
| `max_inflight_bytes` | 2 GB | RAM/đĩa cho file đang tải + chờ stage |
| `ready_queue_max` | 64 | backpressure: QC chậm → ngừng prefetch |
| `min_free_disk_bytes` | 5 GB | reserve trên CẢ cache và staging (tính 2x: blob + staging) |
| `retry_max_attempts` / `retry_deadline_seconds` | 3 / 300 | retry tầng ngoài (SDK và Xet đã retry trong) |
| `allow_splits` | null | lọc split TRƯỚC khi tải |
| `xet_high_performance` | false | opt-in; bị từ chối khi RAM < 64 GB trừ `xet_hp_allow_low_ram: true` |

`revision` trong source spec được resolve **đúng 1 lần** thành commit SHA;
mọi lượt list/tải trong run dùng SHA đó. `allow_patterns` khớp 0 file →
fail-closed (trừ `allow_empty_inventory: true`).

## 3. Hồ sơ theo dataset (đo rồi mới chốt)

- Nhiều file nhỏ → tăng `file_workers` (4–16), giữ Xet adaptive mặc định.
- Một/vài shard lớn → `file_workers: 1–2`, để Xet tự tăng luồng tầng 2.
- Đĩa HDD → thêm `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`.
- Số đo 24/09/2026 (i5-12400F, 16 GB RAM, cache trên HDD, file 9 MB):
  workers=1 → 23.3 Mbps; workers=4 (1 file) → 19.9 Mbps (tầng 1 không giúp
  khi chỉ có 1 file — đúng thiết kế). Hash+stage 0.02 s so với 3.1 s mạng;
  trần đĩa ghi 1108 MB/s, trần hash 1672 MB/s → nút thắt nằm ở mạng/Xet,
  không phải CPU/đĩa local. Chi tiết: `work/bench/bench-20260924-*.json`.

## 4. Giám sát và sự cố

- Mỗi run ghi `work/runs/<id>/download_metrics.json`: `network_bytes`
  (chỉ byte mạng thật, không cộng cache), `cached_bytes`, thời gian từng
  giai đoạn, peak workers/queue, `retries`, `http_429`, `breaker_trips`.
- 429: SDK (≥1.2) tự chờ theo header `RateLimit` rồi retry; tầng Ornix chỉ
  giảm nửa worker + cooldown (circuit breaker), không retry mù.
- 401/403/404: vĩnh viễn, không retry, fail-closed (thường do thiếu quyền
  hoặc dataset gated — kiểm tra điều khoản truy cập, KHÔNG phải lỗi mạng).
- Journal `download_journal.jsonl` + staging nguyên tử (tmp + fsync +
  `os.replace` + read-only): kill giữa chừng rồi chạy lại không mất/không
  trùng bản ghi, không file dở (file `.part-*` được dọn khi khởi động).

## 5. Benchmark (`scripts/bench_hf_download.py`)

```bash
python scripts/bench_hf_download.py --repo-id <id> --revision <sha> \
  --allow '*.wav' --max-files 8 --max-bytes 50000000 \
  --workers 1 4 8 --live --out work/bench/report.json
```

Không `--live` = chỉ đo metadata (0 byte). Mỗi cấu hình chạy process con
riêng với cache tạm riêng, cùng thiết bị với production; không bao giờ
chạm cache thật của người dùng. So sánh công bằng: cùng dataset + commit
+ tập file; cold = cache rỗng, warm = chạy lại đúng thư mục đó.
