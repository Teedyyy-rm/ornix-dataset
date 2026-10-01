# PLAN — Bỏ resample, giữ tần số gốc, loại nguồn không nhất quán

Trạng thái: **DRAFT — chờ operator chốt sau Phase 0**
Ngày: 2026-10-02 · Phạm vi: `dsp/`, `admission`, `ingestion`, `canonical/`, `exporters/`, `contracts/`

---

## 1. Mục tiêu

Bỏ hai nhánh `DOWNSAMPLE` (44.1/48 kHz → 24 kHz) và `UPSAMPLE_NEAR_TARGET`
(22.05 kHz → 24 kHz) khỏi pipeline. Giữ **tần số gốc** của audio. Loại bỏ các
nguồn dữ liệu có **tần số xấu**.

### 1.1 Định nghĩa "tần số xấu" (theo yêu cầu operator)

> "ví dụ tần số 22k nhưng lộn lẫn vào các file chứa mp3 hoặc 8k thì nó là
> tần số xấu"

Đây là tiêu chí **tính nhất quán ở cấp nguồn**, không phải chuẩn hoá từng file:

| Hình dạng nguồn | Phân loại | Hành động |
|---|---|---|
| Mọi file cùng tần số, cùng codec | `HOMOGENEOUS_NATIVE` | **NHẬN** ở tần số gốc, không resample |
| Lẫn 22 kHz với MP3 với 8 kHz (dùng chung một dataset) | `HETEROGENEOUS` | **LOẠI** cả nguồn (fail-closed) |
| Đồng nhất nhưng tần số thấp (8 kHz / 16 kHz) | `LOW_RATE` | **LOẠI** |

Hệ quả thiết kế: tiêu chí này **không đòi hỏi mọi nguồn cùng tần số**. Một
nguồn 48 kHz đồng nhất vẫn hợp lệ. → Phát sinh câu hỏi release một hay nhiều
tần số, giải quyết ở Phase 1 sau khi có số liệu thật.

---

## 2. Hiện trạng: sáu lớp đang ép 24 000 Hz

| # | Vị trí | Nhiệm vụ | Xử lý |
|---|---|---|---|
| 1 | `dsp/render.py:23` `CANONICAL_SR = 24000` | hằng số đích | đổi thành policy |
| 2 | `dsp/render.py:124` `resample_to(mono, CANONICAL_SR)` | **chỗ DUY NHẤT thực sự resample** | bỏ hẳn |
| 3 | `dsp/render.py:84` `verify_canonical_wav` | verify file vừa ghi | theo policy |
| 4 | `contracts/release.py:62` `ReleaseRow.validate` | chặn row không canonical | theo policy |
| 5 | `exporters/verify.py:62` | verify offline release | theo policy |
| 6 | `canonical/verify.py:48` | verify cây canonical | theo policy |
| — | `canonical/schema.py:24` `CANONICAL_FIELDS` | **6 field, KHÔNG có `sample_rate`** | cần quyết định |

Điểm thuận lợi: canonical **không** re-encode — `canonical/normalize.py:99` chỉ
`shutil.copyfile` WAV đã render rồi gọi lại `verify_canonical_wav`. Nên chỉ cần
sửa một chỗ render là downstream nhận đúng bytes.

---

## 3. Phase 0 — Probe (chưa đụng pipeline)

**Mục tiêu:** biết phân bố `(sample_rate × codec × container)` thật của 9 dataset
campaign trước khi chốt chính sách. Không sửa một dòng pipeline nào.

### 3.1 Vấn đề kỹ thuật

`build_inventory()` chỉ gọi `list_repo_tree` → có `path` + `size`, **không có**
tần số hay codec. Muốn đo tần số **bắt buộc tải byte thật**. 24 GB toàn bộ thì
không khả thi, nhưng mẫu là đủ.

### 3.2 Việc cần làm

- Script mới `scripts/probe_source_rates.py` — standalone, chỉ đọc, không ghi
  vào campaign state.
- Tái dùng, không viết mới: `ffprobe` (đã chạy sẵn ở ingest probe qua
  `ORNIX_INGEST_PROBE_WORKERS`), `_sniff_ext()` trong `campaign/parquet.py:28`,
  `_expected_sha()` để kiểm tra LFS.
- **Chọn mẫu có lớp phủ, không phải 30 file đầu tiên:** chia đều theo thứ tự cây
  + ép thêm file lớn nhất và file nhỏ nhất của mỗi dataset. Mẫu đầu file-only thường
  toàn một định dạng và bỏ sót biến thể nằm ở đuôi.
- `N = 30` file/dataset, chạy song song `--workers`.
- Xuất `work/probe/<dataset>.json`: `{sr: count}` × `{codec: count}` ×
  `{container: count}`, kèm `homogeneous: bool` và danh sách file lệch chuẩn.

### 3.3 Gap cần biết trước

`_MAGIC_EXTS` (`campaign/parquet.py:20`) chỉ nhận RIFF/fLaC/OggS/ID3 + MPEG
frame + `ftyp`. **Không** nhận AIFF (`FORM`), WavPack, Opus trong Ogg, hay
M4A không có `ftyp` ở offset 4. Probe chỉ tin **ffprobe** (codec thật), không
tin magic bytes — đúng nguyên tắc `spec §6`: phân loại theo codec đo được, không
theo đuôi file. Magic bytes chỉ dùng để *gợi ý* tên file.

### 3.4 Cổng quyết định

Dừng lại. Trình bày bảng probe cho operator, **chưa viết code Phase 1+** cho tới
khi có câu trả lời cho câu hỏi ở §4.

---

## 4. Câu hỏi quyết định sau probe — release một hay nhiều tần số?

Đây là phần khó nhất. Nếu các nguồn đồng nhất hoá nhưng rải trên nhiều tần số
(ví dụ 24 kHz và 48 kHz), thì:

### Phương án A — Đơn tần số, loại nguồn lệch

Mọi nguồn được nhận phải có tần số **bằng nhau**. Nguồn 48 kHz bị loại khi
campaign chọn 24 kHz.

- Giữ nguyên contract 6 field, 6 lớp verify, mọi giả định downstream
- **Mất dữ liệu**: mọi nguồn 44.1/48 kHz bị bỏ dù sạch và đồng nhất
- Release vẫn là dataset đơn tần số → tự mô tả được

### Phương án B — Nhiều tần số gốc

Giữ nguyên tần số từng nguồn, release chứa 24 kHz + 48 kHz lẫn nhau.

- **Phá contract 6 field**: `CANONICAL_FIELDS` không có `sample_rate`, nên
  `metadata.jsonl` **không ghi được tần số từng file**. Phải mở từng WAV mới biết
  → dataset mất tính tự mô tả
- `datasets`' `AudioFolder` cast cột audio thành feature `Audio`, tự suy ra rate
  → hoạt động, nhưng consumer phải xử lý biến thiên
- Sửa cả 6 lớp verify + schema + card + docs
- `docs/CANONICAL-NORMALIZATION.md` phải viết lại

### Phương án C — Tách theo tần số

Release tách riêng theo tần số (`ornix-vi-24k/`, `ornix-vi-48k/`), mỗi cây là
đơn tần số và giữ nguyên contract 6 field.

- Giữ 6 field và verify đơn tần số cho từng cây
- Thêm một khái niệm mới (nhiều cây) vào loader/exporter
- Độ phức tạp trung bình, nhưng **giữ được cả dữ liệu lẫn tính tự mô tả**

**Khuyến nghị:** quyết định sau khi có số liệu probe. Nếu mọi nguồn đồng nhất ở
cùng một tần số thì A/B/C trùng nhau và toàn bộ phần này trở nên vô nghĩa — đó
chính là lý do phải probe trước.

---

## 5. Phase 2 — Cấu hình (sau khi chốt §4)

`configs/audio_profile.yaml`, thay khối `source_admission` hiện tại:

```yaml
sample_rate_policy:
  mode: native                # native | fixed
  min_native_sample_rate: 24000   # dưới ngưỡng này = LOW_RATE -> loại
  require_homogeneous: true   # nguồn lẫn nhiều rate/codec -> loại cả nguồn
  drop_on_heterogeneity: true
  homogeneity_sample_n: 30    # số file dùng để kết luận homogeneity
  # canonical_rate chỉ khai khi mode=fixed
```

Mọi thay đổi ngưỡng là **versioned config change** — không rải magic constant
trong code (`dsp/admission.py:57` ghi rõ quy tắc này).

---

## 6. Phase 3 — Thay đổi mã nguồn

### 6a. Bỏ resample khỏi render — `dsp/render.py`

- Xoá `CANONICAL_SR`, dùng `src_sr = buf.sample_rate` làm đích
- `resample_to(mono, CANONICAL_SR)` → giữ nguyên mảng
- Bỏ nhánh `linear-degraded` (`render.py:125`) — không còn resampler để hỏng
- `RenderRecipe`: bỏ `resample_method`, đổi `target_sample_rate` →
  `output_sample_rate`, bỏ `upsampled_from_below_target`
- `verify_canonical_wav` nhận tần số kỳ vọng từ tham số thay vì hằng số

### 6b. Admission — `dsp/admission.py`

- Bỏ nhánh `NATIVE_OR_HIGHER`→`DOWNSAMPLE` và `NEAR_TARGET_UPSAMPLE`
- `NATIVE_OR_HIGHER` → luôn `IDENTITY` (tần số gốc)
- `NEAR_TARGET_UPSAMPLE` → `LOW_RATE` (loại, vì 22.05 kHz không còn được nâng)
- Bỏ cờ `CONDITIONAL_UPSAMPLE_NOT_NATIVE_24K` — không còn nâng nên không cần nhãn
- Bỏ `canonical_sample_rate` khỏi `AdmissionConfig` (không còn đích 24k)

### 6c. Homogeneity gate (phần lớn nhất — công việc mới)

Hiện **chưa có** khái niệm "nguồn có nhất quán không". Cần:

- `InventoryItem` (`ingestion/hf_downloader.py:111`) mở rộng: thêm
  `sample_rate`, `codec`, `container` — điền từ probe hoặc từ header khi tải
- `CampaignStore`: lưu **profile tần số/codec của cả job** khi inventory chạy
- Gate mới ở admission, theo cấp **job**: một job có tần số lẫn nhau →
  `REJECT_SOURCE_HETEROGENEOUS`, toàn bộ batch của job bị loại
- Quyết định chưa chốt: heterogeneity đánh giá trên *toàn bộ* file hay trên
  *mẫu*? Toàn bộ thì tốn (tải hết 24 GB); mẫu thì có thể bỏ sót. Đề xuất:
  mẫu 30 file lúc inventory để **gate sớm** (fail-closed, loại nguồn bằng nghi),
  rồi xác nhận lại trên toàn bộ khi đã tải batch đầu.

### 6d. Sửa 6 lớp verify

`dsp/render.py:84` · `contracts/release.py:62` · `exporters/verify.py:62` ·
`canonical/verify.py:48` — thay so sánh cứng `!= 24000` bằng kiểm tra theo
policy (nếu §4 chọn A/C) hoặc theo profile của từng job (nếu chọn B).

---

## 7. Phase 4 — Test, tài liệu, dọn dẹp

### Test

- **79 chỗ tham chiếu `24000` trong 27 file test** phải rà lại — không xoá mù,
  phân loại: chỗ nào khẳng định invariant cũ (đổi) vs chỗ nào ngẫu nhiên vì
  fixture (giữ).
- Test mới bắt buộc:
  - nguồn đồng nhất 48 kHz → giữ 48 kHz, **không** resample
  - nguồn lẫn 22k/mp3/8k → `REJECT_SOURCE_HETEROGENEOUS`
  - nguồn đồng nhất 16 kHz → `LOW_RATE`
  - `verify_canonical_wav` nhận tần số bất kỳ và vẫn bắt được file sai
  - provenance ghi đúng `output_sample_rate`, không còn `resample_method`
- Giữ nguyên tinh thần hiện tại: offline, CPU-only, không network.

### Tài liệu

- `README.md` — đoạn "24 kHz mono PCM16 release" ở phần đầu và mục `audio_output`
- `configs/audio_profile.yaml` — khối `audio_output` + `source_admission`
- `docs/E2E-ORNIX-DATASET.md:125-149` — bảng rate class, invariant I1–I5/I9,
  giải thích "24 kHz output ≠ native 24 kHz source"
- `docs/CANONICAL-NORMALIZATION.md` — nếu chọn B/C thì viết lại contract
- `docs/MULTI-DATASET-AUTOMATION-PLAN.md` — nếu thêm khái niệm nguồn không nhất quán

---

## 8. Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Mất một phần lớn corpus 44.1/48 kHz dù chất lượng tốt | **cao** | Phase 0 probe đo trước; §4 quyết trên số liệu thật |
| Phá contract 6 field nếu chọn phương án B | **cao** | cân nhắc C, hoặc ghi `sample_rate` vào `release_manifest.jsonl` (đã có sẵn field) thay vì `metadata.jsonl` |
| Regression rộng — 79 chỗ test | trung bình | chia nhóm, chạy full suite ở mỗi bước |
| Homogeneity gate tốn I/O nếu đo toàn bộ | trung bị | gate sớm bằng mẫu; xác nhận khi đã tải batch đầu |
| Mất trường `resample_method` trong provenance | thấp | chấp nhận được: không còn resample thì không có ghi |
| Người dùng downstream đang giả định một tần số | trung bình | ghi rõ trong dataset card + CHANGELOG |

---

## 9. Đề xuất thứ tự thực thi

1. **Phase 0** — viết `scripts/probe_source_rates.py`, chạy trên 9 dataset
2. **Dừng lại** — trình bày bảng probe, chốt §4 (A / B / C)
3. **Phase 2** — cấu hình `sample_rate_policy`
4. **Phase 3** — 6a bỏ resample → 6b admission → 6c homogeneity gate → 6d verify
5. **Phase 4** — test + tài liệu
6. Rollback: toàn bộ thay đổi nằm sau config `sample_rate_policy.mode`;
   đặt lại `mode: fixed` + `canonical_rate: 24000` là quay về hành vi cũ.

## 10. Việc KHÔNG làm trong plan này

- Không nâng `min_native_sample_rate` hay đổi ngưỡng nào khác
- Không đụng detector, policy engine, split, dedup, publisher
- Không tự ý thêm/bớt field nào trước khi §4 được chốt
- Không commit — chờ operator review
