# PLAN — Bỏ resample, giữ tần số gốc, loại nguồn không nhất quán

Trạng thái: **IMPLEMENTED một phần (2026-10-02) — operator đã chốt §4 Phương án B
(giữ native rate, đa tần số); Phase 2 + Phase 3 (6a/6b/6b.2/6d) đã vào code, full
suite xanh; 6c (homogeneity gate) hoãn có chủ ý; Phase 0 còn 4 nguồn chưa probe**
Ngày: 2026-10-02 · Phạm vi: `dsp/`, `admission`, `ingestion`, `canonical/`,
`exporters/`, `contracts/`, `configs/`
Ghi chú: bản 2026-10-02 (lần 2) đã đối chiếu lại toàn bộ plan với code và bổ
sung các mục bản lần 1 bỏ sót — xem §2.2, §6b.2, §6e.
Bản lần 3 (2026-10-02): gắn kết quả Phase 0 thật (§3.5), verify lại toàn bộ số
dòng tham chiếu với code hiện tại (sửa `InventoryItem` 111→110, E2E 129→128),
và ghi nhận môi trường chạy: test chỉ chạy local (offline, CPU-only); run nặng
có thể đẩy lên Colab qua MCP `colab-mcp` (`.mcp.json`).
Bản lần 4 (2026-10-02): operator chốt **Phương án B** + phạm vi "chỉ bỏ resample";
đã implement (xem §11). `resample_method`, `DOWNSAMPLE`, `UPSAMPLE_NEAR_TARGET`
đã bị xoá khỏi schema/provenance; canonical output giữ nguyên native rate.

---

## 1. Mục tiêu

Bỏ **toàn bộ** cơ chế chuyển tần số về 24 kHz, không chỉ hai nhánh đang chạy.
Cụ thể là `DOWNSAMPLE` (44.1/48 kHz → 24 kHz) và `UPSAMPLE_NEAR_TARGET`
(22.05 kHz → 24 kHz), cùng mọi chỗ gắn cứng 24 000 Hz trên đường release. Giữ
**tần số gốc** của audio. Loại bỏ các nguồn dữ liệu có **tần số xấu**.

### 1.0 Vì sao bỏ hẳn, không giữ chính sách "resample có kiểm soát"

Operator đánh giá việc ép về 24 kHz là **thừa** và **làm giảm chất lượng**.
Ba lý do kỹ thuật đứng sau đó, dùng để biện minh khi code review:

1. **Resample không thêm thông tin, chỉ đổi hình thức.** `48k → 24k` là
   decimation có band-limit: nó **xoá** nội dung trên 12 kHz. File đầu ra nhỏ
   đi, không phải "sạch hơn". `24k → 48k` là nội suy, tạo số 0 ở dải cao —
   thuần nhiễu thêm, file to thêm.
2. **Nó là nguồn sai lệch âm thầm.** Khi mọi nguồn — kể cả 8 kHz telephony —
   đều ra cùng header `24000 Hz`, người đọc `metadata.jsonl` **không còn cách
   nào biết** dữ liệu gốc bao nhiêu. Chính plan này đang cố dựng lại
   `sample_rate` mà contract 6 field không cho phép ghi (§4). Bỏ resample là
   cách **duy nhất** để `sample_rate` trở lại thành thông tin trung thực.
3. **Nó tốn I/O và phụ thuộc SciPy.** `resample.py` phải có `scipy.signal`
   mới cho clip band-limited; thiếu SciPy thì nhánh `linear-degraded`
   (`render.py:125`) fail-closed và chặn cả clip sạch. Bỏ resample canonical
   thì render không còn phụ thuộc ngoài.

Lưu ý phạm vi: chỉ bỏ resample **canonical**. Resample **phân tích** của
detector (VAD/DNSMOS 16 kHz, PANNs 32 kHz) là bắt buộc theo rate model, giữ
nguyên — xem §2 và §10.

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
tần số, giải quyết ở §4 sau khi có số liệu thật.

---

## 2. Hiện trạng: toàn bộ bề mặt ép 24 000 Hz

Cần phân biệt **hai loại resample** trước khi sửa, nếu không sẽ xoá nhầm:

| Loại | Mục đích | Xử lý trong plan |
|---|---|---|
| **Resample canonical** — `dsp/render.py:124` | ép file WAV xuất ra về 24 kHz | **bỏ hẳn** (mục tiêu của plan) |
| **Resample phân tích** — detectors | đưa tín hiệu về rate model (PANNs 32 kHz, DNSMOS/VAD 16 kHz) | **giữ nguyên** — không liên quan rate release |

### 2.1 Sáu lớp ép rate (lần viết đầu của plan, đã đối chiếu lại)

| # | Vị trí | Nhiệm vụ | Xử lý |
|---|---|---|---|
| 1 | `dsp/render.py:23` `CANONICAL_SR = 24000` | hằng số đích | đổi thành policy |
| 2 | `dsp/render.py:124` `resample_to(mono, CANONICAL_SR)` | **chỗ DUY NHẤT resample canonical** | bỏ hẳn |
| 3 | `dsp/render.py:84` `verify_canonical_wav` | verify file vừa ghi | theo policy |
| 4 | `contracts/release.py:62` `ReleaseRow.validate` | chặn row không canonical | theo policy |
| 5 | `exporters/verify.py:62` | verify offline release | theo policy |
| 6 | `canonical/verify.py:48` | verify cây canonical | theo policy |
| — | `canonical/schema.py:24` `CANONICAL_FIELDS` | **6 field, KHÔNG có `sample_rate`** | cần quyết định (§4) |

### 2.2 Chỗ ép rate bản plan đầu BỎ SÓT

Đã grep toàn bộ `src/`, `configs/`, `scripts/`. Những chỗ dưới đây cũng đang
gắn cứng 24 000 Hz và **không xuất hiện ở §2.1**:

| # | Vị trí | Vì sao nguy hiểm | Xử lý |
|---|---|---|---|
| 7 | `pipeline.py:553` `ReleaseRow(sample_rate=24000, ...)` | **ghi rate giả vào manifest**: WAV 48 kHz vẫn bị ghi `sample_rate: 24000` → `ReleaseRow.validate` pass, provenance sai | đọc `recipe.output_sample_rate` |
| 8 | `dsp/decode.py:102` `sr = info.get("sample_rate") or 24000` | **fail-open**: ffprobe không trả rate thì decode tự giả 24 kHz — trái nguyên tắc fail-closed mà `admission.py:117` áp dụng | raise `DecodeError`, không giả định |
| 9 | `dsp/render.py:105-110` `_action_for(src_sr)` | tự suy ra `DOWNSAMPLE`/`UPSAMPLE_NEAR_TARGET` khi không có `admission` (fallback tại `render.py:151-152`) | xoá hẳn |
| 10 | `configs/audio_profile.yaml:3` `audio_output.sample_rate: 24000` | khối này **không có code nào đọc** (grep `audio_output` chỉ ra file này + plan) — tức là mô tả, không phải policy | xoá hoặc ghi rõ là doc-only |
| 11 | `dsp/technical.py:190` `ref_nyq = min(sr/2, canonical_nyquist_hz=12000)` | tham chiếu băng thông cứng 12 kHz; khi rate gốc lên 48 kHz thì ref vẫn trần 12 kHz | xem §6e |
| 12 | `pipeline.py:248,263,272` `analysis_sample_rate=16000` | hardcode trong 3 evidence reject; không phải rate release nhưng cùng họ "rate" | đặt theo detector thật |
| 13 | `exporters/card.py:43` `"Audio format: 24 kHz, mono, PCM_S16LE"` + `:60` | dataset card hứa một rate duy nhất | theo §4 |
| 14 | `testing/fakes.py:177`, `scripts/make_goldset.py:25`, `scripts/bench_campaign.py:90` | fixture/goldset giả định 24 kHz | rà ở Phase 4 |

### 2.3 Cần sửa một tuyên bố sai trong bản plan đầu

Bản plan đầu ghi `render.py:124` là "**chỗ DUY NHẤT thực sự resample**". Sai:
detectors cũng resample, qua `_resample_to` (`detectors/music_noise.py:159`):

- `detectors/vad.py:83` → 16 000
- `detectors/quality.py:26` → 16 000
- `detectors/music_noise.py:88` → 32 000 (PANNs)

Đây là resample **phân tích**, bắt buộc phải giữ (model chỉ nhận một rate).

Hệ quả cần nói rõ: **`dsp/resample.py` sẽ thành code chết trong `src/`.** Sau khi
bỏ `render.py:124`, không còn call site nào tới nó — `dsp/resample.py` chỉ còn tự
gọi chính nó (`resample_to` → `resample_poly_quality`), còn detectors dùng bản
sao riêng `_resample_to` (`music_noise.py:159`) gọi thẳng `scipy.signal`.
`resample_poly_quality` cũng không có consumer khác trong `src/` ngoài
`resample_to`.

Vì vậy `dsp/resample.py` **không xoá trong plan này** — còn test tham chiếu
(`tests/unit/test_render_verify.py:11`) và nó thuộc `dsp/__init__.py` public API.
Ghi nhận là dead-code-ứng-viện để dọn ở một change riêng; xem §10.

### 2.4 Điểm thuận lợi

Canonical **không** re-encode — `canonical/normalize.py:99` chỉ
`shutil.copyfile` WAV đã render rồi gọi lại `verify_canonical_wav` (`normalize.py:130`).
Nên chỉ cần sửa một chỗ render là downstream nhận đúng bytes.

---

## 3. Phase 0 — Probe (chưa đụng pipeline)

**Mục tiêu:** biết phân bố `(sample_rate × codec × container)` thật của 9 dataset
campaign trước khi chốt chính sách. Không sửa một dòng pipeline nào.

### 3.1 Vấn đề kỹ thuật

`build_inventory()` chỉ gọi `list_repo_tree` → có `path` + `size`, **không có**
tần số hay codec. Muốn đo tần số **bắt buộc tải byte thật**. 24 GB toàn bộ thì
không khả thi, nhưng mẫu là đủ.

### 3.2 Việc cần làm

- ✅ Script `scripts/probe_source_rates.py` đã viết (444 dòng) — standalone, chỉ
  đọc, không ghi vào campaign state; mặc định đọc
  `configs/ornix_campaign.small9.yaml`. Chưa commit (untracked).
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

Dừng lại. Trình bày bảng probe cho operator, **chưa viết code Phase 2+** (§5 trở
đi) cho tới khi có câu trả lời cho câu hỏi ở §4.

### 3.5 Kết quả probe tới nay (5/9 nguồn, 2026-10-02)

`scripts/probe_source_rates.py` đã chạy trên `configs/ornix_campaign.small9.yaml`;
kết quả trong `work/probe/` (tổng hợp: `work/probe/_summary.json`):

| dataset | layout | n probe | rate | codec | container | đồng nhất |
|---|---|---|---|---|---|---|
| thanhpahm_tts | parquet | 30 | 24 000 | pcm_s16le | wav | ✔ |
| ngochuyen_vivoice | parquet | 30 | 24 000 | pcm_s16le | wav | ✔ |
| ngochuyen_voice | parquet | 30 | 24 000 | pcm_s16le | wav | ✔ |
| hatrang_voice | loose | 12 | 24 000 | pcm_s16le | wav | ✔ |
| fpt_fosd | parquet | 12 | 48 000 | mp3 | mp3 | ✔ |
| vietbiblevox | chưa probe | — | — | — | — | — |
| luna_vi | chưa probe | — | — | — | — | — |
| vi_voice_female | chưa probe | — | — | — | — | — |
| w2wmovie_voice_2 | chưa probe | — | — | — | — | — |

Đọc kết quả:

- **Không nguồn nào rơi vào `HETEROGENEOUS` hay `LOW_RATE`** trong 5 nguồn đã
  đo — tiêu chí §1.1 chưa loại nguồn nào, và 4/5 nguồn đã đúng 24 kHz.
- **`fpt_fosd` là nguồn lệch tần số duy nhất tới nay**: 48 kHz, lại là MP3
  (lossy). Theo §1.1 nó vẫn là `HOMOGENEOUS_NATIVE` (một rate, một codec) —
  nhưng nếu chọn Phương án A (đơn tần số 24 kHz) thì mất nguồn này; B/C giữ được.
- `hatrang_voice` và `fpt_fosd` hiện mới probe `n=12` (các nguồn khác `n=30`);
  chạy nốt để đủ mẫu 30 theo §3.2 trước khi kết luận.

Chưa kết luận được §4 cho tới khi 4 nguồn còn lại được probe: `vietbiblevox`
(parquet x100, 9.46 GB), `luna_vi` (loose x30420 file, 7.73 GB),
`vi_voice_female` (parquet x5), `w2wmovie_voice_2` (parquet x3).

---

## 4. Câu hỏi quyết định sau probe — release một hay nhiều tần số?

Đây là phần khó nhất. Nếu các nguồn đồng nhất hoá nhưng rải trên nhiều tần số
(ví dụ 24 kHz và 48 kHz), thì:

> **ĐÃ CHỐT (2026-10-02): Phương án B — giữ tần số gốc, release đa tần số.**
> Rate từng file nằm ở `ReleaseRow.sample_rate` trong `release_manifest.jsonl`;
> `metadata.jsonl` giữ đúng 6 field, không thêm `sample_rate`. §6c (homogeneity
> gate) hoãn có chủ ý (phạm vi lần này: chỉ bỏ resample).

### Phương án A — Đơn tần số, loại nguồn lệch

Mọi nguồn được nhận phải có tần số **bằng nhau**. Nguồn 48 kHz bị loại khi
campaign chọn 24 kHz.

- Giữ nguyên contract 6 field, các lớp verify (§2.1), mọi giả định downstream
- **Mất dữ liệu**: mọi nguồn 44.1/48 kHz bị bỏ dù sạch và đồng nhất
- Release vẫn là dataset đơn tần số → tự mô tả được

### Phương án B — Nhiều tần số gốc

Giữ nguyên tần số từng nguồn, release chứa 24 kHz + 48 kHz lẫn nhau.

- **Phá contract 6 field**: `CANONICAL_FIELDS` không có `sample_rate`, nên
  `metadata.jsonl` **không ghi được tần số từng file**. Phải mở từng WAV mới biết
  → dataset mất tính tự mô tả
- `datasets`' `AudioFolder` cast cột audio thành feature `Audio`, tự suy ra rate
  → hoạt động, nhưng consumer phải xử lý biến thiên
- Sửa các lớp verify ở §2.1 **và** hai chỗ ghi rate ở §6d (`pipeline.py:553`,
  `dsp/decode.py:102`) + schema + card + docs
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

**Dữ liệu tới nay (§3.5):** 4/5 nguồn đã đo đều ở 24 kHz; nguồn lệch duy nhất là
`fpt_fosd` (48 kHz MP3). Trên tập đã đo, Phương án A chỉ mất đúng `fpt_fosd`;
B/C giữ nguyên. Nếu 4 nguồn còn lại cũng 24 kHz thì cục diện là "8 nguồn 24 kHz +
1 nguồn 48 kHz", và câu hỏi thực chất rút gọn thành: **có đáng loại `fpt_fosd`
(~0.82 GB, ~25.9k clip) để giữ release đơn tần số hay không** — thay vì một quyết
định kiến trúc lớn như đặt ra ban đầu.

---

## 5. Phase 2 — Cấu hình (sau khi chốt §4)

`configs/audio_profile.yaml`, thêm khối `sample_rate_policy` cạnh
`source_admission`:

```yaml
sample_rate_policy:
  mode: native                # native | fixed
  min_native_sample_rate: 24000   # dưới ngưỡng này -> loại (nhãn xem §6b)
  require_homogeneous: true   # nguồn lẫn nhiều rate/codec -> loại cả nguồn
  drop_on_heterogeneity: true
  homogeneity_sample_n: 30    # số file dùng để kết luận homogeneity
  # canonical_rate chỉ khai khi mode=fixed
```

Ba việc kèm theo mà bản plan đầu bỏ qua:

1. **Phải viết loader.** `config.py:77 source_admission_config()` chỉ đọc
   `source_admission.clean_hq`, và `config.py:72` chỉ đọc
   `technical_thresholds`. Khối `sample_rate_policy` mới sẽ **bị bỏ qua im
   lặng** nếu không thêm hàm đọc tương ứng ở `config.py`. Cùng cơ chế đó giải
   thích vì sao `audio_output.sample_rate` (`audio_profile.yaml:3`) hiện là
   **doc-only** — không code nào đọc tới. Khối mới phải có loader thật, hoặc
   tuyên bố rõ là doc-only.
2. **`technical_thresholds.canonical_nyquist_hz: 12000`** cũng thuộc nhóm này:
   nó là hằng số của đường 24 kHz, xem quyết định ở §6e.
3. **Tên `min_native_sample_rate` gây hiểu nhầm.** Ngưỡng này giờ mang nghĩa
   "thấp hơn mức này thì loại", không phải "mức để đánh dấu native". 22.05 kHz
   dưới ngưỡng bị loại dù không thấp — cùng vấn đề ngữ nghĩa đã nêu ở §6b.

Mọi thay đổi ngưỡng là **versioned config change** — không rải magic constant
trong code (`dsp/admission.py:57` ghi rõ quy tắc này).

---

## 6. Phase 3 — Thay đổi mã nguồn

### 6a. Bỏ resample khỏi render — `dsp/render.py`

- Xoá `CANONICAL_SR`, dùng `src_sr = buf.sample_rate` làm đích
- `resample_to(mono, CANONICAL_SR)` → giữ nguyên mảng
- Bỏ nhánh `linear-degraded` (`render.py:125`) — không còn resampler để hỏng
- **Xoá `_action_for()` (`render.py:105-110`)** — cả 3 nhánh `IDENTITY`/
  `DOWNSAMPLE`/`UPSAMPLE_NEAR_TARGET` đều viết bằng `CANONICAL_SR`; hàm này và
  fallback tại `render.py:151-152` không còn ý nghĩa. Thay bằng
  `admission.canonicalization_action` bắt buộc (không còn fallback suy đoán)
- `RenderRecipe`: bỏ `resample_method`, đổi `target_sample_rate` →
  `output_sample_rate`, bỏ `upsampled_from_below_target`
- `verify_canonical_wav` nhận tần số kỳ vọng từ tham số thay vì hằng số

### 6b. Admission — `dsp/admission.py`

- Bỏ nhánh `NATIVE_OR_HIGHER`→`DOWNSAMPLE` và `NEAR_TARGET_UPSAMPLE`
- `NATIVE_OR_HIGHER` → luôn `IDENTITY` (tần số gốc)
- `NEAR_TARGET_UPSAMPLE` → `LOW_RATE` (loại, vì 22.05 kHz không còn được nâng)
  — cảnh báo ngữ nghĩa: 22.05 kHz bị loại **không phải vì thấp**, mà vì dưới
  `min_native_sample_rate`. Tên `LOW_RATE` cho 22.05k sẽ gây hiểu nhầm khi đọc
  provenance. Đề xuất: dùng nhãn riêng (`BELOW_MIN_NATIVE`) thay vì tái dùng
  `LOW_RATE`. Quyết định tên — không chặn Phase 3.
- Bỏ cờ `CONDITIONAL_UPSAMPLE_NOT_NATIVE_24K` — không còn nâng nên không cần nhãn
- Bỏ `canonical_sample_rate` khỏi `AdmissionConfig` (không còn đích 24k)

### 6b.2 Enum — `contracts/enums.py` (bản plan đầu BỎ SÓT file này)

Đổi logic admission mà không sửa enum sẽ để lại giá trị chết và thiếu giá trị mới:

- `CanonicalizationAction` (`enums.py:58-66`): **xoá** `DOWNSAMPLE` và
  `UPSAMPLE_NEAR_TARGET`; **thêm** `REJECT_SOURCE_HETEROGENEOUS` (theo §6c),
  `LOW_RATE` hoặc tên thay thế đã chốt ở §6b. Giữ `IDENTITY`, `ERROR`, và hai
  `REJECT_LOW_BANDWIDTH`/`REJECT_NARROWBAND`.
- `SourceRateClass` (`enums.py:45-56`): §1.1 dùng ba nhãn
  `HOMOGENEOUS_NATIVE` / `HETEROGENEOUS` / `LOW_RATE` — **không nẳm suy ra từ
  rate đơn lẻ nữa** (tính ở cấp job). Cần chốt: enum này còn giữ (mô tả rate từng
  file, `HETEROGENEOUS` chỉ mang ở cấp job nên không thuộc enum này), hay tách
  thêm enum cấp nguồn. **Không gộp `HETEROGENEOUS` vào `SourceRateClass`** —
  nó là thuộc tính của tập file, không phải của một file.

### 6c. Homogeneity gate (phần lớn nhất — công việc mới)

Hiện **chưa có** khái niệm "nguồn có nhất quán không". Cần:

- `InventoryItem` (`ingestion/hf_downloader.py:110`) mở rộng: thêm
  `sample_rate`, `codec`, `container` — điền từ probe hoặc từ header khi tải.
  Lưu ý: `cli.py:414` đang ghi inventory JSONL bằng dict thô
  `{path,size,sha}` — phải sửa cả writer, không chỉ dataclass.
- `CampaignStore`: lưu **profile tần số/codec của cả job** khi inventory chạy.
  Chỗ ghi là `inventory_file(job_id)` (`store.py:78`) — thêm file profile cạnh
  inventory, không nhét vào từng dòng JSONL.
- Gate mới ở admission, theo cấp **job**: một job có tần số lẫn nhau →
  `REJECT_SOURCE_HETEROGENEOUS`, toàn bộ batch của job bị loại
- Quyết định chưa chốt: heterogeneity đánh giá trên *toàn bộ* file hay trên
  *mẫu*? Toàn bộ thì tốn (tải hết 24 GB); mẫu thì có thể bỏ sót. Đề xuất:
  mẫu 30 file lúc inventory để **gate sớm** (fail-closed, loại nguồn bằng nghi),
  rồi xác nhận lại trên toàn bộ khi đã tải batch đầu.
- Cổng gate phải đặt **trước** `assess_source` trong `pipeline.py:227`, không
  phải sau: `assess_source` quyết theo rate từng file, nên một job lẫn rate vẫn
  lọt từng file trước khi gate cấp job kịp chặn.

### 6d. Sửa các lớp verify và nơi ghi rate (bản plan đầu thiếu 2 chỗ)

| Vị trí | Sửa gì |
|---|---|
| `dsp/render.py:84` | `verify_canonical_wav` so với rate truyền vào |
| `contracts/release.py:62` | so với rate theo policy (A/C) hoặc rate của row (B) |
| `exporters/verify.py:62` | `_check_wav` cần rate kỳ vọng từ manifest, không phải hằng số |
| `canonical/verify.py:48` | `_check_wav_shape` — đọc rate từ row/metadata |
| **`pipeline.py:553`** | `ReleaseRow.sample_rate=24000` → `recipe.output_sample_rate` (mục #7) |
| **`dsp/decode.py:102`** | bỏ `or 24000`, raise `DecodeError` khi không đo được rate (mục #8) |

Hệ quả kéo theo: reason code `SR_NOT_24K` (emit ở `render.py:85`,
`canonical/verify.py:49`, `exporters/verify.py:63`) **phải đổi tên** — tên gọi
cũ nói dối khi rate hợp lệ là 48 kHz. Đề xuất `SR_MISMATCH`. Phải cập nhật cả
`tests/unit/test_render_verify.py:75` đang assert chuỗi này.

### 6e. Tham chiếu băng thông — `dsp/technical.py:190`

`ref_nyq = min(sr/2, t.canonical_nyquist_hz=12000)` được thiết kế để không
đánh oán nguồn 48 kHz sạch. Sau khi bỏ ép 24 kHz, trần 12 kHz **không còn mô tả
đích** nữa. Cần chốt: giữ 12 kHz (đo băng thông theo rate release, hợp lý với
A/C) hay theo Nyquist của từng rate (hợp lý hơn với B/C, nơi 48 kHz là rate hợp
lệ và nội dung >12 kHz là dữ liệu thật). Đây là quyết định phụ của §4.

---

## 7. Phase 4 — Test, tài liệu, dọn dẹp

### Test

- **79 chỗ tham chiếu `24000` trong 27 file test** (đã đếm lại: 79 dòng, 27 file)
  phải rà lại — không xoá mù, phân loại: chỗ nào khẳng định invariant cũ (đổi) vs
  chỗ nào ngẫu nhiên vì fixture (giữ).
- Test bị đụng **ngoài** con số 79 (tìm theo tên, không theo số):
  - `tests/unit/test_infra.py:79` assert `recipe.resample_method == "scipy-resample_poly"` → field bị xoá
  - `tests/unit/test_render_verify.py:62` monkeypatch `resample_poly_quality` để
    ép nhánh `linear-degraded` → nhánh bị xoá, test mất đối tượng
  - `tests/unit/test_render_verify.py:75` assert chuỗi `SR_NOT_24K` → reason code đổi tên
  - `tests/unit/test_admission.py:67` dựng `AdmissionConfig(allow_near_target_upsample=False)`
    → field bị xoá
- Test mới bắt buộc:
  - nguồn đồng nhất 48 kHz → giữ 48 kHz, **không** resample
  - nguồn lẫn 22k/mp3/8k → `REJECT_SOURCE_HETEROGENEOUS`
  - nguồn đồng nhất 16 kHz → `LOW_RATE`
  - `verify_canonical_wav` nhận tần số bất kỳ và vẫn bắt được file sai
  - provenance ghi đúng `output_sample_rate`, không còn `resample_method`
  - **`ReleaseRow.sample_rate` bám theo WAV thật**: dựng row từ job 48 kHz thì
    row phải là 48000 — chốt lỗi #7 (chỉ đổi render mà quên đổi chỗ ghi row thì
    lỗi này lọt qua mọi verify)
  - **decode fail-closed**: file mà ffprobe không trả `sample_rate` → `DecodeError`,
    không được decode ra buffer 24 kHz
  - **`_action_for` không còn**: `render_canonical_wav` không có `admission`
    thì raise, không tự suy đoán action
- Giữ nguyên tinh thần hiện tại: offline, CPU-only, không network — **test chỉ
  chạy local**, không đẩy lên Colab.
- **Môi trường chạy:** các run nặng (probe diện rộng, campaign đầy đủ) có thể
  đẩy lên Colab qua MCP `colab-mcp` (cấu hình project `.mcp.json`); kết quả probe
  mang về local dưới `work/probe/*.json` để pipeline đọc.

### Tài liệu

- `README.md` — đoạn "24 kHz mono PCM16 release" ở phần đầu và mục `audio_output`
- `configs/audio_profile.yaml` — khối `audio_output` + `source_admission`.
  Lưu ý mục #10: khối `audio_output` hiện **không có loader nào đọc**; hoặc xoá,
  hoặc thêm loader và ghi rõ nó là policy thật.
- `exporters/card.py:43,60` — dòng "Audio format: 24 kHz" và "Upsampled/low-bandwidth
  sources are flagged and excluded" phải theo kết quả §4
- `docs/E2E-ORNIX-DATASET.md:125-149` — bảng rate class, invariant I1–I5/I9,
  giải thích "24 kHz output ≠ native 24 kHz source"; `:154` liệt kê
  `resample_method` trong provenance; `:128` còn mô tả `render_canonical_wav
  (upsample bất kỳ)` ở nhánh TRƯỚC; tuyên bố đích 24 kHz ở `:32` (mục tiêu cuối)
  và `:42` (bảng Audio output)
- `docs/CANONICAL-NORMALIZATION.md` — nếu chọn B/C thì viết lại contract
- `docs/MULTI-DATASET-AUTOMATION-PLAN.md` — nếu thêm khái niệm nguồn không nhất quán

---

## 8. Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Mất corpus 44.1/48 kHz dù chất lượng tốt | trung bình (đo được tới nay: 1/5 nguồn — `fpt_fosd`) | Phase 0 probe đo trước; §4 quyết trên số liệu thật |
| Phá contract 6 field nếu chọn phương án B | **cao** | cân nhắc C, hoặc ghi `sample_rate` vào `release_manifest.jsonl` (đã có sẵn field) thay vì `metadata.jsonl` |
| Regression rộng — 79 chỗ test + 4 test tìm theo tên | trung bình | chia nhóm, chạy full suite ở mỗi bước |
| Homogeneity gate tốn I/O nếu đo toàn bộ | trung bình | gate sớm bằng mẫu; xác nhận khi đã tải batch đầu |
| Mất trường `resample_method` trong provenance | thấp | chấp nhận được: không còn resample thì không có ghi |
| Người dùng downstream đang giả định một tần số | trung bình | ghi rõ trong dataset card + CHANGELOG |
| **Provenance ghi sai rate** nếu sót `pipeline.py:553` | **cao** | đây là lỗi im lặng: file đúng, metadata sai, mọi verify đều pass. Test bắt buộc ở §7 là hàng rào duy nhất |
| **`decode.py:102` fail-open** giả định 24 kHz | trung bình | sửa cùng 6d; nếu không, nguồn lỗi vẫn vào được dataset dưới nhãn 24 kHz |
| Reason code `SR_NOT_24K` đổi tên, downstream parse log theo chuỗi | thấp | ghi vào CHANGELOG cùng danh sách chuỗi mới |
| Enum `CanonicalizationAction` còn giá trị chết sau khi xoá nhánh | thấp | 6b.2 liệt kê tường minh |

---

## 9. Đề xuất thứ tự thực thi

1. **Phase 0** — ✅ `scripts/probe_source_rates.py` đã viết (untracked, cần
   commit khi operator duyệt); đã chạy 5/9 nguồn (§3.5). Còn lại: probe 4 nguồn
   chưa đo (`vietbiblevox`, `luna_vi`, `vi_voice_female`, `w2wmovie_voice_2`) —
   chạy local hoặc đẩy lên Colab qua `colab-mcp`; chạy lại `hatrang_voice` +
   `fpt_fosd` với `n=30` cho đủ mẫu
2. **Dừng lại** — ✅ §4 đã chốt Phương án B (operator, 2026-10-02)
3. **Phase 2** — ✅ thay vì thêm block `sample_rate_policy` mới, giữ
   `source_admission.clean_hq` làm nguồn policy duy nhất (bỏ 2 key chết
   `canonical_sample_rate`/`allow_near_target_upsample`); block mới chỉ cần
   khi làm §6c
4. **Phase 3** — ✅ 6a bỏ resample → 6b admission → 6b.2 enum → 6d verify + nơi
   ghi rate. **Hoãn:** 6c homogeneity gate; 6e giữ `canonical_nyquist_hz=12000`
   làm chính sách hiện hành (ghi chú tại §6e)
5. **Phase 4** — ✅ phần chính: test cập nhật + test mới bắt buộc, docs
   README/E2E/CANONICAL-NORMALIZATION/card/audio_profile đồng bộ; còn lại chỉ
   là hệ quả của 6c nếu làm sau
6. Rollback: **chỉ áp dụng cho phần hành vi**. Đặt lại `mode: fixed` +
   `canonical_rate: 24000` đưa *ngưỡng quyết định* về cũ, **nhưng không hoàn
   tác được** phần đổi schema: `RenderRecipe` đã đổi tên field và mất
   `resample_method` (`render.py:34-41`), `CanonicalizationAction` đã mất hai
   giá trị. Rollback thật cần `git revert` các commit Phase 3, không chỉ sửa
   config. Ghi rõ điều này trong commit message từng bước.

## 10. Việc KHÔNG làm trong plan này

- Không nâng `min_native_sample_rate` hay đổi ngưỡng nào khác
- Không đụng detector, policy engine, split, dedup, publisher
  - **Ngoại lệ đã xác định**: resample *phân tích* của detector
    (`vad.py:83`, `quality.py:26`, `music_noise.py:88`) là bắt buộc theo rate
    model và **không** đụng tới. Chỉ bỏ resample *canonical*.
- Không gộp bản sao `_resample_to` của detectors về `dsp/resample.py`, không
  xoá `dsp/resample.py` (thành dead code sau khi 6a xong) — dọn dẹp ở change riêng
- Không tự ý thêm/bớt field nào trước khi §4 được chốt
- Không commit — chờ operator review

---

## 11. Trạng thái thực thi (2026-10-02)

Đã xong (code + full suite local xanh):

- **6a `dsp/render.py`** — bỏ hẳn resample canonical; output giữ **native rate**;
  xoá `CANONICAL_SR`, `_action_for`, nhánh `linear-degraded` và import
  `resample_to`; `RenderRecipe` đổi `target_sample_rate`→`output_sample_rate`,
  xoá `resample_method` + `upsampled_from_below_target`;
  `render_canonical_wav` **bắt buộc** có `admission` (thiếu → `ValueError`);
  `verify_canonical_wav` nhận `expected_sample_rate` (None = bỏ rate check —
  dùng cho đường copy của cây canonical 6-field)
- **6b `dsp/admission.py`** — `NATIVE_OR_HIGHER` → `IDENTITY` với **mọi**
  `sr >= native_min` (không hạ mẫu); `BELOW_MIN_NATIVE` → `REJECT_BELOW_MIN_NATIVE`
  (không upsample); xoá `canonical_sample_rate` + `allow_near_target_upsample`
  khỏi `AdmissionConfig` và `configs/audio_profile.yaml`
- **6b.2 `contracts/enums.py`** — xoá `DOWNSAMPLE`/`UPSAMPLE_NEAR_TARGET`; thêm
  `REJECT_BELOW_MIN_NATIVE`; `NEAR_TARGET_UPSAMPLE` → `BELOW_MIN_NATIVE`
- **6d** — `pipeline.py` ghi `sample_rate=seg_buf.sample_rate` (rate thật của
  WAV); `ReleaseRow.validate` chỉ đòi `sample_rate > 0` + mono PCM16;
  `exporters/verify.py` đối chiếu rate theo row (row thiếu rate → `SR_MISSING`);
  `canonical/verify.py` chỉ kiểm tra rate đọc được; `dsp/decode.py` raise
  `DecodeError` khi ffprobe không trả rate (hết fail-open)
- **Reason codes** — `SR_NOT_24K` → `SR_MISMATCH` (kèm `SR_MISSING` ở release
  verify)
- **Phase 4 (phần chính)** — test cập nhật + test mới: 48 kHz giữ nguyên,
  `render` thiếu admission → raise, decode fail-closed, 22050 bị loại (và được
  nhận lại khi hạ `native_min_sample_rate`); docs README/E2E/
  CANONICAL-NORMALIZATION/card/audio_profile đã đồng bộ theo Phương án B

Chưa làm (có chủ ý):

- **6c homogeneity gate** (profile rate/codec trong inventory + gate cấp job) —
  hoãn theo quyết định operator
- **6e** — giữ `canonical_nyquist_hz=12000` làm chính sách hiện hành; chỉ sửa
  comment cho trung thực, xem lại khi cần
- **Phase 0** — còn 4 nguồn chưa probe + chạy lại `hatrang_voice`/`fpt_fosd`
  với `n=30`; `scripts/probe_source_rates.py` còn untracked
- **`dsp/resample.py`** — vẫn là dead code-ứng-viện (giữ theo §10)
- **Rollback** — theo đúng lời hứa §9: rollback phải bằng `git revert`, không
  thể chỉ sửa config, vì `RenderRecipe`/enum đã đổi schema
