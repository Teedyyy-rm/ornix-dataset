"""Dataset card (README.md) for the unified Ornix Dataset (spec §9)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

TASK_CATEGORIES = ("text-to-speech", "automatic-speech-recognition")


def render_card(stats: Dict[str, Any], rights: Dict[str, Any],
                changelog: str = "") -> str:
    langs = sorted(stats.get("languages", {})) or ["und"]
    licenses = sorted(rights.get("licenses", [])) or ["see-source-terms"]
    splits = [s for s in ("train", "validation", "test")
              if stats.get("by_split", {}).get(s, 0) > 0]
    fm = ["---", "pretty_name: Ornix Datasets", "language:",
          *[f"  - {l}" for l in langs], "license:",
          *[f"  - {l}" for l in licenses], "task_categories:",
          *[f"  - {c}" for c in TASK_CATEGORIES], "tags:",
          "  - speech", "  - curated", "  - tts", "---"]
    body = f"""
# Ornix Datasets

Unified, curated speech dataset for Ornix Labs TTS. Multiple upstream datasets
are normalized into **one** namespace: opaque file names, normalized speaker
ids, and exactly six metadata fields. Only clips that passed every quality gate
**and** carry verified redistribution rights are included.

## Audio format
- Canonical WAV, **24 000 Hz, mono, signed PCM16**.
- Every file was re-opened and verified after rendering; a clip whose audio does
  not match its metadata is never published.

## Metadata schema (exactly 6 fields)
```json
{{"audio": "audio/ab/ornix_ab....wav", "text": "...", "file_name": "audio/ab/ornix_ab....wav",
  "speaker": "spk_....", "duration": 2.15, "language": "vi"}}
```
| field | meaning |
|---|---|
| `audio` | repo-relative path to the canonical WAV |
| `text` | verified transcript, matched to the output audio |
| `file_name` | same relative path (Hugging Face `AudioFolder` link column) |
| `speaker` | normalized Ornix speaker id (`spk_<opaque>`) |
| `duration` | seconds, measured from the verified WAV |
| `language` | unified language code |

Provenance (source id/uri, original file name, license, checksums, quality
evidence) is deliberately **not** in the training metadata; it is retained in an
internal audit store.

## Coverage
- Clips: **{stats.get('n_rows', 0)}**
- Speakers: **{stats.get('n_speakers', 0)}**
- Total hours: **{round(stats.get('total_duration_s', 0.0) / 3600.0, 4)}**
- Languages: {", ".join(langs)}
- Splits: {", ".join(splits) or "(none)"}

## Splits
`train` / `validation` / `test` are assigned per speaker/recording-family group
with a saved seed; no group straddles splits (leakage-safe). Empty splits are
not created.

## Loading
The canonical contract is read from `metadata.jsonl` directly, which keeps
`audio` and `file_name` as plain strings:
```python
import json, glob
rows = []
for split in ("train", "validation", "test"):
    for line in open(f"{{split}}/metadata.jsonl", encoding="utf-8"):
        r = json.loads(line)
        assert set(r) == {{"audio", "text", "file_name", "speaker", "duration", "language"}}
        rows.append((split, r))
```
> Note: loading through `datasets`' `AudioFolder` will cast the audio column to
> the `Audio` feature (`array`/`path`/`sampling_rate`) and will not preserve
> `audio: str`. Use `ornix_dataset.canonical.loader.load_ornix_dataset` (or read
> `metadata.jsonl`) when the exact six-field contract is required.

## Processing summary
Fail-closed pipeline: immutable ingest → technical WAV gate + source admission →
VAD/windowing → noise/quality/speaker detectors → deterministic policy → segment
+ re-QC → dedup + leakage-safe split → canonical rename (opaque id) → metadata.

## Known limitations
- Detector coverage is bounded by the licensed models available at build time;
  `UNKNOWN` is never promoted to `ACCEPT`.
- Upsampled/low-bandwidth sources are flagged and excluded from "native 24 kHz"
  claims.
- Metrics are evidence, not proof of absolute noise-free audio.

## License, rights & attribution
Sources retain their original attribution and license. This dataset contains
**only** records with verified redistribution permission
(`redistribution_permitted = true`). Public/gated availability of a source does
**not** imply redistribution permission. See `RELEASE_READY.json`.

## Changelog
{changelog or "- Initial canonical export."}
"""
    return "\n".join(fm) + "\n" + body


def write_card(dataset_dir: str, stats: Dict[str, Any], rights: Dict[str, Any],
               changelog: str = "") -> str:
    path = os.path.join(dataset_dir, "README.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_card(stats, rights, changelog))
    return path
