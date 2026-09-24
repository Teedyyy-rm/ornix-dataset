"""Dataset card (README.md) with HF-compatible YAML frontmatter (spec §6, §12)."""

from __future__ import annotations

from typing import Any, Dict, List


def render_dataset_card(release_id: str, rows: List[Dict[str, Any]],
                        rights_report: Dict[str, Any], quality_report: Dict[str, Any],
                        changelog: str = "") -> str:
    langs = sorted({r.get("language", "vi") for r in rows})
    splits = sorted({r.get("split", "train") for r in rows})
    licenses = sorted({v for v in rights_report.get("licenses", [])}) or ["see-source-terms"]
    n = len(rows)
    total_hours = round(sum(r.get("duration_s", 0) for r in rows) / 3600.0, 4)
    fm = [
        "---",
        f"pretty_name: Ornix Datasets ({release_id})",
        "language:",
        *[f"  - {l}" for l in langs],
        "license:",
        *[f"  - {l}" for l in licenses],
        "task_categories:",
        "  - text-to-speech",
        "  - automatic-speech-recognition",
        "tags:",
        "  - vietnamese",
        "  - speech",
        "  - curated",
        "---",
    ]
    body = f"""
# Ornix Datasets — {release_id}

Vietnamese-first, curated speech dataset. Every published clip passed all
required quality gates and carries explicit redistribution rights.

## Contents
- Clips: **{n}**
- Approx. hours: **{total_hours}**
- Languages: {", ".join(langs)}
- Splits: {", ".join(splits)}
- Audio format: 24 kHz, mono, PCM_S16LE WAV

## Provenance & rights
Sources retain their original attribution and license. This release contains
**only** records with `redistribution_permitted = true`. Gated/public availability
of a source does NOT imply redistribution permission. See `RIGHTS_REPORT.json`.

## Curation method
Fail-closed pipeline: immutable ingest -> technical gate -> VAD/windowing ->
noise/music/quality/speaker detectors -> deterministic policy engine ->
segmentation + re-QC -> dedup + leakage-safe splits. Thresholds are
operator-signed after calibration on a human-labeled gold set. See
`QUALITY_REPORT.json`.

## Limitations
- Detector coverage is bounded by the licensed models available at build time;
  `UNKNOWN` was never promoted to `ACCEPT`.
- Upsampled/low-bandwidth sources are flagged and excluded from "native 24 kHz"
  claims.
- Not a proof of absolute noise-free audio; metrics are evidence, not guarantees.

## Splits & leakage
Train/validation/test are split by speaker / recording-family / source-group with
a saved seed; no group straddles splits.

## Changelog
{changelog or "- Initial release."}
"""
    return "\n".join(fm) + "\n" + body
