"""Generate a SYNTHETIC adversarial gold set for harness/plumbing tests only.

This exercises the calibration runner end-to-end (separation guard, FAR/FRR,
coverage) without any real recordings. It is NOT a substitute for the real,
human-labeled gold set: publishable thresholds MUST be signed off on genuine
audio by an operator (spec §5.1). Synthetic labels only validate the machinery.

Usage:
    python scripts/make_goldset.py --out work/goldset
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from fixtures import synth  # noqa: E402


def _clip(root, clip_id, split, family, is_clean, kind):
    sr = 24000
    if kind == "clean":
        sig = synth.speechlike(sr, 5.0, seed=hash(clip_id) % 1000)
    elif kind == "music":
        sig = synth.speechlike(sr, 5.0, seed=1) + 0.9 * synth.musiclike(sr, 5.0)
    elif kind == "noise":
        sig = synth.speechlike(sr, 5.0, seed=2) * 0.3 + 0.6 * synth.white_noise(sr, 5.0)
    elif kind == "silence":
        sig = synth.silence(sr, 3.0)
    else:
        raise ValueError(kind)
    path = os.path.join(root, f"{clip_id}.wav")
    synth.write_wav(path, sig, sr)
    labels = [] if is_clean else (["MUSIC_BACKGROUND"] if kind == "music" else ["HISS_STATIC"])
    return {"clip_id": clip_id, "path": path, "split": split,
            "source_id": f"src_{clip_id}", "speaker_id": f"spk_{family}",
            "recording_family": family, "is_clean": is_clean,
            "labels": labels, "severity": "N0" if is_clean else "N3",
            "domain": "studio" if is_clean else "field"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="work/goldset")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    # calibration split (families A/B) and heldout split (families C/D): no overlap
    plan = [
        ("cal_clean_1", "calibration", "A", True, "clean"),
        ("cal_clean_2", "calibration", "A", True, "clean"),
        ("cal_music_1", "calibration", "B", False, "music"),
        ("cal_noise_1", "calibration", "B", False, "noise"),
        ("cal_silence_1", "calibration", "B", False, "silence"),
        ("hel_clean_1", "heldout", "C", True, "clean"),
        ("hel_music_1", "heldout", "D", False, "music"),
        ("hel_noise_1", "heldout", "D", False, "noise"),
    ]
    for clip_id, split, family, is_clean, kind in plan:
        rows.append(_clip(args.out, clip_id, split, family, is_clean, kind))

    jsonl = os.path.join(args.out, "goldset.jsonl")
    with open(jsonl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {len(rows)} synthetic gold clips -> {jsonl}")
    print("NOTE: synthetic only; real thresholds require human-labeled audio (spec §5.1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
