"""Fetch + checksum-pin OPTIONAL detector weights (spec §4, §8).

Fail-closed philosophy: this script NEVER runs as part of the pipeline. An
operator runs it explicitly to provision licensed weights into ``work/models``.
It downloads each model, records its sha256, and prints a ``models.lock`` stanza
you can paste into ``configs/models.lock.yaml`` after reviewing the license.

Only permissively-licensed, redistributable weights are wired here (Silero VAD =
MIT, DNSMOS code = MIT). PANNs (AudioSet-derived) and pyannote (gated) are NOT
auto-fetched — verify their terms and provide a token yourself.

Usage:
    python scripts/fetch_models.py --dest work/models [--only silero dnsmos]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

# name -> (url, license, note)
SOURCES = {
    "silero": (
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        "MIT",
        "Silero VAD v5 ONNX (16kHz, 512-sample window + state).",
    ),
    "dnsmos": (
        "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "MIT (code) — verify weights terms separately",
        "DNSMOS P.835 sig_bak_ovr ONNX (16kHz, 9.01s window).",
    ),
}

FILENAMES = {"silero": "silero_vad.onnx", "dnsmos": "dnsmos_p835.onnx"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dest_dir: str) -> dict:
    url, lic, note = SOURCES[name]
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, FILENAMES[name])
    req = urllib.request.Request(url, headers={"User-Agent": "ornix-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(out, "wb") as fh:
        fh.write(resp.read())
    return {"name": name, "path": out, "sha256": _sha256(out),
            "bytes": os.path.getsize(out), "license": lic, "note": note}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="work/models")
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), default=sorted(SOURCES))
    args = ap.parse_args(argv)

    results = []
    for name in args.only:
        try:
            info = fetch(name, args.dest)
            results.append(info)
            print(f"[ok] {name}: {info['path']} sha256={info['sha256']} "
                  f"({info['bytes']} bytes, {info['license']})")
        except Exception as e:  # fail-loud, never fake
            print(f"[FAIL] {name}: {e}", file=sys.stderr)

    if results:
        print("\n# Paste into configs/models.lock.yaml (after reviewing licenses):")
        for r in results:
            print(f"#   {r['name']}: model_path={r['path']} "
                  f"weights_sha256={r['sha256']}")
    return 0 if len(results) == len(args.only) else 1


if __name__ == "__main__":
    sys.exit(main())
