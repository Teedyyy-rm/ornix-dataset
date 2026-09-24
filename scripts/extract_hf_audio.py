"""Extract a subset of an HF *parquet-embedded audio* dataset into a local corpus.

Most HF audio datasets store audio bytes INSIDE parquet shards (column
``audio: {bytes, path}``), not as loose files — so the read-only HfSourceAdapter
(which enumerates loose audio files) cannot see them. This helper streams a
bounded number of row groups (range reads; it does NOT download whole shards),
writes each clip's original bytes to disk, and emits a metadata.jsonl the
LocalSourceAdapter understands. Rights are declared by the operator, not guessed.

Usage:
    python scripts/extract_hf_audio.py --repo doof-ferb/fpt_fosd \
        --out work/fpt_fosd_sample --limit 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Loose-file ingestion (LocalSourceAdapter) keys off the file extension, but HF
# parquet audio commonly stores ``path: null`` — so the original container has to
# be recovered from the leading magic bytes, else the extracted clips land with
# no extension and are silently skipped at ingest.
_MAGIC_EXTS = (
    (b"RIFF", ".wav"),      # WAV (RIFF/WAVE)
    (b"fLaC", ".flac"),     # FLAC
    (b"OggS", ".ogg"),      # Ogg (Vorbis/Opus)
    (b"ID3", ".mp3"),       # MP3 with ID3 tag
)


def _sniff_ext(b: bytes) -> str:
    """Best-effort audio container extension from magic bytes (fail-safe .wav)."""
    for magic, ext in _MAGIC_EXTS:
        if b.startswith(magic):
            return ext
    # MPEG audio frame sync (MP3 without an ID3 header): 0xFF Ex/Fx.
    if len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0:
        return ".mp3"
    # ISO-BMFF (m4a/aac): '....ftyp' at offset 4.
    if len(b) >= 12 and b[4:8] == b"ftyp":
        return ".m4a"
    return ".wav"


def _named(path_field, blob: bytes, index: int) -> str:
    """Filename with a real audio extension, recovered from bytes when needed."""
    base = os.path.basename(path_field) if path_field else ""
    root, ext = os.path.splitext(base)
    if ext.lower() in {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}:
        return base
    stem = root or base or f"clip_{index:06d}"
    return stem + _sniff_ext(blob)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config-dir", default="data", help="dir in repo holding parquet shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--text-col", default="transcription")
    ap.add_argument("--language", default="vi")
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi()
    info = api.dataset_info(args.repo, revision="main")
    sha = info.sha
    shards = sorted(s.rfilename for s in (info.siblings or [])
                    if s.rfilename.endswith(".parquet") and s.rfilename.startswith(args.config_dir))
    if not shards:
        print(f"[FAIL] no parquet shards under {args.config_dir}/ in {args.repo}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    fs = HfFileSystem()
    meta = []
    written = 0
    for shard in shards:
        if written >= args.limit:
            break
        pf = pq.ParquetFile(fs.open(f"datasets/{args.repo}/{shard}", "rb"))
        for rg in range(pf.num_row_groups):
            if written >= args.limit:
                break
            tbl = pf.read_row_group(rg, columns=[args.audio_col, args.text_col])
            audio = tbl.column(args.audio_col).to_pylist()
            text = tbl.column(args.text_col).to_pylist()
            for a, t in zip(audio, text):
                if written >= args.limit:
                    break
                b = a.get("bytes")
                if not b:
                    continue
                name = _named(a.get("path"), b, written)
                dst = os.path.join(args.out, name)
                with open(dst, "wb") as fh:
                    fh.write(b)
                meta.append({"file": name, "source_transcript": t,
                             "source_language": args.language})
                written += 1

    mpath = os.path.join(args.out, "metadata.jsonl")
    with open(mpath, "w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[ok] repo={args.repo}@{sha[:12]} wrote {written} clips -> {args.out}")
    print(f"     metadata: {mpath}")
    print("     NOTE: declare rights explicitly in your sources.yaml (do not assume "
          "'public' == redistributable).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
