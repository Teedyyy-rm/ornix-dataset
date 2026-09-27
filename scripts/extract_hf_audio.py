"""Extract a whole HF audio dataset into a local corpus (loose files + metadata.jsonl).

Normalizes THREE on-Hub layouts into the same local shape the LocalSourceAdapter
understands, so the standard ingest/qc/canonical path can consume any of them:

  * ``parquet`` — audio bytes embedded in parquet shards (column ``audio:{bytes,path}``)
  * ``arrow``   — audio bytes embedded in HF ``datasets`` .arrow shards
  * ``loose``   — loose audio files + a separate csv/jsonl transcript table

It extracts the WHOLE dataset in one pass and prints a machine-readable
``RESULT {json}`` line when done. Parquet/arrow shards are still processed one
shard at a time (download shard -> extract rows -> delete shard) so Hub-cache
disk stays bounded; loose files are downloaded concurrently. The orchestrator
runs the full extracted tree through QC once, then deletes it.

Rights are declared by the operator downstream, never guessed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

# Loose-file ingestion keys off the extension, but HF parquet audio commonly stores
# ``path: null`` — recover the container from the leading magic bytes so extracted
# clips are not silently skipped at ingest.
_MAGIC_EXTS = (
    (b"RIFF", ".wav"),      # WAV (RIFF/WAVE)
    (b"fLaC", ".flac"),     # FLAC
    (b"OggS", ".ogg"),      # Ogg (Vorbis/Opus)
    (b"ID3", ".mp3"),       # MP3 with ID3 tag
)
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wv"}


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
    if ext.lower() in _AUDIO_EXTS:
        return base
    stem = root or base or f"clip_{index:06d}"
    return stem + _sniff_ext(blob)


def _meta_row(name, text, lang, speaker_ref):
    row = {"file": name, "source_transcript": text, "source_language": lang}
    if speaker_ref:
        row["source_speaker_ref"] = speaker_ref
    return row


def _write_meta(out, meta):
    mpath = os.path.join(out, "metadata.jsonl")
    with open(mpath, "w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    return mpath


def _speaker_for_row(args, row_speaker):
    if args.speaker_ref:
        return args.speaker_ref
    if args.speaker_col and row_speaker:
        return f"{args.speaker_col}:{row_speaker}"
    return None


def _uniq_name(rel_path, blob, index):
    """Flatten a repo-relative path to a unique local filename with a real ext."""
    flat = rel_path.replace("/", "__")
    root, ext = os.path.splitext(flat)
    if ext.lower() in _AUDIO_EXTS:
        return flat
    return (root or f"clip_{index:06d}") + _sniff_ext(blob)


def _dl_shard(repo, rfilename, tmp_dir):
    """Bulk-download one shard to local disk (fast, resumable) for local reads."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, rfilename, repo_type="dataset",
                           local_dir=tmp_dir)


def extract_parquet(args, api, fs):
    import pyarrow.parquet as pq
    info = api.dataset_info(args.repo, revision="main")
    prefix = args.config_dir or ""
    shards = sorted(s.rfilename for s in (info.siblings or [])
                    if s.rfilename.endswith(".parquet")
                    and (prefix == "" or s.rfilename.startswith(prefix)))
    if not shards:
        shards = sorted(s.rfilename for s in (info.siblings or [])
                        if s.rfilename.endswith(".parquet"))
    if not shards:
        raise SystemExit(f"[FAIL] no parquet shards in {args.repo}")

    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "_shards")
    os.makedirs(tmp, exist_ok=True)
    meta = []
    written = 0
    # Whole-dataset pass: each shard is bulk-downloaded, fully processed, then
    # deleted, so only one shard ever sits on disk at a time.
    for si in range(len(shards)):
        local = _dl_shard(args.repo, shards[si], tmp)
        try:
            pf = pq.ParquetFile(local)
            for rg in range(pf.num_row_groups):
                cols = [args.audio_col, args.text_col]
                if args.speaker_col:
                    cols.append(args.speaker_col)
                tbl = pf.read_row_group(rg, columns=cols)
                audio = tbl.column(args.audio_col).to_pylist()
                text = tbl.column(args.text_col).to_pylist()
                spk = tbl.column(args.speaker_col).to_pylist() if args.speaker_col \
                    else [None] * len(audio)
                for a, t, s in zip(audio, text, spk):
                    b = a.get("bytes") if a else None
                    if not b:
                        continue
                    name = _named(a.get("path"), b, written)
                    with open(os.path.join(args.out, name), "wb") as fh:
                        fh.write(b)
                    meta.append(_meta_row(name, t, args.language, _speaker_for_row(args, s)))
                    written += 1
        finally:
            try:
                os.remove(local)
            except OSError:
                pass
    shutil.rmtree(tmp, ignore_errors=True)
    _write_meta(args.out, meta)
    return {"written": written, "exhausted": True, "sha": info.sha[:12]}


def extract_arrow(args, api, fs):
    import pyarrow as pa
    info = api.dataset_info(args.repo, revision="main")
    shards = sorted(s.rfilename for s in (info.siblings or [])
                    if s.rfilename.endswith(".arrow"))
    if not shards:
        raise SystemExit(f"[FAIL] no .arrow shards in {args.repo}")
    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "_shards")
    os.makedirs(tmp, exist_ok=True)
    meta = []
    written = 0
    # Whole-dataset pass: each shard is bulk-downloaded, fully processed, then
    # deleted, so only one shard ever sits on disk at a time.
    for si in range(len(shards)):
        local = _dl_shard(args.repo, shards[si], tmp)
        try:
            with open(local, "rb") as fh:
                try:
                    reader = pa.ipc.open_stream(fh)
                    batches = list(reader)
                except pa.lib.ArrowInvalid:
                    fh.seek(0)
                    reader = pa.ipc.open_file(fh)
                    batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
                for batch in batches:
                    d = batch.to_pydict()
                    audio = d.get(args.audio_col, [])
                    text = d.get(args.text_col, [None] * len(audio))
                    spk = d.get(args.speaker_col, [None] * len(audio)) if args.speaker_col \
                        else [None] * len(audio)
                    for a, t, s in zip(audio, text, spk):
                        blob = a.get("bytes") if a else None
                        if not blob:
                            continue
                        name = _named(a.get("path"), blob, written)
                        with open(os.path.join(args.out, name), "wb") as w:
                            w.write(blob)
                        meta.append(_meta_row(name, t, args.language, _speaker_for_row(args, s)))
                        written += 1
        finally:
            try:
                os.remove(local)
            except OSError:
                pass
    shutil.rmtree(tmp, ignore_errors=True)
    _write_meta(args.out, meta)
    return {"written": written, "exhausted": True, "sha": info.sha[:12]}


def _load_loose_meta(args):
    """Build {repo_relpath|basename -> transcript} from a csv/jsonl table."""
    from huggingface_hub import hf_hub_download
    if not args.loose_meta:
        return {}
    path = hf_hub_download(args.repo, args.loose_meta, repo_type="dataset")
    fmap = {}

    def put(fn, txt):
        if not fn:
            return
        fmap[fn] = txt
        fmap[os.path.basename(fn)] = txt

    if args.loose_meta_kind == "jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                put(r.get(args.file_col), r.get(args.text_col))
    else:  # csv (sniff pipe vs comma; header optional)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            sample = fh.readline()
            fh.seek(0)
            if "|" in sample and "," not in sample.split("|")[0]:
                for line in fh:
                    parts = line.rstrip("\n").split("|", 1)
                    if len(parts) == 2:
                        put(parts[0], parts[1])
            else:
                for r in csv.DictReader(fh):
                    put(r.get(args.file_col), r.get(args.text_col))
    return fmap


def _loose_workers(args, n_files):
    if getattr(args, "loose_workers", 0):
        try:
            w = max(1, int(args.loose_workers))
            return min(w, n_files)
        except (TypeError, ValueError):
            pass
    return min(max(4, (os.cpu_count() or 8)), n_files or 1)


def extract_loose(args, api, fs):
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    info = api.dataset_info(args.repo, revision="main")
    pfx = args.loose_audio_prefix or ""
    files = sorted(s.rfilename for s in (info.siblings or [])
                   if os.path.splitext(s.rfilename)[1].lower() in _AUDIO_EXTS
                   and s.rfilename.startswith(pfx))
    if not files:
        raise SystemExit(f"[FAIL] no loose audio under {pfx!r} in {args.repo}")
    fmap = _load_loose_meta(args)
    os.makedirs(args.out, exist_ok=True)

    def _one(idx_rel):
        # Downloads are independent (distinct repo files) and hf_hub_download
        # is thread-safe across distinct files; filenames derive from the repo
        # path so concurrent writes never collide.
        idx, rel = idx_rel
        src = hf_hub_download(args.repo, rel, repo_type="dataset")
        with open(src, "rb") as fh:
            blob = fh.read()
        name = _uniq_name(rel, blob, idx)
        with open(os.path.join(args.out, name), "wb") as w:
            w.write(blob)
        txt = fmap.get(rel) or fmap.get(os.path.basename(rel))
        return idx, _meta_row(name, txt, args.language, args.speaker_ref)

    with ThreadPoolExecutor(max_workers=_loose_workers(args, len(files))) as ex:
        rows = sorted(ex.map(_one, enumerate(files)), key=lambda t: t[0])
    meta = [row for _, row in rows]
    _write_meta(args.out, meta)
    return {"written": len(meta), "exhausted": True, "sha": info.sha[:12]}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["parquet", "arrow", "loose"], default="parquet")
    ap.add_argument("--loose-workers", type=int, default=0,
                    help="concurrent downloads for loose format (0=auto)")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--text-col", default="transcription")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--config-dir", default="data", help="parquet shard prefix ('' = all parquet)")
    ap.add_argument("--speaker-ref", default=None, help="fixed speaker ref for every clip")
    ap.add_argument("--speaker-col", default=None, help="per-row speaker column name")
    ap.add_argument("--loose-audio-prefix", default="")
    ap.add_argument("--loose-meta", default=None)
    ap.add_argument("--loose-meta-kind", choices=["csv", "jsonl"], default="csv")
    ap.add_argument("--file-col", default="file_name")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi, HfFileSystem
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    fs = HfFileSystem(token=token)

    if args.format == "parquet":
        res = extract_parquet(args, api, fs)
    elif args.format == "arrow":
        res = extract_arrow(args, api, fs)
    else:
        res = extract_loose(args, api, fs)

    print(f"[ok] repo={args.repo}@{res['sha']} format={args.format} "
          f"wrote {res['written']} clips -> {args.out} (exhausted={res['exhausted']})")
    print("     NOTE: rights are declared by the operator downstream; not inferred here.")
    print("RESULT " + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())

