"""Phase 0 probe — measure the REAL (sample_rate x codec x container) profile
of every campaign dataset before the native-sample-rate policy is fixed.

Read-only. Never writes campaign state, never mutates the source repos. Bytes
are fetched only for a bounded stratified sample: the whole corpus is ~18 GB and
downloading all of it just to count sample rates is not worth it.

Why bytes are necessary at all: ``build_inventory()`` (``hf_downloader.py:161``)
only calls ``list_repo_tree`` -> ``path`` + ``size``. There is no sample rate and
no codec in a repo listing. A header cannot be trusted anyway (spec §6: classify
by the measured codec, never by the file extension), so ffprobe on real bytes is
the only admissible source of truth here.

Sampling is deliberately NOT "the first 30 files": a prefix of a tree is
usually one shard of one recording run, so it reports one format and hides the
variant that only appears at the tail. The sample therefore spreads evenly over
the tree order AND force-includes the largest and smallest file of each dataset
(``spec §3.2``).

Reuses, never reimplements:
  * ``ffprobe_info()``        — dsp/decode.py, the measured-codec probe
  * ``_sniff_ext()``          — campaign/parquet.py, container hint from bytes
  * ``_expected_sha()``       — hf_downloader.py, LFS/Xet OID check
  * campaign manifest         — configs/ornix_campaign*.yaml (same file the
                                real run consumes, so a probe can never drift
                                from what will actually be ingested)

Output: ``work/probe/<dataset>.json`` — ``{sr: count}`` x ``{codec: count}`` x
``{container: count}`` plus ``homogeneous: bool`` and the list of files that
deviate. Standalone: ``python scripts/probe_source_rates.py --help``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

from ornix_dataset.campaign.parquet import _sniff_ext  # noqa: E402
from ornix_dataset.campaign.refs import (canonical_repo_id,  # noqa: E402
                                         default_resolver, pin_revision)
from ornix_dataset.dsp.decode import DecodeError, ffprobe_info  # noqa: E402
from ornix_dataset.ingestion.hf_downloader import _expected_sha  # noqa: E402

# Extensions the real ingest path accepts (``ingestion/local.py:AUDIO_EXTS``).
_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wv")

# Probe defaults (spec §3.2). N=30 files/dataset.
DEFAULT_N = 30
DEFAULT_WORKERS = 8


# --- sampling ---------------------------------------------------------------

def _spread(items: List[Any], n: int) -> List[Any]:
    """Evenly spread ``n`` picks over ``items`` in their existing (tree) order."""
    if not items:
        return []
    if len(items) <= n:
        return list(items)
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def stratified_sample(entries: List[Any], n: int) -> List[Any]:
    """Coverage-first sample: spread over tree order + force largest + smallest.

    A pure prefix (or a pure random draw) can land entirely inside one shard of
    one recording session and therefore report a single homogeneous format for a
    dataset that is actually mixed. Forcing the size extremes as well catches the
    opposite failure: a rare long-form or very short clip whose rate differs.
    """
    if not entries:
        return []
    by_size = sorted(entries, key=lambda e: int(getattr(e, "size", 0) or 0))
    extremes = [by_size[0]]
    if len(by_size) > 1:
        extremes.append(by_size[-1])
    picked: List[Any] = []
    seen = set()
    for e in _spread(sorted(entries, key=lambda e: getattr(e, "path", "")),
                     max(1, n - len(extremes))) + extremes:
        path = getattr(e, "path", "")
        if path in seen:
            continue
        seen.add(path)
        picked.append(e)
    return picked


# --- probing one audio blob -------------------------------------------------

def probe_blob(blob: bytes, name_hint: str) -> Dict[str, Any]:
    """Measure one audio blob with ffprobe. Never trusts the extension.

    The container hint from ``_sniff_ext`` only names the TEMP FILE so ffmpeg
    picks a demuxer quickly; every field reported below comes from ffprobe.
    """
    out: Dict[str, Any] = {"hint": _sniff_ext(blob),
                           "source_name": name_hint,
                           "bytes": len(blob)}
    with tempfile.NamedTemporaryFile(prefix="probe-", suffix=out["hint"],
                                     delete=False) as fh:
        fh.write(blob)
        tmp = fh.name
    try:
        info = ffprobe_info(tmp)
        out["sample_rate"] = info.get("sample_rate")
        out["codec"] = info.get("codec_name")
        out["container"] = info.get("format_name")
        out["channels"] = info.get("channels")
        out["duration_s"] = info.get("duration_s")
        out["ok"] = True
    except DecodeError as e:
        out["ok"] = False
        out["error"] = f"probe-failed:{e}"
    except Exception as e:  # fail-closed: recorded, never silently dropped
        out["ok"] = False
        out["error"] = f"probe-crashed:{type(e).__name__}:{e}"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return out


# --- repo access ------------------------------------------------------------

def _tree_files(api: Any, repo_id: str, sha: str) -> List[Any]:
    """Every file at the pinned commit (no extension filter yet)."""
    out: List[Any] = []
    for entry in api.list_repo_tree(repo_id, revision=sha,
                                    repo_type="dataset", recursive=True):
        if getattr(entry, "type", "file") != "file" and \
                entry.__class__.__name__ != "RepoFile":
            continue
        out.append(entry)
    return out


def _loose_probe(repo_id: str, sha: str, entries: List[Any], n: int,
                 workers: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample loose audio files, download only those, ffprobe each."""
    from huggingface_hub import hf_hub_download

    audio = [e for e in entries
             if os.path.splitext(getattr(e, "path", ""))[1].lower() in _AUDIO_EXTS]
    meta = {"n_audio_files_listed": len(audio), "layout": "loose"}
    if not audio:
        return [], {**meta, "error": "no-loose-audio-files"}

    sample = stratified_sample(audio, n)

    def _one(entry: Any) -> Dict[str, Any]:
        path = getattr(entry, "path", "")
        exp = _expected_sha(entry)
        try:
            local = hf_hub_download(repo_id, path, revision=sha,
                                    repo_type="dataset")
            with open(local, "rb") as fh:
                blob = fh.read()
        except Exception as e:
            return {"ok": False, "source_name": path,
                    "error": f"download-failed:{type(e).__name__}:{e}"}
        rec = probe_blob(blob, path)
        rec["lfs_sha256"] = exp
        if exp and rec.get("bytes") is not None:
            import hashlib
            rec["sha256_match"] = hashlib.sha256(blob).hexdigest() == exp
        return rec

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        records = list(ex.map(_one, sample))
    return records, meta


def _parquet_probe(repo_id: str, sha: str, files: List[Any], n: int,
                   workers: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample clips inside parquet shards WITHOUT downloading whole shards.

    A shard is 0.3-3 GB; downloading one to read 30 clips is the anti-pattern.
    ``HfFileSystem`` + ``pyarrow.parquet`` reads only the row groups actually
    touched, so the cost is proportional to the SAMPLE, not the shard.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    shards = [e for e in files
              if os.path.splitext(getattr(e, "path", ""))[1].lower() == ".parquet"]
    if not shards:
        return [], {"layout": "parquet", "error": "no-parquet-shards"}

    by_size = sorted(shards, key=lambda e: int(getattr(e, "size", 0) or 0))
    # Spread over SHARDS first (coverage across the dataset), then take a few
    # clips from each — the reverse would read 30 clips of one shard.
    n_shards = min(len(shards), max(1, min(6, n // 5)))
    chosen = _spread(sorted(shards, key=lambda e: e.path), n_shards)
    if by_size[0].path not in {s.path for s in chosen}:
        chosen.append(by_size[0])
    if len(by_size) > 1 and by_size[-1].path not in {s.path for s in chosen}:
        chosen.append(by_size[-1])

    per_shard = max(1, n // max(1, len(chosen)))
    fs = HfFileSystem()
    records: List[Dict[str, Any]] = []
    lock_errors: List[str] = []

    def _one_shard(shard: Any) -> List[Dict[str, Any]]:
        spath = shard.path
        out: List[Dict[str, Any]] = []
        try:
            uri = f"datasets/{repo_id}@{sha}/{spath}"
            with fs.open(uri, "rb") as fh:
                pf = pq.ParquetFile(fh)
                n_rg = max(1, pf.num_row_groups)
                # Spread over ROW GROUPS too, then take the first few rows of
                # each: a shard's row groups are usually recording sessions.
                rg_ids = _spread(list(range(n_rg)), per_shard)
                for rg in rg_ids:
                    try:
                        tbl = pf.read_row_group(rg)
                        d = tbl.to_pydict()
                        blobs = self_audio(d)
                    except Exception as e:
                        lock_errors.append(f"{spath}#rg{rg}:{type(e).__name__}:{e}")
                        continue
                    for i, (blob, nm) in enumerate(blobs):
                        if len(out) >= per_shard:
                            break
                        rec = probe_blob(blob, nm)
                        rec["source_name"] = f"{spath}#rg{rg}#row{i}"
                        rec["shard"] = spath
                        out.append(rec)
                    if len(out) >= per_shard:
                        break
        except Exception as e:
            lock_errors.append(f"{spath}:{type(e).__name__}:{e}")
        return out

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(chosen)))) as ex:
        for recs in ex.map(_one_shard, chosen):
            records.extend(recs)

    meta = {"layout": "parquet", "n_parquet_shards_listed": len(shards),
            "shards_sampled": [s.path for s in chosen],
            "n_shard_bytes_skipped": sum(int(getattr(s, "size", 0) or 0)
                                         for s in shards)}
    if lock_errors:
        meta["errors"] = lock_errors
    return records, meta


def self_audio(d: Dict[str, Any]) -> List[Tuple[bytes, str]]:
    """Extract (bytes, name) pairs from a parquet table dict.

    Handles both layouts seen on the Hub: an ``audio:{bytes,path}`` struct and
    flat ``bytes``/``path`` leaves. Mirrors ``expand_parquet_row`` without
    writing anything to disk.
    """
    out: List[Tuple[bytes, str]] = []
    col = d.get("audio")
    if col:
        for a in col:
            if isinstance(a, dict):
                b = a.get("bytes")
                if b:
                    out.append((b, a.get("path") or ""))
            elif isinstance(a, (bytes, bytearray)) and len(a):
                out.append((bytes(a), ""))
        if out:
            return out
    blobs = d.get("bytes") or []
    paths = d.get("path") or [None] * len(blobs)
    for b, p in zip(blobs, paths):
        if b:
            out.append((b, p or ""))
    return out


# --- aggregation ------------------------------------------------------------

def summarize(records: List[Dict[str, Any]], meta: Dict[str, Any],
              n_target: int) -> Dict[str, Any]:
    """Aggregate the per-clip measurements into the Phase 0 report shape."""
    ok = [r for r in records if r.get("ok")]
    failed = [r for r in records if not r.get("ok")]
    sr: Dict[str, int] = {}
    codec: Dict[str, int] = {}
    container: Dict[str, int] = {}
    per_file: Dict[str, List[int]] = {}
    for r in ok:
        s = str(r.get("sample_rate"))
        sr[s] = sr.get(s, 0) + 1
        c = str(r.get("codec"))
        codec[c] = codec.get(c, 0) + 1
        ct = str(r.get("container"))
        container[ct] = container.get(ct, 0) + 1
        per_file.setdefault(r.get("source_name", ""), []).append(
            int(r.get("sample_rate") or 0))

    srs = sorted({int(k) for k in sr if k != "None"})
    codecs = sorted(codec)
    containers = sorted(container)

    # Homogeneity is the §1.1 criterion evaluated on the sample: ONE rate and
    # ONE codec. Several container names that are the same codec family (e.g.
    # "matroska,webm" vs "ogg") do not by themselves break homogeneity.
    homogeneous = len(srs) == 1 and len(codecs) == 1
    dominant_sr = srs[0] if len(srs) == 1 else None
    dominant_codec = codecs[0] if len(codecs) == 1 else None

    deviations: List[Dict[str, Any]] = []
    for r in ok:
        if (homogeneous and int(r.get("sample_rate") or 0) != dominant_sr) or \
           (r.get("codec") != dominant_codec) or \
           (len(srs) > 1 and int(r.get("sample_rate") or 0) != dominant_sr):
            deviations.append({"source_name": r.get("source_name"),
                               "sample_rate": r.get("sample_rate"),
                               "codec": r.get("codec"),
                               "container": r.get("container")})

    low_rate = [d for d in deviations
                if isinstance(d.get("sample_rate"), int)
                and d["sample_rate"] < 24000]
    return {
        **meta,
        "n_target": n_target,
        "n_probed": len(records),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "sample_rate_hist": sr,
        "codec_hist": codec,
        "container_hist": container,
        "distinct_sample_rates": srs,
        "distinct_codecs": codecs,
        "distinct_containers": containers,
        "homogeneous": homogeneous,
        "dominant_sample_rate": dominant_sr,
        "dominant_codec": dominant_codec,
        "deviations": deviations,
        "n_low_rate_under_24k": len(low_rate),
        "failed": [{"source_name": r.get("source_name"),
                    "error": r.get("error")} for r in failed],
    }


# --- driver -----------------------------------------------------------------

def probe_dataset(ds: Dict[str, Any], defaults: Dict[str, Any], n: int,
                  workers: int, token: Optional[str] = None,
                  resolver: Any = default_resolver) -> Dict[str, Any]:
    from huggingface_hub import HfApi

    repo_id = canonical_repo_id(ds["repo"])
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    sha = pin_revision(repo_id, ds.get("revision"), resolver=resolver)

    fmt = ds.get("format", "parquet")
    files = _tree_files(api, repo_id, sha)
    meta_common = {"dataset": ds["name"], "repo": repo_id, "pinned_sha": sha,
                   "format": fmt, "n_files_listed": len(files)}

    if fmt == "loose":
        records, meta = _loose_probe(repo_id, sha, files, n, workers)
    else:
        prefix = ds.get("loose_audio_prefix", "")
        del prefix
        parquet_files = files
        pfx = str(ds.get("parquet_prefix", "data") or "")
        if pfx:
            parquet_files = [f for f in files
                             if getattr(f, "path", "").startswith(pfx)]
        records, meta = _parquet_probe(repo_id, sha, parquet_files, n, workers)

    out = summarize(records, {**meta_common, **meta}, n)
    out["probe_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(REPO, "configs",
                                                      "ornix_campaign.small9.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO, "work", "probe"))
    ap.add_argument("-n", type=int, default=DEFAULT_N,
                    help="clips sampled per dataset (default 30)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="concurrent downloads/probes (default 8)")
    ap.add_argument("--only", default="",
                    help="comma-separated dataset names (default: all)")
    args = ap.parse_args(argv)

    from ornix_dataset.config import load_yaml

    cfg = load_yaml(args.manifest)
    defaults = cfg.get("defaults", {}) or {}
    datasets = cfg.get("datasets", []) or []
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        datasets = [d for d in datasets if d.get("name") in want]
    if not datasets:
        print("[FAIL] no datasets selected", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for ds in datasets:
        t0 = time.time()
        name = ds.get("name", "?")
        try:
            rep = probe_dataset(ds, defaults, args.n, args.workers)
        except Exception as e:
            rep = {"dataset": name, "repo": ds.get("repo"),
                   "error": f"{type(e).__name__}:{e}", "n_ok": 0}
        rep["elapsed_s"] = round(time.time() - t0, 1)
        results.append(rep)
        path = os.path.join(args.out, f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2, sort_keys=True)
        if rep.get("error"):
            print(f"[FAIL] {name}: {rep['error']}")
        else:
            print(f"[ok] {name}: sr={rep['sample_rate_hist']} "
                  f"codec={rep['codec_hist']} homogeneous={rep['homogeneous']} "
                  f"({rep['n_ok']}/{rep['n_probed']} probed, {rep['elapsed_s']}s)")

    with open(os.path.join(args.out, "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())