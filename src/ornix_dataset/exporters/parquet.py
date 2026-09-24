"""Parquet + WebDataset exporters (spec §6, Phase 6).

Parquet embeds audio bytes for small clips; WebDataset (TAR) is the option for
large audio. Either satisfies the contract — we don't build both unless needed.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from typing import Any, Dict, List


def write_parquet(rows: List[Dict[str, Any]], audio_dir: str, out_path: str,
                  row_group_size: int = 512) -> str:
    """Write release rows + embedded audio bytes to a Parquet file. Returns path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        raise ValueError("no rows to export")
    audio_bytes = []
    for r in rows:
        with open(os.path.join(audio_dir, os.path.basename(r["audio"])), "rb") as fh:
            audio_bytes.append(fh.read())
    cols: Dict[str, List[Any]] = {k: [r.get(k) for r in rows] for k in rows[0].keys()}
    cols["audio_bytes"] = audio_bytes
    table = pa.table(cols)
    pq.write_table(table, out_path, row_group_size=row_group_size)
    return out_path


def write_webdataset(rows: List[Dict[str, Any]], audio_dir: str, out_dir: str,
                     shard_size: int = 1000) -> List[str]:
    """Write TAR shards: each sample = <id>.wav + <id>.json. Returns shard paths."""
    os.makedirs(out_dir, exist_ok=True)
    shards: List[str] = []
    for shard_idx in range(0, len(rows), shard_size):
        chunk = rows[shard_idx: shard_idx + shard_size]
        shard_path = os.path.join(out_dir, f"shard-{shard_idx // shard_size:05d}.tar")
        with tarfile.open(shard_path, "w") as tar:
            for r in chunk:
                key = r["audio_id"]
                wav_path = os.path.join(audio_dir, os.path.basename(r["audio"]))
                with open(wav_path, "rb") as fh:
                    data = fh.read()
                _add_bytes(tar, f"{key}.wav", data)
                meta = {k: v for k, v in r.items() if k != "audio_bytes"}
                _add_bytes(tar, f"{key}.json", json.dumps(meta, ensure_ascii=False).encode())
        shards.append(shard_path)
    return shards


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0  # deterministic archive
    tar.addfile(info, io.BytesIO(data))
