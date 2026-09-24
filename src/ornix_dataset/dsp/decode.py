"""Decode + probe. ffprobe gives *real* container metadata; decoding uses
soundfile when possible and falls back to an ffmpeg pipe (spec §4 Decode/probe).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .audio import AudioBuffer


class DecodeError(Exception):
    pass


def probe_available() -> bool:
    return shutil.which("ffprobe") is not None


def ffprobe_info(path: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Return measured stream/format metadata via ffprobe (never trusts header)."""
    if not probe_available():
        raise DecodeError("ffprobe not available on PATH")
    cmd = [
        "ffprobe", "-v", "error", "-hide_banner",
        "-show_entries",
        "stream=codec_type,codec_name,sample_rate,channels,bits_per_raw_sample,sample_fmt,duration:format=format_name,duration,size",
        "-of", "json", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
    except subprocess.CalledProcessError as e:
        raise DecodeError(f"ffprobe failed: {e.stderr.decode('utf-8', 'replace')[:400]}")
    except subprocess.TimeoutExpired:
        raise DecodeError("ffprobe timed out")
    data = json.loads(out.stdout or b"{}")
    astream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    if astream is None:
        raise DecodeError("no audio stream found")
    fmt = data.get("format", {})
    dur = _to_float(astream.get("duration")) or _to_float(fmt.get("duration"))
    return {
        "codec_name": astream.get("codec_name"),
        "sample_rate": _to_int(astream.get("sample_rate")),
        "channels": _to_int(astream.get("channels")),
        "bits_per_raw_sample": _to_int(astream.get("bits_per_raw_sample")),
        "sample_fmt": astream.get("sample_fmt"),
        "duration_s": dur,
        "format_name": fmt.get("format_name"),
        "size_bytes": _to_int(fmt.get("size")),
    }


def decode_to_float(
    path: str, mono: bool = False, timeout: float = 120.0
) -> Tuple[AudioBuffer, str]:
    """Decode to float32 samples in [-1, 1]. Returns (buffer, decoder_name)."""
    try:
        import soundfile as sf  # optional dep

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        buf = AudioBuffer(np.asarray(data), int(sr))
        return (buf.to_mono() if mono else buf), f"soundfile-{sf.__version__}"
    except Exception:
        pass  # fall back to ffmpeg for exotic/mislabelled containers
    return _decode_ffmpeg(path, mono=mono, timeout=timeout), "ffmpeg-f32le"


def _decode_ffmpeg(path: str, mono: bool, timeout: float) -> AudioBuffer:
    if shutil.which("ffmpeg") is None:
        raise DecodeError("ffmpeg not available and soundfile decode failed")
    info = ffprobe_info(path)
    ch = 1 if mono else max(1, info.get("channels") or 1)
    sr = info.get("sample_rate") or 24000
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
           "-ac", str(ch), "-ar", str(sr), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
    except subprocess.CalledProcessError as e:
        raise DecodeError(f"ffmpeg decode failed: {e.stderr.decode('utf-8', 'replace')[:400]}")
    except subprocess.TimeoutExpired:
        raise DecodeError("ffmpeg decode timed out")
    arr = np.frombuffer(out.stdout, dtype="<f4").astype(np.float32)
    if ch > 1:
        arr = arr.reshape(-1, ch)
    return AudioBuffer(arr, int(sr))


def _to_int(v: Optional[Any]) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v: Optional[Any]) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
