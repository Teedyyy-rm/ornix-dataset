"""Hugging Face high-speed batch downloader (bounded producer-consumer).

Why a new module (not more code in ``hf.py``): ``hf.py`` owns the *single-file*
primitive (scan one record / materialize one file). This module owns the *batch*
policy that would otherwise push ``hf.py`` past a reasonable responsibility:
a bounded producer-consumer scheduler, a durable per-stage journal, retry
classification with a circuit breaker, disk/RAM guards, and run metrics.
``hf.py`` is reused, not replaced: inventory joins against ``scan()`` records
(rights/identity stay in the adapter) and staging reuses the same immutable
semantics ``materialize()`` enforces.

Transport is never reimplemented: every byte comes from official
``huggingface_hub`` (``hf_hub_download`` per file) and ``hf_xet`` underneath.
Ornix only controls *file-level* concurrency (tier 1); *intra-file*
concurrency stays with Xet (tier 2, env-configured before import).

Data flow per item:
    PENDING -> DOWNLOADING -> DOWNLOADED -> VERIFIED -> STAGED -> READY
Journal states are appended durably at every transition (reuses
``ops.checkpoint.Checkpoint``); only ``DONE`` counts as complete on replay.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import queue
import random
import shutil
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..ops.checkpoint import Checkpoint

# --- configuration -----------------------------------------------------------

#: Hard caps no configuration may exceed (fail-closed, invariant 6).
HARD_MAX_FILE_WORKERS = 32
HARD_MAX_VERIFY_WORKERS = 16
HARD_MAX_READY_QUEUE = 4096

#: RAM floor for Xet high-performance mode (per Xet docs: needs >= 64 GB).
XET_HP_MIN_RAM_BYTES = 64 * (1024 ** 3)

#: Env keys this module understands. All HF_* vars are read by huggingface_hub
#: AT IMPORT TIME — setting them after import has no effect (verified in docs).
MANAGED_ENV_KEYS = (
    "HF_XET_HIGH_PERFORMANCE",
    "HF_XET_FIXED_DOWNLOAD_CONCURRENCY",
    "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY",
)


@dataclass
class DownloadConfig:
    """Batch-download policy. Safe, back-compatible defaults.

    ``file_workers=1`` reproduces the legacy sequential behavior exactly.
    """

    file_workers: int = 1
    verify_workers: int = 4
    max_inflight_bytes: int = 2 * (1024 ** 3)  # downloading + awaiting verify/stage
    ready_queue_max: int = 64
    min_free_disk_bytes: int = 5 * (1024 ** 3)
    retry_max_attempts: int = 3
    retry_base_delay_s: float = 1.0
    retry_deadline_s: float = 300.0
    retry_jitter: bool = True
    token: Optional[str] = None  # default: HF_TOKEN env at call time
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    allow_splits: Optional[List[str]] = None
    allow_empty_inventory: Optional[bool] = None  # None => auto (see below)
    xet_high_performance: bool = False
    xet_hp_allow_low_ram: bool = False  # explicit override, always logged
    xet_fixed_download_concurrency: Optional[int] = None
    # size-profile thresholds (bytes) for the suggestive profile classifier
    small_file_bytes: int = 5 * (1024 ** 2)
    large_file_bytes: int = 500 * (1024 ** 2)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "DownloadConfig":
        raw = dict(raw or {})
        # tolerated aliases (config-facing names -> field names)
        if "retry_deadline_seconds" in raw and "retry_deadline_s" not in raw:
            raw["retry_deadline_s"] = raw.pop("retry_deadline_seconds")
        else:
            raw.pop("retry_deadline_seconds", None)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})

    def effective(self) -> "DownloadConfig":
        """Clamp to hard caps (never exceed, invariant 6)."""
        self.file_workers = max(1, min(self.file_workers, HARD_MAX_FILE_WORKERS))
        self.verify_workers = max(1, min(self.verify_workers, HARD_MAX_VERIFY_WORKERS))
        self.ready_queue_max = max(1, min(self.ready_queue_max, HARD_MAX_READY_QUEUE))
        return self


# --- inventory ---------------------------------------------------------------

@dataclass
class InventoryItem:
    path_in_repo: str
    size: int
    expected_sha256: Optional[str] = None  # LFS/Xet OID when 64-hex, else None
    index: int = 0

    @property
    def key(self) -> str:
        return self.path_in_repo

    @property
    def verify_method(self) -> str:
        return "FULL_SHA256" if self.expected_sha256 else "BOUND_TO_BYTES"


def matches_patterns(path: str, allow: Optional[List[str]],
                      ignore: Optional[List[str]]) -> bool:
    name = path.rsplit("/", 1)[-1]
    if ignore and any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(name, p)
                      for p in ignore):
        return False
    if allow and not any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(name, p)
                         for p in allow):
        return False
    return True


def split_allowed(source_split: Optional[str],
                  allow_splits: Optional[List[str]]) -> bool:
    """Split pre-filter (before any byte is fetched). Records with an unknown
    (None) split can never be excluded: filtering what we cannot see would
    silently drop data."""
    if not allow_splits:
        return True
    if not source_split:
        return True
    return source_split in allow_splits


def _expected_sha(entry: Any) -> Optional[str]:
    lfs = getattr(entry, "lfs", None) or {}
    oid = (lfs.get("oid") if isinstance(lfs, dict) else getattr(lfs, "oid", None))
    if isinstance(oid, str) and len(oid) == 64:
        try:
            int(oid, 16)
            return oid.lower()
        except ValueError:
            return None
    return None


def build_inventory(api: Any, repo_id: str, revision_sha: str,
                    allow: Optional[List[str]] = None,
                    ignore: Optional[List[str]] = None,
                    audio_exts: Optional[Any] = None,
                    allow_empty: Optional[bool] = None) -> List[InventoryItem]:
    """List files at the PINNED commit and filter BEFORE any byte is fetched.

    ``revision_sha`` must be a full commit hash (never a branch name): every
    listing and every download in the run targets the same commit.
    """
    from .local import AUDIO_EXTS
    exts = audio_exts or AUDIO_EXTS
    items: List[InventoryItem] = []
    for entry in api.list_repo_tree(repo_id, revision=revision_sha,
                                    repo_type="dataset", recursive=True):
        if getattr(entry, "type", "file") != "file":
            continue
        path = getattr(entry, "path", "")
        if not any(path.lower().endswith(e) for e in exts):
            continue
        if not matches_patterns(path, allow, ignore):
            continue
        size = getattr(entry, "size", None) or 0
        items.append(InventoryItem(path_in_repo=path, size=int(size),
                                   expected_sha256=_expected_sha(entry),
                                   index=len(items)))
    patterns_given = bool(allow or ignore)
    if not items and patterns_given and allow_empty is not True:
        raise EmptyInventoryError(
            f"allow/ignore patterns matched 0 files in {repo_id}@{revision_sha[:12]} "
            f"(allow={allow} ignore={ignore}); refusing to treat this as success")
    return items


def classify_profile(items: List[InventoryItem],
                     cfg: DownloadConfig) -> Dict[str, Any]:
    """Suggestive size profile (many_small / mixed / few_large).

    Advisory only: it NEVER changes effective workers by itself. Operators pick
    file_workers explicitly; few_large suggests leaving tier-2 (Xet) in charge.
    """
    if not items:
        return {"profile": "empty", "n": 0}
    small = sum(1 for i in items if i.size <= cfg.small_file_bytes)
    large = sum(1 for i in items if i.size >= cfg.large_file_bytes)
    if large and large * 2 >= len(items):
        profile, hint = "few_large", "keep file_workers low (1-2); let Xet scale tier-2"
    elif small * 2 >= len(items):
        profile, hint = "many_small", "tier-1 file_workers help most; Xet adaptive default"
    else:
        profile, hint = "mixed", "moderate file_workers; measure before tuning tier-2"
    return {"profile": profile, "n": len(items), "n_small": small,
            "n_large": large, "total_bytes": sum(i.size for i in items), "hint": hint}


# --- errors ------------------------------------------------------------------

class DownloadError(RuntimeError):
    pass


class EmptyInventoryError(DownloadError):
    pass


class PermanentDownloadError(DownloadError):
    """401/403/404-style: never retried, fail-closed."""


class FailedVerificationError(DownloadError):
    pass


class DiskFullError(DownloadError):
    pass


class RamGuardError(DownloadError):
    pass


class BatchIncompleteError(DownloadError):
    def __init__(self, failures: List[str]):
        super().__init__(f"batch incomplete: {len(failures)} item(s) failed: "
                         f"{failures[:5]}")
        self.failures = failures


def classify_exception(exc: BaseException) -> str:
    """permanent | rate_limited | transient | disk_full. SDK 429-retry stays
    inside huggingface_hub (>=1.2); we only classify what surfaces to us."""
    name = type(exc).__name__
    if name in ("RepositoryNotFoundError", "RevisionNotFoundError",
                "EntryNotFoundError", "RemoteEntryNotFoundError",
                "GatedRepoError", "DisabledRepoError") or "404" in str(exc)[:60]:
        return "permanent"
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return "permanent"
    if status == 404:
        return "permanent"
    if status == 429:
        return "rate_limited"
    if status is not None and 500 <= status <= 599:
        return "transient"
    msg = str(exc).lower()
    if "no space left" in msg or "disk quota" in msg or isinstance(exc, OSError) and \
            getattr(exc, "errno", None) == 28:
        return "disk_full"
    if isinstance(exc, (TimeoutError, ConnectionError)) or "timeout" in msg \
            or "timed out" in msg or "connection" in msg:
        return "transient"
    return "transient"


# --- runtime env (must precede huggingface_hub import) ------------------------

def apply_env(cfg: DownloadConfig) -> List[str]:
    """Apply tier-2 Xet env BEFORE huggingface_hub is imported.

    Returns warnings. If the SDK is already imported the process env is frozen:
    values are compared and mismatches are reported (never silently assumed).
    Never touches secrets.
    """
    desired = {
        "HF_XET_HIGH_PERFORMANCE": "1" if cfg.xet_high_performance else None,
        "HF_XET_FIXED_DOWNLOAD_CONCURRENCY": (
            str(cfg.xet_fixed_download_concurrency)
            if cfg.xet_fixed_download_concurrency else None),
    }
    warnings: List[str] = []
    frozen = "huggingface_hub" in sys.modules
    for key, want in desired.items():
        if want is None:
            continue
        current = os.environ.get(key)
        if frozen:
            if current != want:
                warnings.append(
                    f"{key}: frozen at import (effective={current!r} != "
                    f"configured={want!r}); set it before the process starts")
        else:
            os.environ[key] = want
    if cfg.xet_high_performance:
        MB = 1024 ** 2
        warnings.append("HF_XET_HIGH_PERFORMANCE requested: needs >=64GB RAM, "
                        "buffers can reach tens of GB")
        _ = MB
    return warnings


def effective_env_snapshot() -> Dict[str, Optional[str]]:
    return {k: os.environ.get(k) for k in MANAGED_ENV_KEYS}


# --- results / metrics -------------------------------------------------------

@dataclass
class DownloadResult:
    index: int
    path_in_repo: str
    staged_path: str
    size: int
    sha256: str
    verify_method: str
    from_cache: bool
    reused_staged: bool = False
    t_download_s: float = 0.0
    t_verify_stage_s: float = 0.0


@dataclass
class DownloadMetrics:
    n_items: int = 0
    n_downloaded: int = 0
    n_cache_hit: int = 0
    n_reused_staged: int = 0
    network_bytes: int = 0  # real network bytes ONLY (cache hits excluded)
    cached_bytes: int = 0
    t_download_s: float = 0.0
    t_verify_stage_s: float = 0.0
    peak_active_downloads: int = 0
    peak_ready_depth: int = 0
    peak_rss_mb: float = 0.0  # process peak RSS (ru_maxrss), all threads
    retries: int = 0
    http_429: int = 0
    breaker_trips: int = 0
    wall_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        if self.wall_s > 0:
            d["throughput_MBps"] = round(self.network_bytes / self.wall_s / 1e6, 3)
            d["throughput_Mbps"] = round(self.network_bytes * 8 / self.wall_s / 1e6, 3)
        return d


# --- internal synchronisation primitives -------------------------------------

class _ByteBudget:
    """Blocks the producer while downloading+unstaged bytes exceed the cap."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max(1, max_bytes)
        self._used = 0
        self._cond = threading.Condition()

    def acquire(self, n: int) -> None:
        n = max(n, 1)  # unknown-size items still occupy one unit
        with self._cond:
            while self._used + n > self.max_bytes:
                self._cond.wait()
            self._used += n

    def release(self, n: int) -> None:
        with self._cond:
            self._used = max(0, self._used - max(n, 1))
            self._cond.notify_all()


class _AdaptiveGate:
    """Circuit breaker + worker ceiling for the network stage.

    Fixed pool underneath; this gate enforces the *current* limit. On a 429/5xx
    streak the limit halves (min 1) with a cooldown; it recovers step by step
    on success. All transitions are counted for metrics.
    """

    def __init__(self, limit: int, cooldown_s: float = 30.0,
                 ceiling: Optional[int] = None):
        self.limit = max(1, limit)
        self.floor = 1
        self.cooldown_s = cooldown_s
        self.ceiling = max(1, ceiling or limit)
        self.active = 0
        self.cooldown_until = 0.0
        self.trips = 0
        self._cond = threading.Condition()

    def acquire(self) -> None:
        with self._cond:
            while True:
                now = time.monotonic()
                if now < self.cooldown_until:
                    self._cond.wait(timeout=self.cooldown_until - now)
                    continue
                if self.active < self.limit:
                    self.active += 1
                    return
                self._cond.wait()

    def release(self, ok: bool) -> None:
        with self._cond:
            self.active = max(0, self.active - 1)
            if ok:
                if self.limit < self.ceiling:
                    self.limit += 1
            self._cond.notify_all()

    def trip(self) -> None:
        with self._cond:
            self.limit = max(self.floor, self.limit // 2)
            self.cooldown_until = time.monotonic() + self.cooldown_s
            self.trips += 1
            self._cond.notify_all()


# --- the downloader ----------------------------------------------------------

# journal states (Checkpoint: only DONE counts as complete on replay)
_STAGED_DONE = "DONE"


class HfBatchDownloader:
    """Bounded concurrent batch downloader over official HF transport."""

    def __init__(self, repo_id: str, revision_sha: str, staging_dir: str,
                 cfg: Optional[DownloadConfig] = None,
                 journal: Optional[Checkpoint] = None,
                 download_fn: Optional[Callable[..., str]] = None,
                 cache_check_fn: Optional[Callable[..., Optional[str]]] = None,
                 sleeper: Callable[[float], None] = time.sleep,
                 rng: Optional[random.Random] = None,
                 cache_dir: Optional[str] = None):
        self.repo_id = repo_id
        self.revision_sha = revision_sha
        self.staging_dir = staging_dir
        self.cfg = (cfg or DownloadConfig()).effective()
        self.token = self.cfg.token or os.environ.get("HF_TOKEN")
        self.journal = journal
        self.sleeper = sleeper
        self.rng = rng or random.Random()
        self.cache_dir = cache_dir
        self._download_fn = download_fn or self._default_download
        self._cache_check = cache_check_fn or self._default_cache_check
        self.metrics = DownloadMetrics()
        self._metrics_lock = threading.Lock()
        self._gate = _AdaptiveGate(self.cfg.file_workers,
                                   ceiling=self.cfg.file_workers)
        self._budget = _ByteBudget(self.cfg.max_inflight_bytes)
        self._ready: "queue.Queue[DownloadResult]" = queue.Queue(
            maxsize=self.cfg.ready_queue_max)
        self._peak_active = 0
        self._active = 0
        self._active_lock = threading.Lock()
        self._abort: Optional[BaseException] = None
        self._abort_lock = threading.Lock()
        self._io_open = True
        self._io_submitted = 0
        self._io_done = 0
        self._io_lock = threading.Lock()

    # -- SDK-backed primitives (default; never reimplement transport) ---------

    def _default_download(self, path_in_repo: str, force: bool = False) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=self.repo_id, filename=path_in_repo,
            revision=self.revision_sha, repo_type="dataset",
            token=self.token, force_download=force)

    def _default_cache_check(self, path_in_repo: str) -> Optional[str]:
        from huggingface_hub import try_to_load_from_cache
        try:
            return try_to_load_from_cache(
                repo_id=self.repo_id, filename=path_in_repo,
                revision=self.revision_sha, repo_type="dataset")
        except Exception:
            return None

    # -- guards ---------------------------------------------------------------

    def _check_ram_guard(self) -> None:
        if not self.cfg.xet_high_performance or self.cfg.xet_hp_allow_low_ram:
            return
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            total = 0
        if total and total < XET_HP_MIN_RAM_BYTES:
            raise RamGuardError(
                f"HF_XET_HIGH_PERFORMANCE refused: {total / 1e9:.1f}GB RAM < 64GB; "
                "set xet_hp_allow_low_ram=true to override explicitly")

    def _check_disk(self, need_bytes: int) -> None:
        need = need_bytes * 2 + self.cfg.min_free_disk_bytes  # cache blob + staging copy
        cache_dir = (self.cache_dir or os.environ.get("HF_HUB_CACHE")
                     or (os.path.join(os.environ["HF_HOME"], "hub")
                         if os.environ.get("HF_HOME") else None)
                     or os.path.expanduser("~/.cache/huggingface/hub"))
        for label, path in (("cache", cache_dir), ("staging", self.staging_dir)):
            try:
                free = shutil.disk_usage(path).free
            except OSError:
                os.makedirs(path, exist_ok=True)
                free = shutil.disk_usage(path).free
            if free < need:
                raise DiskFullError(
                    f"free disk on {label} ({free / 1e9:.2f}GB) < required "
                    f"({need / 1e9:.2f}GB incl. {self.cfg.min_free_disk_bytes / 1e9:.1f}GB "
                    "reserve); refusing to enqueue (fail-closed)")

    # -- per-item stages -------------------------------------------------------

    def _note(self, kind: str, **kw: Any) -> None:
        with self._metrics_lock:
            if kind == "retry":
                self.metrics.retries += 1
            elif kind == "429":
                self.metrics.http_429 += 1

    def _download_with_retry(self, item: InventoryItem) -> tuple[str, bool, float]:
        """Returns (cache_path, from_cache, t_download_s). Raises on failure."""
        t0 = time.monotonic()
        cached = self._cache_check(item.path_in_repo)
        if cached and os.path.exists(cached):
            with self._metrics_lock:
                self.metrics.n_cache_hit += 1
                self.metrics.cached_bytes += item.size
            return cached, True, 0.0
        deadline = time.monotonic() + self.cfg.retry_deadline_s
        attempt = 0
        while True:
            attempt += 1
            try:
                path = self._download_fn(item.path_in_repo, force=False)
                return path, False, time.monotonic() - t0
            except Exception as exc:
                kind = classify_exception(exc)
                if kind == "permanent":
                    raise PermanentDownloadError(
                        f"{item.path_in_repo}: {type(exc).__name__}: {exc}") from exc
                if kind == "disk_full":
                    raise DiskFullError(str(exc)) from exc
                if kind == "rate_limited":
                    self._note("429")
                    self._gate.trip()
                if attempt >= self.cfg.retry_max_attempts or \
                        time.monotonic() >= deadline:
                    raise DownloadError(
                        f"{item.path_in_repo}: {type(exc).__name__} after "
                        f"{attempt} attempt(s): {exc}") from exc
                self._note("retry")
                delay = self.cfg.retry_base_delay_s * (2 ** (attempt - 1))
                if self.cfg.retry_jitter:
                    delay *= 0.5 + self.rng.random()
                self.sleeper(delay)

    @staticmethod
    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _stage_atomic(self, src: str, final_name: str) -> str:
        """Copy to temp name in staging dir, fsync, atomic replace, read-only."""
        os.makedirs(self.staging_dir, exist_ok=True)
        dst = os.path.join(self.staging_dir, final_name)
        tmp = os.path.join(self.staging_dir,
                            f".{final_name}.part-{uuid.uuid4().hex[:8]}")
        with open(src, "rb") as fh_in, open(tmp, "wb") as fh_out:
            shutil.copyfileobj(fh_in, fh_out, length=1 << 20)
            fh_out.flush()
            os.fsync(fh_out.fileno())
        os.replace(tmp, dst)
        os.chmod(dst, 0o444)
        return dst

    def _verify_and_stage(self, item: InventoryItem, cache_path: str,
                          from_cache: bool, t_dl: float,
                          staged_name: Callable[[InventoryItem], str]) -> DownloadResult:
        t0 = time.monotonic()
        # never trust a partial cache artifact: only a path returned by a
        # COMPLETED download call is accepted; .incomplete is never globbed.
        if not os.path.exists(cache_path):
            raise FailedVerificationError(
                f"{item.path_in_repo}: download returned missing path {cache_path}")
        if cache_path.endswith(".incomplete"):
            raise FailedVerificationError(
                f"{item.path_in_repo}: refusing .incomplete artifact")
        actual_size = os.path.getsize(cache_path)
        if item.size and actual_size != item.size:
            raise FailedVerificationError(
                f"{item.path_in_repo}: size {actual_size} != inventory {item.size}")
        digest = self._sha256_file(cache_path)
        method = item.verify_method
        if item.expected_sha256 and digest != item.expected_sha256:
            # one forced re-fetch before failing (fail-closed)
            cache_path = self._download_fn(item.path_in_repo, force=True)
            actual_size = os.path.getsize(cache_path)
            digest = self._sha256_file(cache_path)
            if item.size and actual_size != item.size:
                raise FailedVerificationError(
                    f"{item.path_in_repo}: size mismatch after refetch")
            if digest != item.expected_sha256:
                raise FailedVerificationError(
                    f"{item.path_in_repo}: sha256 mismatch after refetch "
                    f"(FAILED_VERIFICATION, isolated, never into QC)")
        staged = self._stage_atomic(cache_path, staged_name(item))
        if self.journal is not None:
            self.journal.mark(item.key, _STAGED_DONE, sha256=digest,
                              staged=os.path.basename(staged), method=method)
        return DownloadResult(index=item.index, path_in_repo=item.path_in_repo,
                              staged_path=staged, size=actual_size, sha256=digest,
                              verify_method=method, from_cache=from_cache,
                              t_download_s=t_dl,
                              t_verify_stage_s=time.monotonic() - t0)

    # -- orchestration ----------------------------------------------------------

    def _item_key(self, item: InventoryItem) -> str:
        return f"{self.repo_id}@{self.revision_sha}/{item.path_in_repo}"

    def _already_staged(self, item: InventoryItem,
                        skip: Dict[str, str]) -> Optional[DownloadResult]:
        staged = skip.get(self._item_key(item))
        if not staged or not os.path.exists(staged):
            return None
        # never trust existence alone: re-verify size + sha before reuse
        if item.size and os.path.getsize(staged) != item.size:
            return None
        digest = self._sha256_file(staged)
        if item.expected_sha256 and digest != item.expected_sha256:
            return None
        with self._metrics_lock:
            self.metrics.n_reused_staged += 1
            self.metrics.cached_bytes += item.size
        return DownloadResult(index=item.index, path_in_repo=item.path_in_repo,
                              staged_path=staged, size=os.path.getsize(staged),
                              sha256=digest, verify_method=item.verify_method,
                              from_cache=True, reused_staged=True)

    def _clean_partials(self) -> None:
        if not os.path.isdir(self.staging_dir):
            return
        for name in os.listdir(self.staging_dir):
            if ".part-" in name:
                try:
                    os.remove(os.path.join(self.staging_dir, name))
                except OSError:
                    pass

    def _set_abort(self, exc: BaseException) -> None:
        with self._abort_lock:
            if self._abort is None:
                self._abort = exc

    def run(self, items: List[InventoryItem],
            staged_name: Optional[Callable[[InventoryItem], str]] = None,
            skip_staged: Optional[Dict[str, str]] = None,
            on_ready: Optional[Callable[[DownloadResult], None]] = None
            ) -> List[DownloadResult]:
        """Download+verify+stage a batch; returns results in INVENTORY ORDER.

        Out-of-order completion never changes manifest order (single ordered
        writer here in the calling thread). Fatal errors stop the producer,
        running tasks finish cleanly, then the first error is raised.
        """
        self._check_ram_guard()
        self._check_disk(sum(i.size for i in items))
        self._clean_partials()
        staged_name = staged_name or (lambda it: os.path.basename(it.path_in_repo))
        skip_staged = skip_staged or {}
        t_start = time.monotonic()
        ordered: Dict[int, DownloadResult] = {}
        failures: List[str] = []

        # fast path: everything already staged+verified (T6 rerun without work)
        reused_map: Dict[int, DownloadResult] = {}
        pending: List[InventoryItem] = []
        for it in items:
            hit = self._already_staged(it, skip_staged)
            if hit is None:
                pending.append(it)
            else:
                reused_map[it.index] = hit
        ordered.update(reused_map)

        def net_job(item: InventoryItem) -> None:
            if self._abort is not None:
                return
            self._gate.acquire()
            with self._active_lock:
                self._active += 1
                self._peak_active = max(self._peak_active, self._active)
            try:
                if self.journal is not None:
                    self.journal.mark(item.key, "DOWNLOADING")
                cache_path, from_cache, t_dl = self._download_with_retry(item)
                if self.journal is not None:
                    self.journal.mark(item.key, "DOWNLOADED")
                with self._metrics_lock:
                    self.metrics.n_downloaded += 1
                    if not from_cache:
                        self.metrics.network_bytes += \
                            os.path.getsize(cache_path) if os.path.exists(cache_path) else 0
                    self.metrics.t_download_s += t_dl
                with self._io_lock:
                    self._io_submitted += 1
                io_pool.submit(io_job, item, cache_path, from_cache, t_dl)
            except BaseException as exc:  # noqa: BLE001 - must not kill the pool
                self._set_abort(exc)
            finally:
                with self._active_lock:
                    self._active -= 1
                self._gate.release(ok=self._abort is None)

        def io_job(item: InventoryItem, cache_path: str, from_cache: bool,
                   t_dl: float) -> None:
            try:
                res = self._verify_and_stage(item, cache_path, from_cache, t_dl,
                                             staged_name)
                with self._metrics_lock:
                    self.metrics.t_verify_stage_s += res.t_verify_stage_s
                self._ready.put(res)  # blocks when full: backpressure (T4)
            except BaseException as exc:  # noqa: BLE001
                self._set_abort(exc)
            finally:
                self._budget.release(item.size)
                with self._io_lock:
                    self._io_done += 1

        try:
            with ThreadPoolExecutor(max_workers=self.cfg.verify_workers,
                                    thread_name_prefix="ornix-io") as io_pool:
                with ThreadPoolExecutor(max_workers=min(self.cfg.file_workers,
                                                         HARD_MAX_FILE_WORKERS),
                                        thread_name_prefix="ornix-net") as net_pool:
                    for item in pending:
                        if self._abort is not None:
                            break
                        self._check_disk(item.size)
                        self._budget.acquire(item.size)  # never submit-all (T3)
                        net_pool.submit(net_job, item)
                # net stage closed: no more io submissions after this point
                with self._io_lock:
                    self._io_open = False
                # drain in COMPLETION order; after an abort results are still
                # consumed (never block a producer) but discarded, never READY.
                deadline = time.monotonic() + 3600
                while time.monotonic() < deadline:
                    with self._io_lock:
                        io_finished = (not self._io_open
                                       and self._io_done >= self._io_submitted)
                    if io_finished and self._ready.empty():
                        break
                    try:
                        res = self._ready.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if self._abort is None:
                        ordered[res.index] = res
                        with self._metrics_lock:
                            self.metrics.peak_ready_depth = max(
                                self.metrics.peak_ready_depth, self._ready.qsize())
                        if on_ready is not None:
                            on_ready(res)
        except KeyboardInterrupt:
            self._set_abort(KeyboardInterrupt())
            raise
        finally:
            with self._metrics_lock:
                self.metrics.peak_active_downloads = self._peak_active
                self.metrics.breaker_trips = self._gate.trips
                self.metrics.n_items = len(items)
                self.metrics.wall_s = time.monotonic() - t_start
                try:
                    import resource
                    # ru_maxrss is KiB on Linux, bytes on macOS
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    self.metrics.peak_rss_mb = round(
                        rss / 1024.0 if sys.platform != "darwin" else rss / 1e6, 2)
                except Exception:
                    pass

        if self._abort is not None:
            exc = self._abort
            if isinstance(exc, (PermanentDownloadError, DiskFullError,
                                RamGuardError, FailedVerificationError,
                                EmptyInventoryError)):
                raise exc
            if isinstance(exc, KeyboardInterrupt):
                raise exc
            raise BatchIncompleteError([f"{type(exc).__name__}: {exc}"]) from exc
        missing = [it.path_in_repo for it in items if it.index not in ordered]
        if missing:
            raise BatchIncompleteError(missing)
        return [ordered[it.index] for it in items]


def download_snapshot_dry_run(repo_id: str, revision_sha: str, token: Optional[str],
                               allow: Optional[List[str]] = None,
                               ignore: Optional[List[str]] = None) -> List[Any]:
    """Alternative inventory via snapshot_download(dry_run=True).

    Kept for benchmark comparison (5.3: choose by measured cost). Returns raw
    DryRunFileInfo rows; callers map to InventoryItem. Requires network.
    """
    from huggingface_hub import snapshot_download
    rows = snapshot_download(repo_id=repo_id, repo_type="dataset",
                             revision=revision_sha, token=token,
                             allow_patterns=allow, ignore_patterns=ignore,
                             dry_run=True)
    assert isinstance(rows, list)
    return rows


def http_env_snapshot() -> Dict[str, Optional[str]]:
    snap = effective_env_snapshot()
    snap["HF_HUB_DOWNLOAD_TIMEOUT"] = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
    snap["HF_HUB_ETAG_TIMEOUT"] = os.environ.get("HF_HUB_ETAG_TIMEOUT")
    return snap
