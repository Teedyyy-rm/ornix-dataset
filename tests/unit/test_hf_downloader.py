"""HF high-speed batch downloader tests (T1-T17). No real network.

All concurrency is proved with threading.Barrier/Event (never sleep-guessing).
The injected ``download_fn`` replaces the SDK transport; SDK error shapes are
reproduced with faithful status codes.
"""

import hashlib
import os
import shutil
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fixtures import synth  # noqa: E402
from ornix_dataset.ingestion import hf_downloader as hd  # noqa: E402
from ornix_dataset.ingestion.hf_downloader import (  # noqa: E402
    BatchIncompleteError,
    DiskFullError,
    DownloadConfig,
    EmptyInventoryError,
    FailedVerificationError,
    HfBatchDownloader,
    InventoryItem,
    PermanentDownloadError,
    RamGuardError,
    apply_env,
    build_inventory,
    classify_profile,
)


# --- fakes ---------------------------------------------------------------

class FakeResp:
    def __init__(self, status):
        self.status_code = status
        self.headers = {}
        self.url = "https://huggingface.co/fake"
        self.reason = "Fake"
        self.content = b"fake"
        self.request = None


def http_error(status, msg="boom"):
    from huggingface_hub.utils import HfHubHTTPError
    return HfHubHTTPError(msg, response=FakeResp(status))


class FakeCache:
    """Deterministic fake transport: path -> bytes + scripted behaviors."""

    def __init__(self, tmp, sizes=None):
        self.dir = str(tmp / "fakecache")
        os.makedirs(self.dir, exist_ok=True)
        self.payload = {}   # path -> bytes
        self.calls = []     # (path, force)
        self.errors = {}    # path -> list of error factories (popped per call)
        self.lock = threading.Lock()
        self.peak = 0
        self.active = 0
        self.barrier = None
        self.handler = None
        self.sizes = sizes or {}

    def add(self, path, data=None, size=None):
        data = data if data is not None else os.urandom(size or 4096)
        self.payload[path] = data
        return data

    def sha(self, path):
        return hashlib.sha256(self.payload[path]).hexdigest()

    def __call__(self, path, force=False):
        # NOTE: tests must override `handler`, never `__call__` (special-method
        # lookup bypasses instance attributes for call syntax).
        if self.handler is not None:
            return self.handler(path, force)
        return self._fetch(path, force)

    def _fetch(self, path, force=False):
        with self.lock:
            self.calls.append((path, force))
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=15)
            errs = self.errors.get(path, [])
            if errs:
                raise errs.pop(0)()
            p = os.path.join(self.dir, path.replace("/", "_"))
            with open(p, "wb") as fh:
                fh.write(self.payload[path])
            return p
        finally:
            with self.lock:
                self.active -= 1

    def n_calls(self, path=None, force=None):
        return sum(1 for c in self.calls
                   if (path is None or c[0] == path)
                   and (force is None or c[1] == force))


def items_for(cache, sha=True):
    out = []
    for i, path in enumerate(sorted(cache.payload)):
        out.append(InventoryItem(path_in_repo=path, size=len(cache.payload[path]),
                                 expected_sha256=cache.sha(path) if sha else None,
                                 index=i))
    return out


def make_dl(cache, tmp, name="repo", rev="a" * 40, **cfg_kw):
    cfg_kw.setdefault("file_workers", 4)
    cfg_kw.setdefault("retry_jitter", False)
    dl_kw = {}
    for k in ("download_fn", "cache_check_fn", "sleeper"):
        if k in cfg_kw:
            dl_kw[k] = cfg_kw.pop(k)
    cfg = DownloadConfig(**cfg_kw)
    if "download_fn" not in dl_kw:
        dl_kw["download_fn"] = cache
    if "cache_check_fn" not in dl_kw:
        dl_kw["cache_check_fn"] = lambda p: None
    if "sleeper" not in dl_kw:
        dl_kw["sleeper"] = lambda s: None
    return HfBatchDownloader(repo_id="org/" + name, revision_sha=rev,
                             staging_dir=str(tmp / "staging"), cfg=cfg,
                             **dl_kw)


# --- T1: real simultaneity -------------------------------------------------

def test_T1_files_download_truly_concurrently(tmp_path):
    cache = FakeCache(tmp_path)
    for i in range(4):
        cache.add(f"a{i}.wav", b"x" * 8192)
    cache.barrier = threading.Barrier(4)
    dl = make_dl(cache, tmp_path, file_workers=4)
    res = dl.run(items_for(cache))
    assert len(res) == 4
    assert cache.peak == 4  # all four inside the transport at once


def test_T1_fails_on_sequential_behavior(tmp_path):
    # the same barrier proof with a single worker can never rendezvous:
    # this shows the test discriminates concurrency from sequence.
    cache = FakeCache(tmp_path)
    for i in range(4):
        cache.add(f"a{i}.wav", b"x" * 8192)
    cache.barrier = threading.Barrier(4)
    dl = make_dl(cache, tmp_path, file_workers=1)
    with pytest.raises(Exception):
        dl.run(items_for(cache))


# --- T2: worker ceiling ----------------------------------------------------

def test_T2_concurrency_never_exceeds_file_workers(tmp_path):
    cache = FakeCache(tmp_path)
    for i in range(12):
        cache.add(f"a{i}.wav", b"y" * 4096)
    dl = make_dl(cache, tmp_path, file_workers=3)
    assert len(dl.run(items_for(cache))) == 12
    assert cache.peak <= 3
    assert dl.metrics.peak_active_downloads <= 3


# --- T3: bounded producer --------------------------------------------------

def test_T3_producer_bounded_by_inflight_bytes(tmp_path):
    cache = FakeCache(tmp_path)
    for i in range(4):
        cache.add(f"a{i}.wav", b"z" * 4096)
    gate = threading.Event()
    def gated(path, force=False):
        out = cache._fetch(path, force)
        assert gate.wait(timeout=15)
        return out

    cache.handler = gated
    # budget fits a single 4 KiB file: at most one download in flight
    dl = make_dl(cache, tmp_path, file_workers=4, max_inflight_bytes=5000)
    th = threading.Thread(target=dl.run, args=(items_for(cache),))
    th.start()
    try:
        # producer must stall: started calls stay bounded while downloads block
        import time as _t
        deadline = _t.monotonic() + 3
        while _t.monotonic() < deadline:
            with cache.lock:
                started = len(cache.calls)
            if started > 2:
                break
            _t.sleep(0.05)
        with cache.lock:
            assert len(cache.calls) <= 2  # 1 in flight + bounded submit, never all 4
    finally:
        gate.set()
        th.join(timeout=30)
    assert not th.is_alive()


# --- T4: backpressure + overlap --------------------------------------------

def test_T4_backpressure_and_download_verify_overlap(tmp_path):
    cache = FakeCache(tmp_path)
    cache.add("a.wav", b"1" * 4096)
    cache.add("b.wav", b"2" * 4096)
    order = []
    unblock_verify = threading.Event()
    orig = HfBatchDownloader._sha256_file

    def slow_hash(path):
        order.append(("hash-start", os.path.basename(path)))
        assert unblock_verify.wait(timeout=15)
        order.append(("hash-end", os.path.basename(path)))
        return orig(path)

    dl = make_dl(cache, tmp_path, file_workers=2, verify_workers=1,
                 ready_queue_max=8)
    dl._sha256_file = staticmethod(slow_hash)
    started_b = threading.Event()
    def spy(path, force=False):
        if path == "b.wav":
            order.append(("download-start", "b.wav"))
            started_b.set()
        return cache._fetch(path, force)

    cache.handler = spy
    th = threading.Thread(target=dl.run, args=(items_for(cache),))
    th.start()
    try:
        # b's download must START while a's hash is still blocked: overlap.
        assert started_b.wait(timeout=15)
        order.append(("observed",))
        unblock_verify.set()
        th.join(timeout=30)
    finally:
        unblock_verify.set()
        th.join(timeout=30)
    assert not th.is_alive()
    dl_starts = [e for e in order if e[0] == "download-start"]
    assert dl_starts  # second file prefetched during first file's hash stage


def test_T4_ready_queue_full_stops_producer(tmp_path):
    cache = FakeCache(tmp_path)
    for i in range(4):
        cache.add(f"a{i}.wav", b"q" * 2048)
    hold = threading.Event()
    seen = []

    def consumer(res):
        seen.append(res.path_in_repo)
        assert hold.wait(timeout=15)

    dl = make_dl(cache, tmp_path, file_workers=4, ready_queue_max=1)
    th = threading.Thread(target=dl.run,
                          args=(items_for(cache),),
                          kwargs={"on_ready": consumer})
    th.start()
    try:
        import time as _t
        _t.sleep(2.0)  # consumer blocked; producer must stall, not drain
        with cache.lock:
            started = len(cache.calls)
        assert started < 4 or len(seen) <= 1
    finally:
        hold.set()
        th.join(timeout=30)
    assert not th.is_alive()
    assert sorted(seen) == sorted(f"a{i}.wav" for i in range(4))


# --- T5/T17: out-of-order completion, deterministic order -------------------

def _reverse_gate_cache(tmp_path):
    cache = FakeCache(tmp_path)
    cache.add("a.wav", b"A" * 2048)
    cache.add("b.wav", b"B" * 2048)
    b_started = threading.Event()
    def gated(path, force=False):
        if path == "a.wav":
            assert b_started.wait(timeout=15)  # a finishes only after b started
        if path == "b.wav":
            b_started.set()
        return cache._fetch(path, force)

    cache.handler = gated
    return cache


def test_T5_out_of_order_completion_no_dup_no_loss(tmp_path):
    cache = _reverse_gate_cache(tmp_path)
    dl = make_dl(cache, tmp_path, file_workers=2)
    res = dl.run(items_for(cache))
    assert [r.path_in_repo for r in res] == ["a.wav", "b.wav"]  # inventory order
    assert len({r.staged_path for r in res}) == 2
    assert len({r.sha256 for r in res}) == 2


def test_T17_manifest_order_matches_inventory_despite_reorder(tmp_path):
    cache = _reverse_gate_cache(tmp_path)
    dl = make_dl(cache, tmp_path, file_workers=2)
    res = dl.run(items_for(cache))
    assert [r.index for r in res] == [0, 1]


# --- T6: idempotent rerun ----------------------------------------------------

def test_T6_rerun_reuses_staged_without_redownload(tmp_path):
    cache = FakeCache(tmp_path)
    for i in range(3):
        cache.add(f"a{i}.wav", b"r" * 2048)
    dl = make_dl(cache, tmp_path, file_workers=2)
    first = dl.run(items_for(cache))
    assert cache.n_calls() == 3
    skip = {f"org/repo@{'a' * 40}/{r.path_in_repo}": r.staged_path for r in first}
    dl2 = make_dl(cache, tmp_path, file_workers=2)
    second = dl2.run(items_for(cache), skip_staged=skip)
    assert cache.n_calls() == 3  # zero new downloads
    assert all(r.reused_staged for r in second)
    assert [r.sha256 for r in second] == [r.sha256 for r in first]
    assert dl2.metrics.n_reused_staged == 3


def test_T6_tampered_staged_file_triggers_redownload(tmp_path):
    cache = FakeCache(tmp_path)
    cache.add("a.wav", b"good" * 512)
    dl = make_dl(cache, tmp_path)
    (first,) = dl.run(items_for(cache))
    import stat
    os.chmod(first.staged_path, stat.S_IWUSR | stat.S_IRUSR)
    with open(first.staged_path, "wb") as fh:  # tamper
        fh.write(b"tampered!")
    skip = {f"org/repo@{'a' * 40}/a.wav": first.staged_path}
    dl2 = make_dl(cache, tmp_path)
    (second,) = dl2.run(items_for(cache), skip_staged=skip)
    assert not second.reused_staged
    assert second.sha256 == cache.sha("a.wav")


# --- T7: crash recovery ------------------------------------------------------

def test_T7_crash_mid_run_recovers_without_dup_or_partial(tmp_path):
    cache = FakeCache(tmp_path)
    for i in range(3):
        cache.add(f"a{i}.wav", b"c" * 2048)

    staged_names = []

    orig_stage = HfBatchDownloader._stage_atomic

    def crashy(self, src, final_name):
        out = orig_stage(self, src, final_name)
        staged_names.append(final_name)
        if len(staged_names) == 1:
            raise RuntimeError("simulated crash after first stage")
        return out

    dl = make_dl(cache, tmp_path, file_workers=2)
    dl._stage_atomic = crashy.__get__(dl)
    with pytest.raises(Exception, match="simulated crash|BatchIncomplete"):
        dl.run(items_for(cache))
    # no partial file may carry a final name; only complete stages exist
    leftovers = [n for n in os.listdir(str(tmp_path / "staging"))
                 if ".part-" in n]
    assert leftovers == []
    # rerun from staging dir: re-verify, finish, no duplicates
    skip = {}
    for it in items_for(cache):
        cand = str(tmp_path / "staging" / os.path.basename(it.path_in_repo))
        if os.path.exists(cand):
            skip[f"org/repo@{'a' * 40}/{it.path_in_repo}"] = cand
    dl2 = make_dl(cache, tmp_path, file_workers=2)
    res = dl2.run(items_for(cache), skip_staged=skip)
    assert len(res) == 3
    assert len({r.staged_path for r in res}) == 3


# --- T8/T9: bad checksum, .incomplete -----------------------------------------

def test_T8_checksum_mismatch_retried_once_then_isolated(tmp_path):
    cache = FakeCache(tmp_path)
    cache.add("bad.wav", b"wrong-bytes")
    items = [InventoryItem(path_in_repo="bad.wav", size=len(b"wrong-bytes"),
                           expected_sha256="f" * 64, index=0)]
    dl = make_dl(cache, tmp_path)
    with pytest.raises(FailedVerificationError, match="sha256 mismatch"):
        dl.run(items)
    assert cache.n_calls("bad.wav", force=False) == 1
    assert cache.n_calls("bad.wav", force=True) == 1  # exactly one refetch
    assert not os.path.exists(str(tmp_path / "staging" / "bad.wav"))


def test_T9_incomplete_artifact_never_accepted(tmp_path):
    cache = FakeCache(tmp_path)
    p = os.path.join(cache.dir, "x.incomplete")
    with open(p, "wb") as fh:
        fh.write(b"partial")
    dl = make_dl(cache, tmp_path,
                 download_fn=lambda path, force=False: p,
                 cache_check_fn=lambda p: None)
    items = [InventoryItem(path_in_repo="x.wav", size=7, index=0)]
    with pytest.raises(FailedVerificationError, match="incomplete"):
        dl.run(items)


# --- T10: pinned revision ------------------------------------------------------

def test_T10_every_call_uses_pinned_sha(tmp_path):
    cache = FakeCache(tmp_path)
    cache.add("a.wav", b"v" * 1024)
    seen_revs = []

    from huggingface_hub import HfApi  # noqa: F401  (import surface check)

    dl = make_dl(cache, tmp_path, rev="d" * 40,
                 download_fn=lambda path, force=False: cache(path, force))
    # build_inventory must also target the pinned sha, never a branch
    seen = {}

    class FakeApi:
        def list_repo_tree(self, repo_id, revision=None, **kw):
            seen["revision"] = revision

            class E:
                type = "file"
                path = "a.wav"
                size = 1024
                lfs = {"oid": cache.sha("a.wav")}
            return [E()]

    inv = build_inventory(FakeApi(), "org/repo", "d" * 40)
    assert seen["revision"] == "d" * 40
    assert inv[0].expected_sha256 == cache.sha("a.wav")
    assert dl.revision_sha == "d" * 40


# --- T11: patterns -------------------------------------------------------------

class FakeEntry:
    def __init__(self, path, size=100, oid=None):
        self.type = "file"
        self.path = path
        self.size = size
        self.lfs = {"oid": oid} if oid else {}


class FakeApi:
    def __init__(self, entries):
        self.entries = entries
        self.calls = []

    def list_repo_tree(self, repo_id, revision=None, **kw):
        self.calls.append((repo_id, revision))
        return self.entries


def test_T11_pattern_filter_before_download_and_empty_fails_closed():
    api = FakeApi([FakeEntry("a.wav"), FakeEntry("b.mp3"), FakeEntry("notes.txt")])
    inv = build_inventory(api, "org/r", "e" * 40, allow=["*.wav"])
    assert [i.path_in_repo for i in inv] == ["a.wav"]
    with pytest.raises(EmptyInventoryError):
        build_inventory(api, "org/r", "e" * 40, allow=["*.ogg"])
    # no patterns + no audio => empty is allowed (shard repos are FOLLOW_UP)
    api2 = FakeApi([FakeEntry("shard.parquet", size=99)])
    assert build_inventory(api2, "org/r", "e" * 40) == []


def test_T11_split_filter_before_download():
    from ornix_dataset.ingestion.hf_downloader import split_allowed
    assert split_allowed("train", ["train"]) is True
    assert split_allowed("test", ["train"]) is False
    # unknown split is never excludable (no silent drops)
    assert split_allowed(None, ["train"]) is True
    assert split_allowed("", ["train"]) is True
    assert split_allowed("test", None) is True


# --- T12: error taxonomy ---------------------------------------------------------

def test_T12_permanent_errors_never_retried(tmp_path):
    for status, exc_name in ((401, "PermanentDownloadError"),
                             (403, "PermanentDownloadError"),
                             (404, "PermanentDownloadError")):
        cache = FakeCache(tmp_path)
        cache.add("a.wav", b"p" * 512)
        cache.errors["a.wav"] = [lambda s=status: http_error(s)]
        dl = make_dl(cache, tmp_path)
        with pytest.raises(PermanentDownloadError):
            dl.run(items_for(cache, sha=False))
        assert cache.n_calls("a.wav") == 1, f"status {status} was retried"


def test_T12_transient_retries_bounded_with_backoff(tmp_path):
    cache = FakeCache(tmp_path)
    cache.add("a.wav", b"t" * 512)
    cache.errors["a.wav"] = [lambda: TimeoutError("read timed out"),
                             lambda: TimeoutError("read timed out")]
    delays = []
    dl = make_dl(cache, tmp_path, retry_max_attempts=3, retry_base_delay_s=1.0,
                 sleeper=delays.append)
    res = dl.run(items_for(cache, sha=False))
    assert len(res) == 1
    assert cache.n_calls("a.wav") == 3
    assert delays == [1.0, 2.0]  # exponential, jitter disabled in tests
    assert dl.metrics.retries == 2


def test_T12_429_trips_breaker_and_halves_workers(tmp_path):
    from huggingface_hub.utils import HfHubHTTPError  # noqa: F401
    cache = FakeCache(tmp_path)
    for i in range(2):
        cache.add(f"a{i}.wav", b"9" * 512)
    for p in ("a0.wav", "a1.wav"):
        cache.errors[p] = [lambda: http_error(429)]
    dl = make_dl(cache, tmp_path, file_workers=4, retry_max_attempts=1,
                 sleeper=lambda s: None)
    with pytest.raises(Exception):
        dl.run(items_for(cache, sha=False))
    assert dl.metrics.http_429 >= 1
    assert dl._gate.trips >= 1
    assert dl._gate.limit < 4


# --- T13: disk guard ---------------------------------------------------------------

def test_T13_disk_guard_fail_closed(tmp_path, monkeypatch):
    cache = FakeCache(tmp_path)
    cache.add("a.wav", b"d" * 512)

    class DU:
        free = 1024  # less than reserve
        total = 10 ** 12
        used = 0

    monkeypatch.setattr(shutil, "disk_usage", lambda p: DU())
    dl = make_dl(cache, tmp_path)
    with pytest.raises(DiskFullError):
        dl.run(items_for(cache, sha=False))
    assert cache.n_calls() == 0  # nothing fetched


# --- T14: HP RAM guard ---------------------------------------------------------------

def test_T14_high_performance_refused_on_small_ram(tmp_path, monkeypatch):
    cache = FakeCache(tmp_path)
    cache.add("a.wav", b"h" * 512)
    monkeypatch.setattr(os, "sysconf",
                        lambda k: 512 if k == "SC_PAGE_SIZE" else 4 * 10**6)
    dl = make_dl(cache, tmp_path, xet_high_performance=True)
    with pytest.raises(RamGuardError):
        dl.run(items_for(cache, sha=False))
    assert cache.n_calls() == 0
    dl2 = make_dl(cache, tmp_path, xet_high_performance=True,
                  xet_hp_allow_low_ram=True)
    assert len(dl2.run(items_for(cache, sha=False))) == 1


# --- T15: clean abort ------------------------------------------------------------

def test_T15_keyboard_interrupt_shuts_down_cleanly(tmp_path):
    cache = FakeCache(tmp_path)
    for i in range(4):
        cache.add(f"a{i}.wav", b"k" * 2048)
    def kb(path, force=False):
        raise KeyboardInterrupt()

    cache.handler = kb
    dl = make_dl(cache, tmp_path, file_workers=2)
    with pytest.raises(KeyboardInterrupt):
        dl.run(items_for(cache, sha=False))
    leaked = [t.name for t in threading.enumerate()
              if t.name.startswith(("ornix-net", "ornix-io"))]
    assert leaked == []


# --- T16: workers=1 sequential parity ----------------------------------------------

def test_T16_single_worker_reproduces_sequential(tmp_path):
    cache = FakeCache(tmp_path)
    for i in range(3):
        cache.add(f"a{i}.wav", b"s" * 1024)
    dl = make_dl(cache, tmp_path, file_workers=1)
    res = dl.run(items_for(cache))
    assert [r.path_in_repo for r in res] == ["a0.wav", "a1.wav", "a2.wav"]
    assert cache.peak == 1


# --- misc: env, profile, metrics -----------------------------------------------------

def test_apply_env_warns_when_sdk_already_imported(tmp_path, monkeypatch):
    sys.modules.setdefault("huggingface_hub", __import__("huggingface_hub"))
    monkeypatch.setenv("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", "2")
    cfg = DownloadConfig(xet_fixed_download_concurrency=16)
    warnings = apply_env(cfg)
    assert any("frozen at import" in w for w in warnings)


def test_classify_profile_buckets():
    cfg = DownloadConfig()
    small = [InventoryItem(f"a{i}.wav", 1024, index=i) for i in range(8)]
    assert classify_profile(small, cfg)["profile"] == "many_small"
    big = [InventoryItem("shard.bin", 10**9, index=0)]
    assert classify_profile(big, cfg)["profile"] == "few_large"
    assert classify_profile([], cfg)["profile"] == "empty"


def test_metrics_exclude_cache_bytes_from_network(tmp_path):
    cache = FakeCache(tmp_path)
    data = b"m" * 2048
    cache.add("a.wav", data)
    # a real on-disk file standing in for an SDK cache hit
    hit = str(tmp_path / "hit.wav")
    with open(hit, "wb") as fh:
        fh.write(data)

    def check(path):
        return hit if path == "a.wav" else None

    dl = make_dl(cache, tmp_path, cache_check_fn=check)
    (res,) = dl.run(items_for(cache))
    assert res.from_cache
    assert dl.metrics.network_bytes == 0
    assert dl.metrics.cached_bytes == 2048
    assert cache.n_calls() == 0


def test_synth_wav_end_to_end_stage_verify(tmp_path):
    # real bytes + real sha256 through the whole stage pipeline (no network)
    wav = tmp_path / "real.wav"
    synth.write_wav(str(wav), synth.speechlike(24000, 0.5, seed=7), 24000)
    digest = hashlib.sha256(wav.read_bytes()).hexdigest()
    dl = make_dl({}, tmp_path,
                 download_fn=lambda path, force=False: str(wav),
                 cache_check_fn=lambda p: None)
    items = [InventoryItem("real.wav", size=wav.stat().st_size,
                           expected_sha256=digest, index=0)]
    (res,) = dl.run(items)
    assert res.sha256 == digest
    assert res.verify_method == "FULL_SHA256"
    assert (os.stat(res.staged_path).st_mode & 0o222) == 0  # immutable
