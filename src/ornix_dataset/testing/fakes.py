"""Shared fakes: mock transport, analyzer, Hub (unit + integration + bench)."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ..util.jsonl import append_jsonl


class FakeNet:
    """Mock byte transport: ``__call__(path, force=False)`` -> local file."""

    def __init__(self, tmp):
        self.dir = os.path.join(str(tmp), "net")
        os.makedirs(self.dir, exist_ok=True)
        self.payload: Dict[str, bytes] = {}
        self.calls: List[str] = []
        self.errors: Dict[str, list] = {}  # path -> error factories (popped)
        self.lock = threading.Lock()
        self.barrier: Optional[threading.Barrier] = None
        self.delay = 0.0
        self.handler = None  # optional (path, force) -> str override
        self.peak = 0
        self.active = 0

    def add(self, path: str, data: bytes) -> None:
        self.payload[path] = data

    def sha(self, path: str) -> str:
        return hashlib.sha256(self.payload[path]).hexdigest()

    def __call__(self, path: str, force: bool = False) -> str:
        # NOTE: tests override `handler`, never `__call__`.
        if self.handler is not None:
            with self.lock:
                self.calls.append(path)
            return self.handler(path, force)
        with self.lock:
            self.calls.append(path)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=15)
            if self.delay:
                time.sleep(self.delay)
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


class FakeAnalyzer:
    """Mirrors the analyzer contract: appends its own evidence, returns rows."""

    def __init__(self, delay: float = 0, fail_on: tuple = (),
                 speaker_mod: int = 2):
        self.delay = delay
        self.fail_on = set(fail_on)
        self.speaker_mod = speaker_mod
        self.seen: List[str] = []

    def __call__(self, src: Any, paths: Any, audit: Any) -> Any:
        if self.delay:
            time.sleep(self.delay)
        self.seen.append(src.source_id)
        if src.original_file_id in self.fail_on:
            raise RuntimeError("bad clip")
        append_jsonl(paths.evidence, {
            "source_id": src.source_id, "decision": "ACCEPT"})
        i = len(self.seen)
        row = {"audio_id": src.original_file_id,
               "source_id": src.source_id,
               "speaker_id": f"spk-{i % self.speaker_mod}",
               "source_sha256": src.source_sha256,
               "audio_sha256": src.source_sha256}
        return SimpleNamespace(
            evidences=[{"source_id": src.source_id}],
            accepted=[SimpleNamespace(to_dict=lambda r=row: dict(r))])


class FakeHub:
    """In-memory Hub: remote path -> bytes, with scripted upload failures."""

    def __init__(self):
        self.remote: Dict[str, bytes] = {}
        self.upload_failures: list = []
        self.upload_calls = 0
        self.branches: List[str] = []
        self.missing_repos: set = set()

    def repo_info(self, repo_id: str, repo_type: str = "dataset") -> Dict[str, str]:
        if repo_id in self.missing_repos:
            raise RuntimeError(f"404 repo not found: {repo_id}")
        return {"id": repo_id}

    def create_branch(self, repo_id: str, branch: str,
                      repo_type: str = "dataset", exist_ok: bool = True) -> None:
        self.branches.append(branch)

    def upload_folder(self, folder_path: str, repo_id: str,
                      repo_type: str = "dataset", revision: Optional[str] = None,
                      path_in_repo: Optional[str] = None,
                      commit_message: str = "") -> Any:
        self.upload_calls += 1
        if self.upload_failures:
            raise self.upload_failures.pop(0)
        for dp, _ds, fs in os.walk(folder_path):
            for f in fs:
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, folder_path)
                key = f"{path_in_repo}/{rel}" if path_in_repo else rel
                with open(full, "rb") as fh:
                    self.remote[key] = fh.read()

        class _Commit:
            oid = "c" * 40
        return _Commit()

    def list_repo_files(self, repo_id: str, repo_type: str = "dataset",
                        revision: Optional[str] = None) -> List[str]:
        return list(self.remote)

    def list_repo_tree(self, repo_id: str, revision: Optional[str] = None,
                       repo_type: str = "dataset", recursive: bool = True
                       ) -> List[Any]:
        from types import SimpleNamespace
        return [SimpleNamespace(type="file", path=p, size=len(b), lfs={})
                for p, b in self.remote.items()]


def fake_download_factory(hub: FakeHub):
    def _dl(repo_id: str, filename: str, repo_type: str = "dataset",
            revision: Optional[str] = None) -> str:
        p = f"/tmp/fakehub-{abs(hash(filename)) % 999999}.bin"
        with open(p, "wb") as fh:
            fh.write(hub.remote[filename])
        return p
    return _dl


def wire_hub(monkeypatch: Any, hub: FakeHub) -> None:
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        fake_download_factory(hub))
    monkeypatch.setenv("HF_TOKEN", "dummy")


def approval_for(release_dir: str, revision: str,
                 repo_id: str = "org/dest") -> Dict[str, Any]:
    from ..publishing.approval import release_digest

    return {"release_digest": release_digest(release_dir),
            "repo_id": repo_id, "revision": revision,
            "max_bytes": 10**12, "operator_id": "op",
            "expires_utc": "2999-01-01T00:00:00Z",
            "policy_version": "v1", "license_ack": True}


def accepted_row(audio_id: str, audio_bytes: bytes,
                 release_id: str = "R") -> Dict[str, Any]:
    sha = hashlib.sha256(audio_bytes).hexdigest()
    return {"audio_id": audio_id, "audio": audio_id, "language": "vi",
            "speaker_id": "spk-0", "transcript": "x",
            "sample_rate": 24000, "channels": 1, "encoding": "PCM_S16LE",
            "duration_s": 3.0, "source_id": "SRC_" + sha[:16],
            "source_sha256": sha, "audio_sha256": sha,
            "segment_start_sample_source": 0,
            "segment_end_sample_source": 72000,
            "rights_record_id": "rr", "quality_evidence_id": "qe",
            "quality_policy_version": "v1", "quality_gate": "ACCEPT",
            "split": "train", "release_id": release_id}
