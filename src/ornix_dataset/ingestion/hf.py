"""Hugging Face read-only source adapter (spec Phase 1, §12).

Pins the exact commit SHA for a (repo, revision) and enumerates audio files
WITHOUT forcing decode or asserting rights. Being public/gated NEVER implies
redistribution permission — rights come only from the permission gate. Fails
closed if huggingface_hub is missing or the revision cannot be resolved.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from ..contracts.enums import IngestStatus, RightsStatus
from ..contracts.source import SourceRecord
from ..dsp.admission import AdmissionConfig, _rate_class, classify_lossy
from ..dsp.decode import DecodeError, ffprobe_info
from ..util.hashing import sha256_file, short_id
from .base import SourceAdapter
from .hf_downloader import DownloadConfig, matches_patterns
from .local import AUDIO_EXTS
from .permissions import PermissionGate


class HfSourceAdapter(SourceAdapter):
    name = "hf"

    def __init__(self, repo_id: str, gate: PermissionGate, revision: str = "main",
                 token: Optional[str] = None, metadata: Optional[Dict[str, Dict[str, Any]]] = None,
                 allow_patterns: Optional[List[str]] = None,
                 ignore_patterns: Optional[List[str]] = None,
                 download: Optional[Dict[str, Any]] = None):
        self.repo_id = repo_id
        self.gate = gate
        self.revision = revision
        # explicit token wins; otherwise resolved from HF_TOKEN at download time
        # (never logged). None relies on the SDK implicit-token behavior.
        self.token = token
        self.metadata = metadata or {}
        self.allow_patterns = allow_patterns
        self.ignore_patterns = ignore_patterns
        self.download_cfg = DownloadConfig.from_dict(download)
        if allow_patterns and not self.download_cfg.allow_patterns:
            self.download_cfg.allow_patterns = list(allow_patterns)
        if ignore_patterns and not self.download_cfg.ignore_patterns:
            self.download_cfg.ignore_patterns = list(ignore_patterns)
        self.allow_patterns = self.download_cfg.allow_patterns
        self.ignore_patterns = self.download_cfg.ignore_patterns
        self._resolved_sha: Optional[str] = None

    def _api(self):
        try:
            from huggingface_hub import HfApi
        except Exception as e:  # fail-closed
            raise RuntimeError(f"huggingface_hub unavailable: {e}")
        return HfApi(token=self.token)

    def resolve_commit(self) -> str:
        api = self._api()
        info = api.dataset_info(self.repo_id, revision=self.revision)
        sha = getattr(info, "sha", None)
        if not sha:
            raise RuntimeError(f"could not resolve commit SHA for {self.repo_id}@{self.revision}")
        self._resolved_sha = sha
        return sha

    def scan(self) -> Iterator[SourceRecord]:
        sha = self.resolve_commit()
        api = self._api()
        info = api.dataset_info(self.repo_id, revision=sha, files_metadata=True)
        for sib in getattr(info, "siblings", []) or []:
            rfn = sib.rfilename
            if not any(rfn.lower().endswith(e) for e in AUDIO_EXTS):
                continue
            # pre-download filter: patterns apply here so non-matching files
            # never reach the manifest/staging (invariant 5), and the batch
            # inventory joins against exactly these records.
            if not matches_patterns(rfn, self.allow_patterns, self.ignore_patterns):
                continue
            yield self._record(rfn, sib, sha)

    def _record(self, rfilename: str, sibling: Any, sha: str) -> SourceRecord:
        uri = f"hf://datasets/{self.repo_id}@{sha}/{rfilename}"
        # LFS blob sha (if present) is provenance; otherwise derive a stable id.
        lfs = getattr(sibling, "lfs", None)
        blob_sha = (lfs or {}).get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        source_sha = blob_sha or short_id(uri, length=64).ljust(64, "0")
        nbytes = getattr(sibling, "size", None) or 0
        source_id = "SRC_" + short_id(source_sha, sha)
        meta = self.metadata.get(rfilename, {})
        decision = self.gate.evaluate(source_id, uri, meta.get("source_license"))
        status = IngestStatus.INGESTED
        reasons: List[str] = []
        if not blob_sha:
            reasons.append("SOURCE_SHA_UNVERIFIED_FROM_REMOTE")
        if decision.rights_status in (RightsStatus.LICENSE_REVIEW, RightsStatus.UNKNOWN,
                                      RightsStatus.FORBIDDEN):
            status = IngestStatus.QUARANTINE
            reasons.append(f"RIGHTS:{decision.rights_status.value}")
        return SourceRecord(
            source_id=source_id, source_uri=uri, source_revision=sha,
            original_file_id=rfilename, source_sha256=source_sha, source_bytes=int(nbytes),
            source_license=decision.source_license,
            license_evidence_uri=meta.get("license_evidence_uri"),
            rights_status=decision.rights_status,
            redistribution_permitted=decision.redistribution_permitted,
            commercial_training_permitted=decision.commercial_training_permitted,
            attribution_required=bool(meta.get("attribution_required", True)),
            source_speaker_ref=meta.get("source_speaker_ref"),
            source_transcript=meta.get("source_transcript"),
            source_language=meta.get("source_language", "vi"),
            ingest_status=status, ingestion_timestamp_utc=utc_now_iso(),
            reason_codes=reasons,
        )

    def materialize(self, record: SourceRecord, dest_dir: str) -> str:
        """Download the pinned-commit blob, verify its content SHA, stage immutably.

        Downloads ``record.original_file_id`` at the exact resolved commit
        (``record.source_revision``) into ``dest_dir``. When the source carries a
        verified LFS blob sha256, the downloaded bytes MUST match it or we raise
        (fail-closed — never stage unverified content). Returns the staged path
        and sets it read-only (0o444) so the byte-exact provenance is immutable.

        After staging, the staged bytes are probed (ffprobe, no decode) and the
        record's preliminary codec/container/rate-class provenance is patched —
        same preliminary fields ``LocalSourceAdapter`` fills at scan time
        (authoritative admission still uses the *measured* rate at analyze time).
        A probe failure is recorded, never fatal: the staged bytes are valid.
        """
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:  # fail-closed
            raise RuntimeError(f"huggingface_hub unavailable: {e}")
        os.makedirs(dest_dir, exist_ok=True)
        # explicit token (constructor or HF_TOKEN env); never logged.
        token = self.token or os.environ.get("HF_TOKEN")
        local = hf_hub_download(
            repo_id=self.repo_id, filename=record.original_file_id,
            revision=record.source_revision, repo_type="dataset", token=token)
        content_sha = sha256_file(local)
        # only enforce when we actually resolved a remote blob sha (not a derived id)
        if "SOURCE_SHA_UNVERIFIED_FROM_REMOTE" not in record.reason_codes:
            if content_sha != record.source_sha256:
                raise RuntimeError(
                    f"content sha mismatch for {record.source_id}: "
                    f"downloaded {content_sha} != pinned {record.source_sha256}")
        else:
            # no remote blob sha was available; bind provenance to the bytes we got
            record.source_sha256 = content_sha
        ext = os.path.splitext(record.original_file_id)[1] or ".bin"
        staged = os.path.join(dest_dir, f"{record.source_id}{ext}")
        if not os.path.exists(staged):
            import shutil
            shutil.copyfile(local, staged)
            os.chmod(staged, 0o444)  # immutable staging
        record.staged_path = staged
        self._patch_probe_provenance(record, staged)
        return staged

    @staticmethod
    def _patch_probe_provenance(record: SourceRecord, staged_path: str) -> None:
        """Fill preliminary codec/container/rate-class from an ffprobe of staged bytes."""
        try:
            info = ffprobe_info(staged_path)
        except DecodeError as e:
            record.reason_codes = list(record.reason_codes) + [f"PROBE_FAILED:{e}"]
            return
        codec = info.get("codec_name")
        declared_sr = info.get("sample_rate")
        record.source_container = info.get("format_name")
        record.source_codec = codec
        record.source_lossy = classify_lossy(codec)
        record.source_sample_rate = declared_sr
        record.source_channels = info.get("channels")
        record.source_duration_s = info.get("duration_s")
        record.source_rate_class = (_rate_class(declared_sr, AdmissionConfig()).value
                                    if declared_sr else None)
