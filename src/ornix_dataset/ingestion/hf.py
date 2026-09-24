"""Hugging Face read-only source adapter (spec Phase 1, §12).

Pins the exact commit SHA for a (repo, revision) and enumerates audio files
WITHOUT forcing decode or asserting rights. Being public/gated NEVER implies
redistribution permission — rights come only from the permission gate. Fails
closed if huggingface_hub is missing or the revision cannot be resolved.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from ..contracts.enums import IngestStatus, RightsStatus
from ..contracts.source import SourceRecord
from ..util.hashing import short_id
from ..util.timeutil import utc_now_iso
from .base import SourceAdapter
from .local import AUDIO_EXTS
from .permissions import PermissionGate


class HfSourceAdapter(SourceAdapter):
    name = "hf"

    def __init__(self, repo_id: str, gate: PermissionGate, revision: str = "main",
                 token: Optional[str] = None, metadata: Optional[Dict[str, Dict[str, Any]]] = None,
                 allow_patterns: Optional[List[str]] = None):
        self.repo_id = repo_id
        self.gate = gate
        self.revision = revision
        self.token = token
        self.metadata = metadata or {}
        self.allow_patterns = allow_patterns
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
            source_license=meta.get("source_license", "UNKNOWN"),
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
