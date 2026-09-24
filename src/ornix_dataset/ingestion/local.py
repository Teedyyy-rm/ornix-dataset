"""Local filesystem source adapter (spec Phase 1).

Scans a directory for audio files, computes source SHA-256 + measured metadata,
and attaches rights via the permission gate. Source bytes are never modified;
staging is a copy/hardlink into an immutable area handled by the pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, Optional

from ..contracts.enums import IngestStatus, RightsStatus
from ..contracts.source import SourceRecord
from ..dsp.admission import AdmissionConfig, classify_lossy, _rate_class
from ..dsp.decode import DecodeError, ffprobe_info
from ..util.hashing import sha256_file, short_id
from ..util.timeutil import utc_now_iso
from .base import SourceAdapter
from .permissions import PermissionGate

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wv"}


class LocalSourceAdapter(SourceAdapter):
    name = "local"

    def __init__(self, root: str, gate: PermissionGate, source_revision: str = "local",
                 metadata: Optional[Dict[str, Dict[str, Any]]] = None, probe: bool = True):
        self.root = root
        self.gate = gate
        self.source_revision = source_revision
        self.metadata = metadata or {}
        self.probe = probe

    def scan(self) -> Iterator[SourceRecord]:
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"source root not found: {self.root}")
        for dirpath, _dirs, files in os.walk(self.root):
            for fn in sorted(files):
                ext = os.path.splitext(fn)[1].lower()
                if ext not in AUDIO_EXTS:
                    continue
                yield self._record(os.path.join(dirpath, fn))

    def _record(self, path: str) -> SourceRecord:
        rel = os.path.relpath(path, self.root)
        uri = f"file://{os.path.abspath(path)}"
        sha = sha256_file(path)
        nbytes = os.path.getsize(path)
        source_id = "SRC_" + short_id(sha, self.source_revision)
        meta = self.metadata.get(rel, self.metadata.get(fn_key(rel), {}))
        info: Dict[str, Any] = {}
        reasons = []
        if self.probe:
            try:
                info = ffprobe_info(path)
            except DecodeError as e:
                reasons.append(f"PROBE_FAILED:{e}")
        decision = self.gate.evaluate(source_id, uri, meta.get("source_license"))
        status = IngestStatus.INGESTED
        if decision.rights_status in (RightsStatus.LICENSE_REVIEW, RightsStatus.UNKNOWN,
                                      RightsStatus.FORBIDDEN):
            status = IngestStatus.QUARANTINE
            reasons.append(f"RIGHTS:{decision.rights_status.value}")
        # preliminary codec/container/rate-class from the probe (extension != codec).
        # Authoritative admission uses the *measured* rate at analyze time.
        codec = info.get("codec_name")
        declared_sr = info.get("sample_rate")
        rate_class = (_rate_class(declared_sr, AdmissionConfig()).value
                      if declared_sr else None)
        return SourceRecord(
            source_id=source_id, source_uri=uri, source_revision=self.source_revision,
            original_file_id=rel, source_sha256=sha, source_bytes=nbytes,
            source_mime=None, source_container=info.get("format_name"),
            source_codec=codec, source_lossy=classify_lossy(codec),
            source_sample_rate=declared_sr, source_channels=info.get("channels"),
            source_duration_s=info.get("duration_s"),
            source_rate_class=rate_class,
            source_license=meta.get("source_license", "UNKNOWN"),
            license_evidence_uri=meta.get("license_evidence_uri"),
            rights_owner=meta.get("rights_owner"), consent_reference=meta.get("consent_reference"),
            rights_status=decision.rights_status,
            redistribution_permitted=decision.redistribution_permitted,
            commercial_training_permitted=decision.commercial_training_permitted,
            attribution_required=bool(meta.get("attribution_required", True)),
            source_split=meta.get("source_split"), source_speaker_ref=meta.get("source_speaker_ref"),
            source_transcript=meta.get("source_transcript"),
            source_language=meta.get("source_language", "vi"),
            ingest_status=status, ingestion_timestamp_utc=utc_now_iso(),
            staged_path=None, reason_codes=reasons,
        )


def fn_key(rel: str) -> str:
    return os.path.basename(rel)
