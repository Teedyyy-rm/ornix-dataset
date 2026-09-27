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
from ..dsp.decode import DecodeError, ffprobe_info, wav_only_enabled
from ..util.hashing import sha256_file, short_id
from ..util.timeutil import utc_now_iso
from .base import SourceAdapter
from .permissions import PermissionGate

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wv"}

# wav-only fast path (default on): non-WAV sources are never probed, hashed or
# staged — transcoding them costs a full decode per file. Set ORNIX_WAV_ONLY=0
# to restore the legacy multi-format scan.
WAV_ONLY_EXTS = {".wav"}


def wanted_exts():
    return WAV_ONLY_EXTS if wav_only_enabled() else AUDIO_EXTS


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
        exts = wanted_exts()
        paths = []
        for dirpath, _dirs, files in os.walk(self.root):
            for fn in sorted(files):
                ext = os.path.splitext(fn)[1].lower()
                if ext not in exts:
                    continue
                paths.append(os.path.join(dirpath, fn))
        # Per-file _record cost is dominated by a per-file ffprobe subprocess plus a
        # full-file sha256 read — both release the GIL, so a thread pool overlaps them
        # across otherwise-idle cores. ThreadPoolExecutor.map preserves input order, so
        # the emitted record sequence (and thus manifest order) is identical to the
        # sequential path. 0/unset = auto (~2x cores). Set =1 for exact legacy behavior.
        try:
            workers = int(os.environ.get("ORNIX_INGEST_PROBE_WORKERS", "0") or 0)
        except ValueError:
            workers = 0
        if workers <= 0:
            workers = max(2, 2 * (os.cpu_count() or 8))
        if workers <= 1 or len(paths) <= 1:
            for p in paths:
                yield self._record(p)
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as ex:
            for rec in ex.map(self._record, paths):
                yield rec

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
            source_license=decision.source_license,
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
