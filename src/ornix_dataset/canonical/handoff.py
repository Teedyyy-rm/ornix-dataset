"""Ornix-TTS handoff manifest (spec: stable inter-project contract).

``manifest.jsonl`` is a *handoff-only* artifact written at the canonical dataset
root. It exists so Ornix-TTS can ingest Ornix-Datasets directly, and it is
deliberately NOT part of the internal contracts: it does not change, wrap or
replace ``metadata.jsonl`` (the strict six-field training metadata) or
``release_manifest.jsonl`` (the provenance-rich release manifest). Both keep
their own schemas and their own writers.

Only what a consumer cannot recompute is carried over:

    id         <- stable ``Ornix_<digits>`` utterance identity (canonical tree)
    audio      <- ``<split>/audio/<shard>/<file>`` relative to the release root
    transcript <- verified transcript (canonical ``text``)
    language   <- verified compact language code
    speaker_id <- canonical ``spk_<opaque>`` (scoped per source dataset)
    split      <- train | validation | test
    license    <- verified license from rights evidence
    consent    <- verified redistribution scope

Everything Ornix-TTS can derive itself (sample rate, channels, duration,
checksums, quality scores, codec, run/audit ids) is deliberately absent — it
stays in ``release_manifest.jsonl`` and the internal evidence rows.

Determinism: rows are sorted by ``(split, id)`` and serialized with sorted
keys, so the same release always yields byte-identical bytes. The file is
written through the project's atomic writer, so a crash never leaves a
half-written manifest.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..util.io import atomic_write_text
from .layout import SPLITS, read_all_rows
from .schema import SPEAKER_RE, normalize_language

HANDOFF_NAME = "manifest.jsonl"

# Fields carried over from the internal contracts (documented in docs/).
HANDOFF_FIELDS: Tuple[str, ...] = (
    "id", "audio", "transcript", "language", "speaker_id", "split",
    "license", "consent",
)

# A handoff license must be a real identifier. These sentinels mean "we do not
# actually know", and a production handoff must never emit them.
_UNKNOWN_LICENSES = {
    "UNKNOWN", "UNLICENSED", "NONE", "NULL", "N/A", "NA", "",
    "SPECIMEN", "UNKNOWN-OPERATOR-DECLARED",
}
_UNKNOWN_CONSENT = {
    "UNKNOWN", "UNSPECIFIED", "UNLICENSED", "NONE", "NULL", "N/A", "NA", "",
}

# Conservative default derived from verified rights when the source carries no
# explicit consent scope: redistribution was approved, so the guaranteed scope is
# redistribution. Never widened to commercial without evidence.
_DEFAULT_CONSENT = "redistribution-approved"

# Speaker split isolation: the canonical tree already refuses to publish a
# speaker in two splits (canonical/verify.py SPEAKER_SPLIT_LEAK). Keep the
# handoff consistent with that; see docs for the policy note.
_SAFE_REL_RE = re.compile(r"^(?!/)(?!.*\.\./)(?!.*//)(?!.*(?:^|/)\./)(?!.*\.\.$)[^:\\]+$")


class HandoffError(ValueError):
    """Raised when a sample cannot be mapped to a handoff row with certainty."""


@dataclass(frozen=True)
class HandoffRow:
    """One utterance on one line of ``manifest.jsonl``."""

    id: str
    audio: str
    transcript: str
    language: str
    speaker_id: str
    split: str
    license: str
    consent: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "audio": self.audio, "transcript": self.transcript,
            "language": self.language, "speaker_id": self.speaker_id,
            "split": self.split, "license": self.license, "consent": self.consent,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


# --- validation --------------------------------------------------------------

def _require_str(row: Dict[str, Any], field: str, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"{where}: {field} must be a non-empty string")
    return value


def _safe_audio_path(path: str, where: str) -> str:
    """Relative POSIX path from the release root: no absolute, traversal, URL."""
    if not path or path != path.strip():
        raise HandoffError(f"{where}: audio path must be non-empty and trimmed")
    if path.startswith("/") or path.startswith("\\"):
        raise HandoffError(f"{where}: audio path must be relative, got {path!r}")
    if "://" in path or ":" in path.split("/")[0]:
        raise HandoffError(f"{where}: audio path must not be a URL, got {path!r}")
    if "\\" in path:
        raise HandoffError(f"{where}: audio path must use POSIX separators, got {path!r}")
    if not _SAFE_REL_RE.match(path):
        raise HandoffError(f"{where}: unsafe audio path {path!r}")
    head = path.split("/", 1)[0]
    if head not in SPLITS and head != "audio":
        raise HandoffError(
            f"{where}: audio path must be inside a split or audio/, got {path!r}")
    return path


def validate_license(license_id: Any, where: str) -> str:
    """License must be a verified identifier; unknown sentinels are rejected."""
    if not isinstance(license_id, str) or license_id.strip().upper() in _UNKNOWN_LICENSES:
        raise HandoffError(
            f"{where}: license is unknown ({license_id!r}); refusing to hand off")
    return license_id.strip()


def validate_consent(scope: Any, where: str) -> str:
    """Consent must be a verified scope; unknown sentinels are rejected."""
    if not isinstance(scope, str) or scope.strip().upper() in _UNKNOWN_CONSENT:
        raise HandoffError(
            f"{where}: consent is unknown ({scope!r}); refusing to hand off")
    return scope.strip()


def validate_handoff_row(row: Dict[str, Any], dataset_dir: str = "",
                         where: str = "") -> HandoffRow:
    """Validate one handoff record (structure + on-disk audio when given)."""
    rid = _require_str(row, "id", where)
    if where == "":
        where = rid
    audio = _safe_audio_path(_require_str(row, "audio", where), where)
    transcript = _require_str(row, "transcript", where)
    language = normalize_language(_require_str(row, "language", where))
    if not language:
        raise HandoffError(f"{where}: language is not a known code")
    speaker_id = _require_str(row, "speaker_id", where)
    if not SPEAKER_RE.match(speaker_id):
        raise HandoffError(f"{where}: speaker_id is not spk_<opaque>: {speaker_id!r}")
    split = _require_str(row, "split", where)
    if split not in SPLITS:
        raise HandoffError(f"{where}: split must be one of {SPLITS}, got {split!r}")
    if not audio.startswith(split + "/"):
        raise HandoffError(f"{where}: audio path must live under its split {split!r}")
    license_id = validate_license(row.get("license"), where)
    consent = validate_consent(row.get("consent"), where)
    if dataset_dir:
        full = os.path.join(dataset_dir, *audio.split("/"))
        if not os.path.isfile(full):
            raise HandoffError(f"{where}: audio file missing: {audio}")
    return HandoffRow(id=rid, audio=audio, transcript=transcript, language=language,
                      speaker_id=speaker_id, split=split, license=license_id,
                      consent=consent)


# --- exporter ----------------------------------------------------------------

def build_handoff_rows(dataset_dir: str, state_dir: str, *,
                       require_verified_license: bool = True) -> List[HandoffRow]:
    """Map every canonical sample to a handoff row, fail-closed.

    Reads the six-field ``metadata.jsonl`` files (unchanged) for the row content
    and the persistent canonical state for the rights evidence that is not part
    of the public metadata. Nothing is invented: a sample whose rights evidence
    is missing or unknown raises instead of being emitted.
    """
    from .identity import open_state

    rows_by_split = read_all_rows(dataset_dir)
    with open_state(state_dir) as state:
        provenance = dict(state.data.get("provenance", {}))
        ids: Dict[str, str] = {}
        for entry in state.data.get("ids", {}).values():
            ids[entry["ornix_id"]] = entry["ornix_id"]

    out: List[HandoffRow] = []
    seen: Dict[str, str] = {}
    for split in SPLITS:
        for file_name, row in sorted(rows_by_split[split].items()):
            ornix_id = file_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            where = ornix_id
            prov = provenance.get(ornix_id)
            if prov is None:
                raise HandoffError(
                    f"{where}: no canonical provenance for this sample; cannot hand off")
            rights_status = prov.get("rights_status") or "UNKNOWN"
            if rights_status != "REDISTRIBUTION_APPROVED" or \
                    not prov.get("redistribution_permitted"):
                raise HandoffError(
                    f"{where}: rights not verified for handoff "
                    f"(status={rights_status}, "
                    f"redistribute={prov.get('redistribution_permitted')})")
            license_id = validate_license(prov.get("source_license"), where) \
                if require_verified_license else prov.get("source_license", "")
            consent = validate_consent(prov.get("consent_scope"), where) \
                if prov.get("consent_scope") else _DEFAULT_CONSENT

            handoff = {
                "id": ornix_id,
                "audio": f"{split}/{row.audio}",
                "transcript": row.text,
                "language": row.language,
                "speaker_id": row.speaker,
                "split": split,
                "license": license_id,
                "consent": consent,
            }
            checked = validate_handoff_row(handoff, dataset_dir=dataset_dir, where=where)
            if checked.id in seen:
                raise HandoffError(
                    f"{checked.id}: duplicate id (already emitted for split "
                    f"{seen[checked.id]})")
            seen[checked.id] = checked.split
            out.append(checked)
    _check_speaker_disjoint(out)
    return out


def _check_speaker_disjoint(rows: Iterable[HandoffRow]) -> None:
    """A speaker must not straddle splits (mirrors canonical/verify.py)."""
    by_speaker: Dict[str, set] = {}
    for row in rows:
        by_speaker.setdefault(row.speaker_id, set()).add(row.split)
    for speaker, splits in sorted(by_speaker.items()):
        if len(splits) > 1:
            raise HandoffError(
                f"{speaker}: speaker appears in multiple splits {sorted(splits)}")


def write_handoff_manifest(dataset_dir: str, state_dir: str, *,
                           require_verified_license: bool = True
                           ) -> Tuple[str, List[HandoffRow]]:
    """Write ``<dataset_dir>/manifest.jsonl``. Returns (path, rows).

    Deterministic: rows sorted by (split, id), sorted keys, atomic replace.
    """
    rows = build_handoff_rows(dataset_dir, state_dir,
                              require_verified_license=require_verified_license)
    rows.sort(key=lambda r: (r.split, r.id))
    body = "".join(r.to_jsonl() + "\n" for r in rows)
    path = os.path.join(dataset_dir, HANDOFF_NAME)
    atomic_write_text(path, body)
    return path, rows