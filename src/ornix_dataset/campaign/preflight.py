"""Destination preflight (G3 / plan R2).

The Hub exposes NO public API for an account's remaining storage quota — the
only authoritative source is the operator's billing page
(https://huggingface.co/settings/billing). Rather than pretend otherwise, the
campaign takes the quota as a CONFIG DECLARATION (`destination.max_bytes`) and
enforces it fail-closed against the destination repo's real measured size. If
the quota is not declared, that is a warning (the run may still be authorized
by the approval receipt's own byte cap), never a silent pass.

Measured facts this module can rely on:
- per-repo size is readable via ``list_repo_tree`` (sums entry sizes);
- the Hub hard-caps a SINGLE file at 500 GB (docs: storage-limits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

HARD_MAX_FILE_BYTES = 500 * 1024**3  # Hub hard limit per file (docs)


@dataclass
class DestinationPreflight:
    ok: bool
    repo_id: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    current_bytes: int = 0
    n_remote_files: int = 0
    max_bytes: Optional[int] = None
    projected_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "repo_id": self.repo_id,
                "reasons": self.reasons, "warnings": self.warnings,
                "current_bytes": self.current_bytes,
                "n_remote_files": self.n_remote_files,
                "max_bytes": self.max_bytes,
                "projected_bytes": self.projected_bytes}


def repo_current_bytes(api: Any, repo_id: str,
                       revision: Optional[str] = None) -> Dict[str, Any]:
    """Measure (bytes, n_files) in the destination repo. Read-only."""
    total, n = 0, 0
    for entry in api.list_repo_tree(repo_id, revision=revision,
                                    repo_type="dataset", recursive=True):
        if getattr(entry, "type", "file") != "file":
            continue
        total += int(getattr(entry, "size", 0) or 0)
        n += 1
    return {"bytes": total, "n_files": n}


def check_destination(api: Any, destination: Optional[Dict[str, Any]],
                      projected_bytes: Optional[int] = None,
                      repo_id: Optional[str] = None) -> DestinationPreflight:
    """Preflight the destination BEFORE an automatic campaign uploads.

    ``projected_bytes`` is what the campaign expects to add (e.g. the sum of
    planned release footprints). Missing/quota-exceeding => ok=False; missing
    write access / unresolvable repo => ok=False.
    """
    dest = destination or {}
    rid = repo_id or dest.get("repo_id")
    if not rid:
        return DestinationPreflight(False, reasons=["NO_DESTINATION_REPO"])
    result = DestinationPreflight(True, repo_id=rid)
    try:
        api.repo_info(rid, repo_type="dataset")  # existence + (implicit) access
    except Exception as e:
        return DestinationPreflight(False, repo_id=rid,
                                    reasons=[f"DESTINATION_UNREACHABLE:{e}"])
    try:
        measured = repo_current_bytes(api, rid, revision=dest.get("revision"))
    except Exception as e:
        return DestinationPreflight(False, repo_id=rid,
                                    reasons=[f"DESTINATION_LIST_FAILED:{e}"])
    result.current_bytes = measured["bytes"]
    result.n_remote_files = measured["n_files"]
    max_bytes = dest.get("max_bytes")
    if max_bytes is None:
        result.warnings.append(
            "QUOTA_NOT_DECLARED: destination.max_bytes unset; only the "
            "approval receipt's byte cap protects the destination")
    else:
        result.max_bytes = int(max_bytes)
        projected = result.current_bytes + int(projected_bytes or 0)
        result.projected_bytes = projected
        if projected > result.max_bytes:
            result.ok = False
            result.reasons.append(
                f"DESTINATION_QUOTA_EXCEEDED: {projected} > {result.max_bytes} "
                f"(current={result.current_bytes}, "
                f"projected_add={int(projected_bytes or 0)})")
    if not dest.get("redistribution_confirmed"):
        result.warnings.append(
            "REDISTRIBUTION_NOT_CONFIRMED: destination.redistribution_confirmed "
            "is unset; operator must confirm rights to redistribute "
            "independently of the approval receipt")
    return result


def expected_upload_bytes(store: Any) -> int:
    """Conservative estimate of what a campaign will add to the destination.

    Uses each batch's accountable bytes (known sizes + unknown-size estimates)
    — the same numbers the resource planner already trusts — so the estimate
    never under-counts relative to the source material.
    """
    total = 0
    try:
        campaign = store.load_campaign()
    except FileNotFoundError:
        return 0
    for job_id in campaign.job_ids:
        job = store.load_job(job_id)
        for bid in job.batch_ids:
            batch = store.load_batch(bid)
            total += int((batch.reservation or {}).get("accountable_bytes",
                                                       batch.total_bytes) or 0)
    return total


def campaign_preflight(store: Any, api: Any = None) -> Dict[str, Any]:
    """Read-only preflight of a whole campaign against its destination (G3).

    Returns a dict with ``ok``; the caller decides whether to refuse to run.
    """
    try:
        campaign = store.load_campaign()
    except FileNotFoundError:
        return {"ok": False, "reasons": ["NO_CAMPAIGN"]}
    dest = campaign.destination or {}
    projected = expected_upload_bytes(store)
    if not dest.get("repo_id"):
        return {"ok": False, "reasons": ["NO_DESTINATION_REPO"],
                "projected_bytes": projected}
    if api is None:
        import os

        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
    rep = check_destination(api, dest, projected_bytes=projected)
    return {**rep.to_dict(), "projected_bytes": projected}


def oversized_files(release_dir: str,
                    limit: int = HARD_MAX_FILE_BYTES) -> List[Dict[str, Any]]:
    """Files in a release dir exceeding the Hub's single-file hard limit."""
    import os

    bad: List[Dict[str, Any]] = []
    for dp, _ds, fs in os.walk(release_dir):
        for f in fs:
            full = os.path.join(dp, f)
            size = os.path.getsize(full)
            if size > limit:
                bad.append({"path": os.path.relpath(full, release_dir),
                            "bytes": size, "limit": limit})
    return bad
