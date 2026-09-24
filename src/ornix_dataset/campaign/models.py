"""Campaign → Job → Batch state models (MD-001).

Identity rules:
- job_id is derived from the NORMALIZED repo_id + pinned SHA — never from a
  raw URL — so re-creating a campaign from equivalent inputs is idempotent.
- batch_id = sha256(repo_id, pinned_sha, index, sorted file set): the same
  file list always yields the same batch ids (stable across re-plans).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    CREATED = "CREATED"          # pinned, awaiting batch planning
    BATCHED = "BATCHED"          # batches planned
    IN_PROGRESS = "IN_PROGRESS"  # a batch is being worked (later phases)
    BLOCKED = "BLOCKED"          # needs operator (unresolvable ref, policy, …)
    DONE = "DONE"


class BatchStatus(str, Enum):
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    BATCH_PROCESSED = "BATCH_PROCESSED"  # outcome+evidence, maybe not releasable
    RELEASE_READY = "RELEASE_READY"      # all gates incl. global ones passed
    BLOCKED = "BLOCKED"
    DONE = "DONE"  # uploaded + remote-verified + cleaned


def slug_repo_id(repo_id: str) -> str:
    """Filesystem-safe slug: 'org/name' -> 'org--name' (no URL ever lands here)."""
    return repo_id.strip().replace("/", "--")


def job_id_for(repo_id: str, pinned_sha: str) -> str:
    return f"{slug_repo_id(repo_id)}-{pinned_sha[:12]}"


def batch_id_for(repo_id: str, pinned_sha: str, index: int,
                 files: List[str]) -> str:
    canonical = "\0".join([repo_id, pinned_sha, f"{index:06d}",
                           *sorted(files)])
    return "b-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class Batch:
    batch_id: str
    job_id: str
    index: int
    files: List[str] = field(default_factory=list)
    file_sizes: Dict[str, Optional[int]] = field(default_factory=dict)
    total_bytes: int = 0       # known bytes only; unknown sizes excluded
    n_unknown_bytes: int = 0   # files whose size is not known yet
    reservation: Dict[str, Any] = field(default_factory=dict)  # MD-002 fills in
    checkpoint_rel: str = ""   # relative to campaign root (convention, see store)
    status: str = BatchStatus.PLANNED.value
    result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Batch":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class DatasetJob:
    job_id: str
    repo_id: str
    requested_revision: Optional[str] = None
    pinned_sha: Optional[str] = None  # None => BLOCKED, needs operator
    splits: List[str] = field(default_factory=list)
    rights: Dict[str, Any] = field(default_factory=dict)
    status: str = JobStatus.CREATED.value
    batch_ids: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    created_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetJob":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class Campaign:
    campaign_id: str
    name: str
    destination: Dict[str, Any] = field(default_factory=dict)
    publish_approved: bool = False
    workspace: Dict[str, Any] = field(default_factory=dict)
    job_ids: List[str] = field(default_factory=list)
    created_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Campaign":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})
