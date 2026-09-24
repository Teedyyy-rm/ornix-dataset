"""Multi-dataset campaign: Campaign → Dataset Job → Batch (MD-001).

One dataset is the active job at a time; inside it, batches move through
download → QC → publish stages independently. All state is JSON on disk,
written atomically (see ops + util.io); resume is idempotent and never
mutates a pinned revision.
"""

from .models import Batch, BatchStatus, Campaign, DatasetJob, JobStatus, batch_id_for
from .downloading import (
    MANIFEST_NAME,
    batch_staging_dir,
    downloaded_files,
    fetch_job_inventory,
    manifest_path,
    read_manifest,
    run_batch,
)
from .processing import (
    gate_batch_release,
    gate_receipt_path,
    manifest_to_source_record,
    qc_done_files,
    qc_paths,
    qc_run_id,
    run_qc_batch,
)
from .planning import plan_batches
from .refs import (
    canonical_repo_id,
    default_resolver,
    is_full_sha,
    normalize_repo_ref,
    pin_revision,
)
from .resources import (
    STAGES,
    BatchPlan,
    Budget,
    ReservationLedger,
    StageProfile,
    WatermarkGate,
    plan_with_budget,
    try_admit,
)
from .store import CampaignStore

__all__ = [
    "STAGES",
    "Batch",
    "BatchPlan",
    "BatchStatus",
    "Budget",
    "Campaign",
    "CampaignStore",
    "DatasetJob",
    "JobStatus",
    "MANIFEST_NAME",
    "ReservationLedger",
    "StageProfile",
    "WatermarkGate",
    "batch_id_for",
    "batch_staging_dir",
    "canonical_repo_id",
    "default_resolver",
    "downloaded_files",
    "fetch_job_inventory",
    "gate_batch_release",
    "gate_receipt_path",
    "is_full_sha",
    "manifest_path",
    "manifest_to_source_record",
    "normalize_repo_ref",
    "pin_revision",
    "plan_batches",
    "plan_with_budget",
    "qc_done_files",
    "qc_paths",
    "qc_run_id",
    "read_manifest",
    "run_batch",
    "run_qc_batch",
    "try_admit",
]
