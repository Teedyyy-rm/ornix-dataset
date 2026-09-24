"""Multi-dataset campaign: Campaign → Dataset Job → Batch (MD-001).

One dataset is the active job at a time; inside it, batches move through
download → QC → publish stages independently. All state is JSON on disk,
written atomically (see ops + util.io); resume is idempotent and never
mutates a pinned revision.
"""

from .models import Batch, BatchStatus, Campaign, DatasetJob, JobStatus, batch_id_for
from .cleanup import (
    CleanupPolicy,
    cleanup_batch,
    eligibility,
    manifest_archive_path,
    pump_campaign,
)
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
from .publishing import (
    list_releases,
    load_receipt,
    path_prefix_for,
    prepare_release,
    publish_release,
    receipt_path_for,
    record_path_for,
    release_dir_for,
    release_id_for,
    staging_revision_for,
)
from .planning import plan_batches
from .preflight import (
    HARD_MAX_FILE_BYTES,
    DestinationPreflight,
    campaign_preflight,
    check_destination,
    expected_upload_bytes,
    oversized_files,
    repo_current_bytes,
)
from .refs import (
    canonical_repo_id,
    default_resolver,
    is_full_sha,
    normalize_repo_ref,
    pin_revision,
)
from .report import campaign_report, dry_run_plan
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
    "CleanupPolicy",
    "DatasetJob",
    "DestinationPreflight",
    "HARD_MAX_FILE_BYTES",
    "JobStatus",
    "MANIFEST_NAME",
    "ReservationLedger",
    "StageProfile",
    "WatermarkGate",
    "batch_id_for",
    "batch_staging_dir",
    "campaign_preflight",
    "campaign_report",
    "canonical_repo_id",
    "check_destination",
    "cleanup_batch",
    "default_resolver",
    "downloaded_files",
    "dry_run_plan",
    "eligibility",
    "expected_upload_bytes",
    "fetch_job_inventory",
    "gate_batch_release",
    "gate_receipt_path",
    "is_full_sha",
    "list_releases",
    "load_receipt",
    "manifest_archive_path",
    "manifest_path",
    "manifest_to_source_record",
    "normalize_repo_ref",
    "oversized_files",
    "path_prefix_for",
    "pin_revision",
    "plan_batches",
    "plan_with_budget",
    "prepare_release",
    "publish_release",
    "pump_campaign",
    "qc_done_files",
    "qc_paths",
    "qc_run_id",
    "read_manifest",
    "receipt_path_for",
    "record_path_for",
    "release_dir_for",
    "release_file_inventory",
    "release_id_for",
    "repo_current_bytes",
    "run_batch",
    "run_qc_batch",
    "staging_revision_for",
    "try_admit",
]
