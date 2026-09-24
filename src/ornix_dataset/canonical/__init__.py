"""Unified Ornix Dataset normalization: canonical 6-field metadata, opaque
``ornix_<32hex>`` filenames, normalized speakers, and HF ``train/validation/test``
layout. Built on top of the existing QC, rights, release and publish contracts."""

from .bridge import (
    export_campaign_batch,
    export_run,
    sample_from_accepted_row,
    scope_of,
)
from .card import render_card, write_card
from .identity import (
    IDENTITY_SCHEMA,
    CanonicalState,
    internal_sample_key,
    open_state,
    speaker_scope_key,
)
from .layout import (
    MANIFEST_NAME,
    METADATA_NAME,
    READY_NAME,
    SPLITS,
    audio_abs_path,
    metadata_path,
    read_all_rows,
    read_split_rows,
    split_stats,
    upsert_rows,
    write_manifest_sha,
    write_ready,
    write_split_rows,
)
from .loader import (
    OrnixSample,
    iter_samples,
    load_ornix_dataset,
    read_wav_int16,
)
from .normalize import SampleInput, normalize_samples
from .publish import (
    dataset_inventory,
    finalize_dataset,
    publish_dataset,
    receipt_path_for,
)
from .schema import (
    AUDIO_PATH_RE,
    CANONICAL_FIELDS,
    ORNIX_ID_RE,
    SPEAKER_RE,
    CanonicalRow,
    SchemaError,
    audio_path_for,
    is_ornix_id,
    normalize_language,
)
from .verify import CanonicalVerifyResult, verify_canonical_dataset

__all__ = [
    "AUDIO_PATH_RE",
    "CANONICAL_FIELDS",
    "IDENTITY_SCHEMA",
    "MANIFEST_NAME",
    "METADATA_NAME",
    "ORNIX_ID_RE",
    "OrnixSample",
    "READY_NAME",
    "SPEAKER_RE",
    "SPLITS",
    "CanonicalRow",
    "CanonicalState",
    "CanonicalVerifyResult",
    "SampleInput",
    "SchemaError",
    "audio_abs_path",
    "audio_path_for",
    "dataset_inventory",
    "export_campaign_batch",
    "export_run",
    "finalize_dataset",
    "internal_sample_key",
    "is_ornix_id",
    "iter_samples",
    "load_ornix_dataset",
    "metadata_path",
    "normalize_language",
    "normalize_samples",
    "open_state",
    "publish_dataset",
    "read_all_rows",
    "read_split_rows",
    "read_wav_int16",
    "receipt_path_for",
    "render_card",
    "sample_from_accepted_row",
    "scope_of",
    "speaker_scope_key",
    "split_stats",
    "upsert_rows",
    "verify_canonical_dataset",
    "write_card",
    "write_manifest_sha",
    "write_ready",
    "write_split_rows",
]
