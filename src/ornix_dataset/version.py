"""Single source of truth for the package + policy/schema versions."""

__version__ = "0.1.0"

# Schema + policy revision tags. Bump these (never mutate in place) whenever the
# on-disk contract or default policy semantics change (spec §5.3, §6).
# v2: source-admission provenance (rate class, codec/lossy, container),
# target-aware effective bandwidth, canonicalization action + post-render verify.
SCHEMA_VERSION = "ornix-schema-v2"
DEFAULT_POLICY_VERSION = "ornix-qc-v1"
ANALYZER_VERSION = "ornix-analyzer-0.2.0"
