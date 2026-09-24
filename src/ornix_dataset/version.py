"""Single source of truth for the package + policy/schema versions."""

__version__ = "0.1.0"

# Schema + policy revision tags. Bump these (never mutate in place) whenever the
# on-disk contract or default policy semantics change (spec §5.3, §6).
SCHEMA_VERSION = "ornix-schema-v1"
DEFAULT_POLICY_VERSION = "ornix-qc-v1"
ANALYZER_VERSION = "ornix-analyzer-0.1.0"
