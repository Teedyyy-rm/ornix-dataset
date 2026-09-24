"""Ornix Dataset — end-to-end speech dataset quality & curation toolkit.

Design principles (see docs/E2E-ORNIX-DATASET.md):

* **Fail-closed.** Missing metric / model / right / transcript never auto-PASS.
* **No vendor lock-in.** Detectors, quality models and exporters are pluggable
  via registries; a missing back-end degrades to ``UNAVAILABLE`` and blocks the
  required gate instead of substituting a different model silently.
* **Provenance everywhere.** Every artifact references source SHA-256, recipe,
  model revision and policy version.
"""

from .version import __version__

__all__ = ["__version__"]
