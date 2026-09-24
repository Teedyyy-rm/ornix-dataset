"""Regression: LocalSourceAdapter propagates the declared source license.

The rights declaration (per-source, in sources.yaml) is the source of truth for
the license; before the fix the adapter only read a per-file metadata key and the
canonical card / RELEASE_READY recorded ``license: UNKNOWN`` for a fully declared
CC-BY-NC-SA source.
"""

import os

from fixtures import synth
from ornix_dataset.contracts.enums import RightsStatus
from ornix_dataset.ingestion.local import LocalSourceAdapter
from ornix_dataset.ingestion.permissions import PermissionGate


def _adapter(tmp_path):
    root = str(tmp_path)
    synth.write_wav(os.path.join(root, "utt.wav"),
                    synth.speechlike(dur=3.5), 24000)
    prefix = "file://" + os.path.abspath(root)
    gate = PermissionGate({"src": {
        "uri_prefix": prefix,
        "rights_status": "REDISTRIBUTION_APPROVED",
        "redistribution_permitted": True,
        "source_license": "CC-BY-NC-SA-4.0"}})
    return LocalSourceAdapter(root=root, gate=gate, probe=False)


def test_local_adapter_carries_declared_license(tmp_path):
    recs = list(_adapter(tmp_path).scan())
    assert len(recs) == 1
    rec = recs[0]
    assert rec.source_license == "CC-BY-NC-SA-4.0"
    assert rec.rights_status == RightsStatus.REDISTRIBUTION_APPROVED
    assert rec.redistribution_permitted is True
