"""Sequential Ornix_<digits> identity tests (no audio needed).

New samples take the next number from a persistent counter
(Ornix_0000001, Ornix_0000002, ...); retries reuse the mapped id without
advancing the counter; legacy ornix_<hex> ids keep validating and keep
their original first-2-hex shard mapping.
"""

import pytest

from ornix_dataset.canonical import open_state
from ornix_dataset.canonical.schema import (
    AUDIO_PATH_RE,
    CanonicalRow,
    SchemaError,
    audio_path_for,
    is_new_ornix_id,
    is_ornix_id,
)


def test_sequential_allocation_in_order(tmp_path):
    with open_state(str(tmp_path)) as st:
        assert st.ornix_id_for("k1", "s1") == "Ornix_0000001"
        assert st.ornix_id_for("k2", "s2") == "Ornix_0000002"
        assert st.data["file_seq"] == 3
        st.commit()
    with open_state(str(tmp_path)) as st:
        assert st.ornix_id_for("k3", "s3") == "Ornix_0000003"


def test_retry_reuses_id_without_advancing_counter(tmp_path):
    with open_state(str(tmp_path)) as st:
        first = st.ornix_id_for("k1", "s1")
        assert st.ornix_id_for("k1", "s1") == first
        assert st.ornix_id_for("k2", "s2") == "Ornix_0000002"


def test_shard_uses_last_two_digits(tmp_path):
    assert audio_path_for("Ornix_0000001") == "audio/01/Ornix_0000001.wav"
    assert audio_path_for("Ornix_0000123") == "audio/23/Ornix_0000123.wav"
    assert audio_path_for("Ornix_1234567") == "audio/67/Ornix_1234567.wav"


def test_legacy_ids_validate_and_keep_old_shard():
    legacy = "ornix_" + "ab" + "c" * 30
    assert is_ornix_id(legacy) and not is_new_ornix_id(legacy)
    assert audio_path_for(legacy) == f"audio/ab/{legacy}.wav"
    assert AUDIO_PATH_RE.match(f"audio/ab/{legacy}.wav")
    row = {"audio": f"audio/ab/{legacy}.wav", "text": "t",
           "file_name": f"audio/ab/{legacy}.wav",
           "speaker": "spk_" + "d" * 32, "duration": 1.0, "language": "vi"}
    assert CanonicalRow.from_dict(row).audio.startswith("audio/ab/")


def test_new_ids_validate_and_reject_garbage():
    assert is_ornix_id("Ornix_0000001") and is_new_ornix_id("Ornix_0000001")
    assert is_new_ornix_id("Ornix_12345678")  # grows past 7 digits
    assert not is_ornix_id("ornix_0000001")
    assert not is_ornix_id("Ornix_123")
    assert not is_ornix_id("Ornix_abcdefg")
    with pytest.raises(SchemaError):
        audio_path_for("nope.wav")
