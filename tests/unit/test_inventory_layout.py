"""Inventory listing across huggingface_hub layouts (regression).

huggingface_hub 2.x dropped ``entry.type`` and renamed ``lfs.oid`` to
``lfs.sha256``. The old code defaulted a missing type to "file" (folders leaked
into the inventory and the downloader then stat'ed a directory) and read only
``lfs.oid`` (so every LFS file lost its content hash and downloads degraded
to size-only verification).
"""

from types import SimpleNamespace

from ornix_dataset.campaign import CampaignStore, fetch_job_inventory
from ornix_dataset.campaign.downloading import _is_repo_folder, _lfs_sha

SHA = "a" * 40
HEX = "5" * 64


class _RepoFolder:
    """hub 2.x folder: no ``type``, no ``size``, no ``lfs``."""

    def __init__(self, path):
        self.path = path


class _RepoFile2x:
    """hub 2.x file: no ``type``, carries ``lfs.sha256``."""

    def __init__(self, path, size, sha=None):
        self.path = path
        self.size = size
        self.lfs = SimpleNamespace(sha256=sha, size=size) if sha else None


class _RepoFile1x:
    """hub 1.x file: ``type='file'``, ``lfs.oid``."""

    def __init__(self, path, size, sha=None):
        self.type = "file"
        self.path = path
        self.size = size
        self.lfs = SimpleNamespace(oid=sha) if sha else None


def _api(entries):
    return SimpleNamespace(
        list_repo_tree=lambda *a, **k: iter(entries))


def _job(tmp_path):
    store = CampaignStore(str(tmp_path / "c"))
    _, out = store.create_campaign("inv", [{"repo": f"org/I@{SHA}"}],
                                   resolver=lambda r, v: SHA)
    return store.load_job(out[0]["job_id"])


def test_is_repo_folder_both_layouts():
    assert _is_repo_folder(_RepoFolder("data")) is True
    assert _is_repo_folder(_RepoFile2x("a/b.wav", 10)) is False
    assert _is_repo_folder(_RepoFile1x("a/b.wav", 10)) is False
    assert _is_repo_folder(SimpleNamespace(type="directory", path="data")) is True
    assert _is_repo_folder(SimpleNamespace(type="file", path="a")) is False


def test_lfs_sha_both_layouts():
    assert _lfs_sha(_RepoFile1x("a", 1, HEX)) == HEX
    assert _lfs_sha(_RepoFile2x("a", 1, HEX)) == HEX
    assert _lfs_sha(_RepoFile2x("a", 1, None)) is None
    # dict-shaped lfs (older dict API)
    assert _lfs_sha(SimpleNamespace(lfs={"oid": HEX})) == HEX
    assert _lfs_sha(SimpleNamespace(lfs={"sha256": HEX})) == HEX
    # junk never becomes a hash
    assert _lfs_sha(SimpleNamespace(lfs={"oid": "nothex"})) is None
    assert _lfs_sha(SimpleNamespace(lfs={"sha256": "z" * 64})) is None


def test_inventory_excludes_folders_and_keeps_hashes(tmp_path):
    job = _job(tmp_path)
    entries = [
        _RepoFolder("data"),
        _RepoFile2x("README.md", 12),
        _RepoFile2x("data/a.parquet", 100, HEX),
        _RepoFile2x("data/b.wav", 7),
    ]
    inv = dict((p, (s, h)) for p, s, h in fetch_job_inventory(_api(entries), job))
    assert "data" not in inv, "directory leaked into inventory"
    assert set(inv) == {"README.md", "data/a.parquet", "data/b.wav"}
    assert inv["data/a.parquet"][1] == HEX
    assert inv["data/b.wav"][1] is None


def test_inventory_1x_layout_still_works(tmp_path):
    job = _job(tmp_path)
    entries = [_RepoFile1x("data/a.parquet", 100, HEX)]
    assert fetch_job_inventory(_api(entries), job) == [("data/a.parquet", 100, HEX)]


def test_inventory_requires_pin(tmp_path):
    job = _job(tmp_path)
    job.pinned_sha = None
    try:
        fetch_job_inventory(_api([]), job)
    except ValueError:
        return
    raise AssertionError("unpinned job must fail closed")
