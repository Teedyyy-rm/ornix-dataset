"""extract_loose: whole-dataset single pass with concurrent downloads (no network).

Fakes the Hub API + downloader: proves every listed file is fetched exactly
once, metadata stays in sorted-repo order despite concurrent completion, and
the result reports a full pass (exhausted, no cursor).
"""

import importlib.util
import os

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..",
                       "scripts", "extract_hf_audio.py")


def _load():
    spec = importlib.util.spec_from_file_location("extract_hf_audio", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Sib:
    def __init__(self, rfilename):
        self.rfilename = rfilename


class _Info:
    def __init__(self, siblings):
        self.siblings = siblings
        self.sha = "abc123def456789"


class _Api:
    def __init__(self, siblings):
        self._siblings = siblings

    def dataset_info(self, repo, revision="main"):
        return _Info(self._siblings)


def test_loose_full_pass_parallel_ordered(tmp_path, monkeypatch):
    import huggingface_hub

    mod = _load()
    rels = [f"raw_audio/b_{i:02d}.wav" for i in range(10)] + ["notes.txt"]
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    calls = []

    def fake_download(repo, filename, repo_type=None, **kw):
        calls.append(filename)
        p = blob_dir / filename.replace("/", "__")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF" + filename.encode())
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    out = tmp_path / "out"
    args = _NS(repo="org/ds", out=str(out), loose_audio_prefix="raw_audio/",
               loose_meta=None, language="vi", speaker_ref=None, loose_workers=4)
    res = mod.extract_loose(args, _Api([_Sib(r) for r in rels]), None)

    assert res["written"] == 10
    assert res["exhausted"] is True
    assert "next_shard" not in res  # no paging cursor anymore
    assert sorted(calls) == sorted(r for r in rels if r.endswith(".wav"))
    assert len(set(calls)) == len(calls)  # each file fetched exactly once
    import json
    rows = [json.loads(l) for l in open(out / "metadata.jsonl", encoding="utf-8")]
    assert [r["file"] for r in rows] == sorted(
        r.replace("/", "__") for r in rels if r.endswith(".wav"))
    for r in rows:
        assert os.path.exists(os.path.join(str(out), r["file"]))
