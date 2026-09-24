"""`.env` loading tests (offline; never touches the network or prints secrets)."""

import json
import os
import pathlib

from ornix_dataset.ops.env import env_status, find_env_file, load_dotenv


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_parse_supports_quotes_export_and_comments(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    _write(env, "\n".join([
        "# comment",
        "",
        "HF_TOKEN=hf_plain",
        " export QUOTED='a b # c' ",
        'DBL="line"',
        "WITH_COMMENT=value # trailing",
        "BADLINE_NO_EQUALS",
        "1INVALID=x",
    ]) + "\n")
    for k in ("HF_TOKEN", "QUOTED", "DBL", "WITH_COMMENT"):
        monkeypatch.delenv(k, raising=False)
    names = load_dotenv(str(env))
    assert set(names) == {"HF_TOKEN", "QUOTED", "DBL", "WITH_COMMENT"}
    assert os.environ["HF_TOKEN"] == "hf_plain"
    assert os.environ["QUOTED"] == "a b # c"      # inner # kept inside quotes
    assert os.environ["DBL"] == "line"
    assert os.environ["WITH_COMMENT"] == "value"  # inline comment stripped
    assert "BADLINE_NO_EQUALS" not in os.environ
    assert "1INVALID" not in os.environ


def test_existing_env_wins_and_override_opts_in(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    _write(env, "HF_TOKEN=from_file\nOTHER=x\n")
    monkeypatch.setenv("HF_TOKEN", "from_shell")
    monkeypatch.delenv("OTHER", raising=False)
    load_dotenv(str(env))
    assert os.environ["HF_TOKEN"] == "from_shell"   # never clobbered by default
    assert os.environ["OTHER"] == "x"
    load_dotenv(str(env), override=True)
    assert os.environ["HF_TOKEN"] == "from_file"


def test_missing_file_is_noop(tmp_path):
    assert load_dotenv(str(tmp_path / "nope.env")) == []


def test_find_env_file_walks_up_and_respects_override(tmp_path, monkeypatch):
    monkeypatch.delenv("ORNIX_ENV_FILE", raising=False)
    root = tmp_path / "proj"
    deep = root / "a" / "b"
    deep.mkdir(parents=True)
    _write(root / ".env", "HF_TOKEN=x\n")
    assert find_env_file(str(deep)) == str(root / ".env")
    override = tmp_path / "other.env"
    _write(override, "HF_TOKEN=y\n")
    monkeypatch.setenv("ORNIX_ENV_FILE", str(override))
    assert find_env_file(str(deep)) == str(override)


def test_env_status_never_returns_value(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "super-secret")
    assert env_status() == {"HF_TOKEN": True}
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert env_status() == {"HF_TOKEN": False}


def test_cli_loads_dotenv_at_startup(tmp_path, monkeypatch, capsys):
    from ornix_dataset.cli import main

    monkeypatch.chdir(tmp_path)
    _write(tmp_path / ".env", "HF_TOKEN=hf_from_dotenv\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert main(["env", "status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["HF_TOKEN"] is True
    assert payload["env_file"] == str(tmp_path / ".env")
    assert os.environ["HF_TOKEN"] == "hf_from_dotenv"


def test_env_example_is_shipped_and_safe():
    root = pathlib.Path(__file__).resolve().parents[2]
    example = root / ".env.example"
    assert example.exists(), ".env.example must be committed as a template"
    text = example.read_text(encoding="utf-8")
    assert "HF_TOKEN=" in text
    assert "hf_replace_with_your_token" in text  # placeholder, never a real token
