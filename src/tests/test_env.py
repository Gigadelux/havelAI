from src.services.common import env


def test_parse_dotenv_handles_comments_quotes_and_export():
    text = "# comment\n\nA=1\nexport B=\"two words\"\nC='x'\nD=val # trailing\nnot a line\nE=\n"
    assert env.parse_dotenv(text) == {"A": "1", "B": "two words", "C": "x", "D": "val", "E": ""}


def test_environment_wins_over_dotenv(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("HAVEL_T1=from_file\nHAVEL_T2=from_file\n")
    monkeypatch.setenv("HAVEL_ENV_FILE", str(f))
    monkeypatch.setenv("HAVEL_T1", "from_env")
    env.reload_env()
    try:
        assert env.get_env("HAVEL_T1") == "from_env"
        assert env.get_env("HAVEL_T2") == "from_file"
        assert env.get_env("HAVEL_T3", "default") == "default"
    finally:
        env.reload_env()


def test_missing_dotenv_is_fine(tmp_path, monkeypatch):
    monkeypatch.setenv("HAVEL_ENV_FILE", str(tmp_path / "nope.env"))
    env.reload_env()
    try:
        assert env.get_env("HAVEL_SURELY_UNSET") is None
    finally:
        env.reload_env()
