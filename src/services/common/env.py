"""Read settings from the environment, falling back to a `.env` file. Stdlib only.

Values already in `os.environ` win over `.env`. The file is read once, from the current working
directory (or `HAVEL_ENV_FILE`), and never written. Values are never logged.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse `KEY=value` lines. Supports comments, blank lines, `export`, and quoted values."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    path = Path(os.environ.get("HAVEL_ENV_FILE", ".env"))
    try:
        return parse_dotenv(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def get_env(name: str, default: str | None = None) -> str | None:
    """`os.environ[name]`, else the `.env` value, else `default`. Empty strings count as unset."""
    value = os.environ.get(name) or _dotenv().get(name)
    return value if value else default


def reload_env() -> None:
    """Forget the cached `.env` (for tests, or after editing the file)."""
    _dotenv.cache_clear()
