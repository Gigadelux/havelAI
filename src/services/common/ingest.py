"""Read one text column from a csv, parquet or SQLite file, for the vector layer's initialization.

The column must hold text. Anything else raises `IngestError` instead of being silently turned
into strings:

- **csv/tsv**: every cell is a string on disk, so the type is inferred. A column whose non-empty
  values are all numbers or booleans is not text.
- **parquet**: the column's Arrow type must be a string type (`string`, `large_string`,
  `string_view`). Read with `pyarrow` (in the `local` / `hosted` extras).
- **SQLite** (`.db`, `.sqlite`, `.sqlite3`): needs `table`. The declared column type must have
  text affinity (contains CHAR, CLOB or TEXT) and every stored value must be TEXT.

Empty values and NULLs are skipped. Duplicates are kept; the vector store deduplicates by text.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

CSV_SUFFIXES = {".csv", ".tsv"}
PARQUET_SUFFIXES = {".parquet", ".pq"}
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_BOOLEAN = {"true", "false"}


class IngestError(ValueError):
    """The file, table or column cannot be read as text."""


def read_text_column(path: str | Path, column: str, table: str | None = None) -> list[str]:
    """Return the non-empty text values of `column`, in file order."""
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"{p} does not exist")
    suffix = p.suffix.lower()
    if suffix in CSV_SUFFIXES:
        return _read_csv(p, column, delimiter="\t" if suffix == ".tsv" else ",")
    if suffix in PARQUET_SUFFIXES:
        return _read_parquet(p, column)
    if suffix in SQLITE_SUFFIXES:
        if not table:
            raise IngestError("a SQLite file needs `table`")
        return _read_sqlite(p, table, column)
    supported = sorted(CSV_SUFFIXES | PARQUET_SUFFIXES | SQLITE_SUFFIXES)
    raise IngestError(f"unsupported file type {suffix!r}; use one of {supported}")


def looks_non_text(value: str) -> bool:
    v = value.strip()
    return bool(_NUMBER.match(v)) or v.lower() in _BOOLEAN


def _read_csv(path: Path, column: str, delimiter: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        try:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise IngestError(f"column {column!r} not in {path.name} ({reader.fieldnames})")
            values = [row[column] for row in reader if row.get(column) not in (None, "")]
        except (csv.Error, UnicodeDecodeError) as exc:
            raise IngestError(f"could not parse {path.name}: {exc}") from exc
    if values and all(looks_non_text(v) for v in values):
        raise IngestError(f"column {column!r} holds numbers or booleans, not text")
    return [v for v in values if v.strip()]


def _read_parquet(path: Path, column: str) -> list[str]:
    try:
        import pyarrow.parquet as pq
        import pyarrow.types as pat
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise IngestError("reading parquet needs pyarrow: pip install havelai[local]") from exc
    schema = pq.read_schema(path)
    if column not in schema.names:
        raise IngestError(f"column {column!r} not in {path.name} ({schema.names})")
    arrow_type = schema.field(column).type
    is_text = pat.is_string(arrow_type) or pat.is_large_string(arrow_type)
    is_text = is_text or getattr(pat, "is_string_view", lambda _t: False)(arrow_type)
    if not is_text:
        raise IngestError(f"column {column!r} has type {arrow_type}, not text")
    values = pq.read_table(path, columns=[column]).column(column).to_pylist()
    return [v for v in values if isinstance(v, str) and v.strip()]


def _text_affinity(declared: str) -> bool:
    # SQLite affinity rules: a declared type containing CHAR, CLOB or TEXT has TEXT affinity
    d = declared.upper()
    return any(word in d for word in ("CHAR", "CLOB", "TEXT"))


def _read_sqlite(path: Path, table: str, column: str) -> list[str]:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise IngestError(f"could not open {path.name}: {exc}") from exc
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in tables:
            raise IngestError(f"table {table!r} not in {path.name} ({sorted(tables)})")
        quoted_table = '"' + table.replace('"', '""') + '"'
        info = {r[1]: r[2] for r in con.execute(f"PRAGMA table_info({quoted_table})")}
        if column not in info:
            raise IngestError(f"column {column!r} not in table {table!r} ({sorted(info)})")
        if not _text_affinity(info[column] or ""):
            raise IngestError(f"column {column!r} is declared {info[column]!r}, not text")
        quoted_col = '"' + column.replace('"', '""') + '"'
        rows = con.execute(
            f"SELECT {quoted_col}, typeof({quoted_col}) FROM {quoted_table}"  # noqa: S608 - quoted identifiers
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise IngestError(f"could not read {path.name}: {exc}") from exc
    finally:
        con.close()
    bad = sorted({kind for _, kind in rows if kind not in ("text", "null")})
    if bad:
        raise IngestError(f"column {column!r} holds {bad} values, not only text")
    return [v for v, kind in rows if kind == "text" and v.strip()]
