import sqlite3

import pytest

from src.services.common.ingest import IngestError, looks_non_text, read_text_column


def test_csv_reads_text_and_skips_empty(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text('id,text\n1,hello there\n2,\n3,"with, comma"\n4,42\n')
    assert read_text_column(p, "text") == ["hello there", "with, comma", "42"]


def test_tsv(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_text("text\tn\nfirst\t1\n")
    assert read_text_column(p, "text") == ["first"]


def test_csv_numeric_column_is_rejected(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("id,text\n1,a\n2,b\n")
    with pytest.raises(IngestError, match="not text"):
        read_text_column(p, "id")


def test_csv_missing_column(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x\n1\n")
    with pytest.raises(IngestError, match="not in"):
        read_text_column(p, "text")


def _db(tmp_path):
    p = tmp_path / "a.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE prompts (id INTEGER, body TEXT, mixed TEXT, n REAL)")
    con.executemany(
        "INSERT INTO prompts VALUES (?, ?, ?, ?)",
        [(1, "one", "x", 1.0), (2, None, b"\x00\x01", 2.0), (3, "three", "y", 3.0)],
    )
    con.commit()
    con.close()
    return p


def test_sqlite_reads_text(tmp_path):
    assert read_text_column(_db(tmp_path), "body", table="prompts") == ["one", "three"]


@pytest.mark.parametrize(
    "column, table, message",
    [
        ("n", "prompts", "declared"),
        ("id", "prompts", "declared"),
        ("mixed", "prompts", "blob"),
        ("body", "missing", "table"),
        ("nope", "prompts", "not in table"),
        ("body", None, "needs `table`"),
    ],
)
def test_sqlite_errors(tmp_path, column, table, message):
    with pytest.raises(IngestError, match=message):
        read_text_column(_db(tmp_path), column, table=table)


def test_sqlite_table_name_is_not_injectable(tmp_path):
    with pytest.raises(IngestError, match="table"):
        read_text_column(_db(tmp_path), "body", table='prompts"; DROP TABLE prompts; --')


def test_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    p = tmp_path / "a.parquet"
    pq.write_table(pa.table({"text": ["a", None, "b"], "n": [1, 2, 3]}), p)
    assert read_text_column(p, "text") == ["a", "b"]
    with pytest.raises(IngestError, match="not text"):
        read_text_column(p, "n")


def test_unsupported_and_missing_files(tmp_path):
    p = tmp_path / "a.json"
    p.write_text("{}")
    with pytest.raises(IngestError, match="unsupported"):
        read_text_column(p, "text")
    with pytest.raises(IngestError, match="does not exist"):
        read_text_column(tmp_path / "nope.csv", "text")


def test_looks_non_text():
    assert looks_non_text("12") and looks_non_text("-1.5e3") and looks_non_text("TRUE")
    assert not looks_non_text("12 apples")
