"""Local analytics store: a SQLite file with the tables from SCHEMA.sql. Stdlib only."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .base import RecordRepository, RepositoryError, VectorRecord
from .schema import SCHEMA_SQL

DETECTION_COLUMNS = (
    "id", "created_at", "mode", "score", "is_injection", "attack_type", "detected_by",
    "detect_threshold", "catch_threshold", "learn_threshold", "detection_ms", "prompt_tokens",
    "chunk_size", "chunk_count", "detecting_chunk_tokens", "detecting_chunk_index",
    "vectors_in_store", "max_similarity", "learned_chunks", "db_type", "embedder_model",
    "embedder_dimensions", "library_version", "local_model_version", "hosted_model",
    "hosted_endpoint",
)  # fmt: skip
LAYER_COLUMNS = (
    "layer",
    "enabled",
    "score",
    "weight",
    "contribution",
    "fired",
    "attack_type",
    "seconds",
)
HIT_COLUMNS = ("rank", "text", "similarity", "source", "attack_type")


class SQLiteRecordRepository(RecordRepository):
    kind = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._con: sqlite3.Connection | None = None

    @property
    def con(self) -> sqlite3.Connection:
        if self._con is None:
            self.connect()
        assert self._con is not None
        return self._con

    def connect(self) -> None:
        if self._con is not None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(self.path)
            con.execute("PRAGMA foreign_keys = ON")
            con.executescript(SCHEMA_SQL)
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not open {self.path}: {exc}") from exc
        con.row_factory = sqlite3.Row
        self._con = con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def count(self) -> int:
        return int(self.con.execute("SELECT COUNT(*) FROM detections").fetchone()[0])

    def write_report(self, report: Mapping[str, Any]) -> str:
        row = {k: report.get(k) for k in DETECTION_COLUMNS}
        row["is_injection"] = int(bool(report["is_injection"]))
        row["detected_by"] = ",".join(report.get("detected_by") or ())
        marks = ", ".join("?" for _ in DETECTION_COLUMNS)
        try:
            with self.con:
                self.con.execute(
                    f"INSERT OR REPLACE INTO detections ({', '.join(DETECTION_COLUMNS)}) VALUES ({marks})",
                    [row[k] for k in DETECTION_COLUMNS],
                )
                self.con.executemany(
                    f"INSERT OR REPLACE INTO layer_scores (detection_id, {', '.join(LAYER_COLUMNS)}) "
                    f"VALUES (?, {', '.join('?' for _ in LAYER_COLUMNS)})",
                    [
                        [
                            report["id"],
                            *[
                                int(v) if isinstance(v, bool) else v
                                for v in (layer[c] for c in LAYER_COLUMNS)
                            ],
                        ]
                        for layer in (report.get("layers") or {}).values()
                    ],
                )
                self.con.executemany(
                    f"INSERT OR REPLACE INTO vector_hits (detection_id, {', '.join(HIT_COLUMNS)}) "
                    f"VALUES (?, {', '.join('?' for _ in HIT_COLUMNS)})",
                    [
                        [report["id"], *[hit[c] for c in HIT_COLUMNS]]
                        for hit in report.get("vector_top") or ()
                    ],
                )
        except sqlite3.Error as exc:
            raise RepositoryError(f"could not write to {self.path}: {exc}") from exc
        return str(report["id"])

    def write_vectors(self, records: Sequence[VectorRecord], dimensions: int | None = None) -> int:
        """Mirror vector-store records (without the vectors) into `learned_vectors`."""
        with self.con:
            self.con.executemany(
                "INSERT OR REPLACE INTO learned_vectors "
                "(id, text, source, attack_type, score, embedder, dimensions, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (r.id, r.text, r.source, r.attack_type, r.score, r.embedder,
                     dimensions or len(r.vector), r.created_at)
                    for r in records
                ],
            )  # fmt: skip
        return len(records)

    def reports(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM detections ORDER BY created_at DESC"
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        out = []
        for row in self.con.execute(sql, params).fetchall():
            d = dict(row)
            d["is_injection"] = bool(d["is_injection"])
            d["detected_by"] = [x for x in (d["detected_by"] or "").split(",") if x]
            layers = self.con.execute(
                f"SELECT {', '.join(LAYER_COLUMNS)} FROM layer_scores WHERE detection_id = ?",
                [d["id"]],
            ).fetchall()
            d["layers"] = {
                r["layer"]: {**dict(r), "enabled": bool(r["enabled"]), "fired": bool(r["fired"])}
                for r in layers
            }
            hits = self.con.execute(
                f"SELECT {', '.join(HIT_COLUMNS)} FROM vector_hits WHERE detection_id = ? ORDER BY rank",
                [d["id"]],
            ).fetchall()
            d["vector_top"] = [dict(h) for h in hits]
            out.append(d)
        return out
