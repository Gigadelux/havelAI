"""Abstract data-write layer. Every write the library makes goes through a `Repository`.

    Repository (ABC)            connect / close / count / write, context manager
    ├── VectorRepository        + upsert / search / is_empty / iter_records   (vector layer)
    │   ├── milvus_lite.MilvusLiteRepository     local single-file db
    │   └── qdrant.QdrantRepository              hosted (or in-memory for tests)
    └── RecordRepository        + write_report / reports                      (analytics)
        ├── sqlite.SQLiteRecordRepository        local file, tables from SCHEMA.sql
        └── remote_http.RemoteRecordRepository   hosted: POST to an ingest URL

Layers and engines only see these abstract types; `factory.py` picks the connector. Connectors
import their client library lazily, so importing this module stays stdlib-only.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Self

SOURCE_IMPORT = "import"  # uploaded by the user (csv / parquet / db)
SOURCE_LEARNED = "learned"  # stored by the learning step after a confident detection


class RepositoryError(RuntimeError):
    """A connector failed to connect, read or write."""


def text_id(text: str) -> str:
    """Stable id for a text: identical texts map to one record (32 hex chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class VectorRecord:
    text: str
    vector: Sequence[float]
    source: str = SOURCE_IMPORT
    attack_type: str | None = None
    score: float | None = None
    embedder: str | None = None
    created_at: str = field(default_factory=utc_now)

    @property
    def id(self) -> str:
        return text_id(self.text)

    def payload(self) -> dict[str, Any]:
        """Everything except the vector, for the store's metadata fields."""
        return {
            "text": self.text,
            "source": self.source,
            "attack_type": self.attack_type or "",
            "score": -1.0 if self.score is None else float(self.score),
            "embedder": self.embedder or "",
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class VectorHit:
    text: str
    similarity: float
    source: str = SOURCE_IMPORT
    attack_type: str | None = None


def record_from_payload(payload: Mapping[str, Any], vector: Sequence[float]) -> VectorRecord:
    score = payload.get("score")
    return VectorRecord(
        text=str(payload.get("text", "")),
        vector=list(vector),
        source=str(payload.get("source") or SOURCE_IMPORT),
        attack_type=payload.get("attack_type") or None,
        score=None if score is None or score < 0 else float(score),
        embedder=payload.get("embedder") or None,
        created_at=str(payload.get("created_at") or utc_now()),
    )


def hit_from_payload(payload: Mapping[str, Any], similarity: float) -> VectorHit:
    return VectorHit(
        text=str(payload.get("text", "")),
        similarity=float(similarity),
        source=str(payload.get("source") or SOURCE_IMPORT),
        attack_type=payload.get("attack_type") or None,
    )


class Repository(ABC):
    """A place the library writes data to."""

    kind: ClassVar[str] = "abstract"  # stored in analytics as the db type

    @abstractmethod
    def connect(self) -> None:
        """Open the connection and create collections or tables if needed. Idempotent."""

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def write(self, records: Sequence[Any]) -> int:
        """Write records; returns how many were written."""

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VectorRepository(Repository):
    """Stores texts with their embedding; cosine similarity search."""

    def __init__(self, dimension: int, collection: str) -> None:
        if dimension < 1:
            raise RepositoryError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension
        self.collection = collection

    def write(self, records: Sequence[VectorRecord]) -> int:
        return self.upsert(records)

    def check_vectors(self, records: Sequence[VectorRecord]) -> None:
        for r in records:
            if len(r.vector) != self.dimension:
                raise RepositoryError(
                    f"vector has {len(r.vector)} dimensions, the collection has {self.dimension}"
                )

    @abstractmethod
    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or replace by text id."""

    @abstractmethod
    def search(self, vector: Sequence[float], k: int) -> list[VectorHit]:
        """The `k` most similar texts, most similar first; similarity is cosine, -1..1."""

    @abstractmethod
    def iter_records(self, batch_size: int = 512) -> Iterator[VectorRecord]:
        """Every stored record with its vector, for copying to another store."""

    def is_empty(self) -> bool:
        return self.count() == 0


class RecordRepository(Repository):
    """Stores detection reports (see SCHEMA.sql). Implemented by the analytics connectors."""

    def write(self, records: Sequence[Mapping[str, Any]]) -> int:
        for r in records:
            self.write_report(r)
        return len(records)

    @abstractmethod
    def write_report(self, report: Mapping[str, Any]) -> str:
        """Store one report (as from `DetectionReport.to_dict()`); returns its id."""

    @abstractmethod
    def reports(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Stored reports, newest first."""
