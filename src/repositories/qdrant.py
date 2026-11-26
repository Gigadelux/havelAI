"""Hosted vector store: Qdrant (`qdrant-client`, `hosted` extra).

`url` + `api_key` for Qdrant Cloud or a self-hosted server; `location=":memory:"` or `path=` for
an embedded store (tests, offline use). Point ids are UUIDs derived from the text id, so the same
text always maps to the same point.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from typing import Any

from .base import (
    RepositoryError,
    VectorHit,
    VectorRecord,
    VectorRepository,
    hit_from_payload,
    record_from_payload,
)


def point_id(text_id: str) -> str:
    return str(uuid.UUID(hex=text_id))


class QdrantRepository(VectorRepository):
    kind = "qdrant"

    def __init__(
        self,
        dimension: int,
        collection: str = "havel_vectors",
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
        path: str | None = None,
        timeout: int = 30,
    ) -> None:
        super().__init__(dimension, collection)
        if not (url or location or path):
            raise RepositoryError("Qdrant needs `url` (set QDRANT_URL), `location` or `path`")
        self.url, self.location, self.path = url, location, path
        self._api_key = api_key
        self.timeout = timeout
        self._client: Any = None

    def __repr__(self) -> str:
        where = self.url or self.location or self.path
        return f"QdrantRepository({where!r}, collection={self.collection!r})"

    @property
    def client(self) -> Any:
        if self._client is None:
            self.connect()
        return self._client

    def connect(self) -> None:
        if self._client is not None:
            return
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise RepositoryError(
                "Qdrant needs qdrant-client: pip install havelai[hosted]"
            ) from exc
        try:
            if self.url:
                client = QdrantClient(url=self.url, api_key=self._api_key, timeout=self.timeout)
            elif self.path:
                client = QdrantClient(path=self.path)
            else:
                client = QdrantClient(location=self.location)
            if not client.collection_exists(self.collection):
                client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(
                        size=self.dimension, distance=models.Distance.COSINE
                    ),
                )
            else:
                size = client.get_collection(self.collection).config.params.vectors.size
                if size != self.dimension:
                    raise RepositoryError(
                        f"Qdrant collection {self.collection!r} has dimension {size}, "
                        f"the embedder gives {self.dimension}"
                    )
        except RepositoryError:
            raise
        except Exception as exc:
            raise RepositoryError(f"could not connect to Qdrant ({self!r}): {exc}") from exc
        self._client = client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        from qdrant_client import models

        self.check_vectors(records)
        unique = {r.id: r for r in records}
        points = [
            models.PointStruct(
                id=point_id(rid), vector=[float(x) for x in r.vector], payload=r.payload()
            )
            for rid, r in unique.items()
        ]
        self.client.upsert(self.collection, points=points, wait=True)
        return len(points)

    def search(self, vector: Sequence[float], k: int) -> list[VectorHit]:
        res = self.client.query_points(
            self.collection, query=[float(x) for x in vector], limit=k, with_payload=True
        )
        return [hit_from_payload(p.payload or {}, p.score) for p in res.points]

    def iter_records(self, batch_size: int = 512) -> Iterator[VectorRecord]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in points:
                yield record_from_payload(p.payload or {}, p.vector)
            if offset is None:
                break
