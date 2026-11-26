"""Local vector store: Milvus Lite, a single `.db` file on disk (`pymilvus`, `local` extra)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .base import (
    RepositoryError,
    VectorHit,
    VectorRecord,
    VectorRepository,
    hit_from_payload,
    record_from_payload,
)

PAYLOAD_FIELDS = ["text", "source", "attack_type", "score", "embedder", "created_at"]


class MilvusLiteRepository(VectorRepository):
    kind = "milvus_lite"

    def __init__(self, path: str | Path, dimension: int, collection: str = "havel_vectors") -> None:
        super().__init__(dimension, collection)
        self.path = Path(path)
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            self.connect()
        return self._client

    def connect(self) -> None:
        if self._client is not None:
            return
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise RepositoryError("Milvus Lite needs pymilvus: pip install havelai[local]") from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client = MilvusClient(str(self.path))
            if not self._client.has_collection(self.collection):
                self._client.create_collection(
                    collection_name=self.collection,
                    dimension=self.dimension,
                    primary_field_name="id",
                    id_type="string",
                    max_length=64,
                    vector_field_name="vector",
                    metric_type="COSINE",
                    auto_id=False,
                    enable_dynamic_field=True,
                )
            else:
                self._check_dimension()
        except RepositoryError:
            raise
        except Exception as exc:
            raise RepositoryError(f"could not open Milvus Lite at {self.path}: {exc}") from exc

    def _check_dimension(self) -> None:
        desc = self._client.describe_collection(self.collection)
        for f in desc.get("fields", []):
            dim = (f.get("params") or {}).get("dim")
            if f.get("name") == "vector" and dim is not None and int(dim) != self.dimension:
                raise RepositoryError(
                    f"{self.path} collection {self.collection!r} has dimension {dim}, "
                    f"the embedder gives {self.dimension}; use another collection or db file"
                )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def count(self) -> int:
        res = self.client.query(self.collection, filter="", output_fields=["count(*)"])
        return int(res[0]["count(*)"]) if res else 0

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        self.check_vectors(records)
        unique = {r.id: r for r in records}  # last one wins within a batch
        data = [
            {"id": rid, "vector": [float(x) for x in r.vector], **r.payload()}
            for rid, r in unique.items()
        ]
        self.client.upsert(collection_name=self.collection, data=data)
        return len(data)

    def search(self, vector: Sequence[float], k: int) -> list[VectorHit]:
        res = self.client.search(
            collection_name=self.collection,
            data=[[float(x) for x in vector]],
            limit=k,
            output_fields=PAYLOAD_FIELDS,
            search_params={"metric_type": "COSINE"},
        )
        hits = res[0] if res else []
        return [hit_from_payload(h.get("entity", {}), h["distance"]) for h in hits]

    def iter_records(self, batch_size: int = 512) -> Iterator[VectorRecord]:
        it = self.client.query_iterator(
            collection_name=self.collection,
            batch_size=batch_size,
            filter="",
            output_fields=[*PAYLOAD_FIELDS, "vector"],
        )
        try:
            while batch := it.next():
                for row in batch:
                    yield record_from_payload(row, row["vector"])
        finally:
            it.close()
