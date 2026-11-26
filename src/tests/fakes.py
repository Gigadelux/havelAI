"""Test doubles shared by the service tests: a deterministic embedder and an in-memory store."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator, Sequence

from src.repositories.base import VectorHit, VectorRecord, VectorRepository


class HashEmbedder:
    """Bag of words hashed into `dim` buckets, L2-normalized: same words -> similar vectors."""

    name = "test/hash-embedder"

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for t in texts:
            v = [0.0] * self._dim
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self._dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class MemoryVectorRepository(VectorRepository):
    kind = "memory"

    def __init__(self, dimension: int, collection: str = "test") -> None:
        super().__init__(dimension, collection)
        self.rows: dict[str, VectorRecord] = {}
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def count(self) -> int:
        return len(self.rows)

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        self.check_vectors(records)
        for r in records:
            self.rows[r.id] = r
        return len(records)

    def search(self, vector: Sequence[float], k: int) -> list[VectorHit]:
        def cos(a, b):
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)

        scored = sorted(self.rows.values(), key=lambda r: -cos(vector, r.vector))[:k]
        return [VectorHit(r.text, cos(vector, r.vector), r.source, r.attack_type) for r in scored]

    def iter_records(self, batch_size: int = 512) -> Iterator[VectorRecord]:
        yield from self.rows.values()
