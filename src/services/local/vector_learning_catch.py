"""Vector layer: semantic similarity to known attack texts, which also grows as it is used.

Three phases:

1. **initialize**: load known attack texts from a csv, parquet or SQLite file (one text column,
   see `common/ingest.py`), embed them and store them.
2. **learn**: after a detection, the engine passes each chunk with its regex + model score; every
   chunk at or above `learn_threshold` (default 0.7) is stored with its attack type. Off with
   `VectorConfig(learning=False)`.
3. **detect**: embed the prompt and return the highest cosine similarity plus the `top_k`
   (default 10) most similar stored texts. The layer's score is that similarity (clipped to
   0..1); it fires at `catch_threshold` (default 0.8).

Every stored text is treated as an attack example. The store is a `VectorRepository`: Milvus Lite
(one local `.db` file) here, Qdrant in the hosted service. Nothing here writes data directly.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ...data.taxonomy import AttackType
from ...repositories.base import (
    SOURCE_IMPORT,
    SOURCE_LEARNED,
    VectorHit,
    VectorRecord,
    VectorRepository,
)
from ...repositories.factory import get_vector_repository
from ..common.config import VectorConfig
from ..common.embedder import Embedder, FastEmbedder
from ..common.ingest import IngestError, read_text_column
from ..common.layer import LayerResult

NAME = "vector"
VectorIngestError = IngestError  # the error `initialize` raises for unreadable or non-text data


@dataclass(frozen=True, slots=True)
class VectorDetection:
    max_similarity: float
    top: list[VectorHit] = field(default_factory=list)
    flagged: bool = False
    vectors_in_store: int = 0


class VectorLearningCatcher:
    """`Layer` over any `VectorRepository` and `Embedder`."""

    name = NAME

    def __init__(
        self,
        config: VectorConfig,
        repository: VectorRepository,
        embedder: Embedder,
    ) -> None:
        self.config = config
        self.repository = repository
        self.embedder = embedder
        self._count: int | None = None

    # ----------------------------------------------------------------------- state

    def count(self) -> int:
        if self._count is None:
            self._count = self.repository.count()
        return self._count

    def is_empty(self) -> bool:
        return self.count() == 0

    def _store(self, records: list[VectorRecord]) -> int:
        n = self.repository.upsert(records)
        self._count = None  # upserts may replace, so recount on demand
        return n

    # ----------------------------------------------------------------------- 1. initialization

    def initialize(
        self,
        path: str | Path,
        column: str,
        table: str | None = None,
        attack_type: AttackType | str | None = None,
        batch_size: int = 256,
    ) -> int:
        """Embed and store the texts of `column`. Raises `VectorIngestError` for non-text data."""
        texts = read_text_column(path, column, table)
        if not texts:
            raise VectorIngestError(f"column {column!r} has no text values")
        label = AttackType(attack_type).value if attack_type else None
        stored = 0
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = self.embedder.embed(batch)
            stored += self._store(
                [
                    VectorRecord(
                        text=t,
                        vector=v,
                        source=SOURCE_IMPORT,
                        attack_type=label,
                        embedder=self.embedder.name,
                    )
                    for t, v in zip(batch, vectors, strict=True)
                ]
            )
        return stored

    # ----------------------------------------------------------------------- 2. learning

    def learn(
        self,
        chunks: Sequence[str],
        scores: Sequence[float],
        attack_type: AttackType | None = None,
    ) -> int:
        """Store every chunk whose score reaches `learn_threshold`. Returns how many."""
        if not self.config.learning:
            return 0
        picked = [
            (c, s)
            for c, s in zip(chunks, scores, strict=True)
            if c.strip() and s >= self.config.learn_threshold
        ]
        if not picked:
            return 0
        vectors = self.embedder.embed([c for c, _ in picked])
        label = attack_type.value if attack_type and attack_type is not AttackType.BENIGN else None
        return self._store(
            [
                VectorRecord(
                    text=c,
                    vector=v,
                    source=SOURCE_LEARNED,
                    attack_type=label,
                    score=float(s),
                    embedder=self.embedder.name,
                )
                for (c, s), v in zip(picked, vectors, strict=True)
            ]
        )

    # ----------------------------------------------------------------------- 3. detection

    def detect(self, text: str) -> VectorDetection:
        n = self.count()
        if n == 0 or not text.strip():
            return VectorDetection(0.0, [], False, n)
        vector = self.embedder.embed([text])[0]
        top = self.repository.search(vector, self.config.top_k)
        best = max((h.similarity for h in top), default=0.0)
        return VectorDetection(best, top, best >= self.config.catch_threshold, n)

    def score(self, text: str) -> LayerResult:
        start = time.perf_counter()
        d = self.detect(text)
        score = min(max(d.max_similarity, 0.0), 1.0)
        attack_type = None
        if d.flagged:
            typed = next((h.attack_type for h in d.top if h.attack_type), None)
            attack_type = AttackType(typed) if typed else AttackType.OTHER_INJECTION
        return LayerResult(
            layer=NAME,
            score=score,
            fired=d.flagged,
            attack_type=attack_type,
            seconds=time.perf_counter() - start,
            detail={
                "max_similarity": d.max_similarity,
                "top_texts": [
                    {
                        "text": h.text,
                        "similarity": h.similarity,
                        "source": h.source,
                        "attack_type": h.attack_type,
                    }
                    for h in d.top
                ],
                "vectors_in_store": d.vectors_in_store,
                "db_type": self.repository.kind,
                "embedder": self.embedder.name,
                "embedder_dimensions": self.embedder.dimension,
                "catch_threshold": self.config.catch_threshold,
                "learn_threshold": self.config.learn_threshold,
            },
        )


def local_vector_catcher(
    config: VectorConfig, embedder: Embedder | None = None
) -> VectorLearningCatcher:
    """Milvus Lite file at `config.db_path` + fastembed."""
    embedder = embedder or FastEmbedder(config.embedder)
    repo = get_vector_repository(
        "milvus_lite",
        path=config.db_path,
        dimension=embedder.dimension,
        collection=config.collection,
    )
    return VectorLearningCatcher(config, repo, embedder)
