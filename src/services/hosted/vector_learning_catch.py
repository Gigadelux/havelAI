"""Vector layer for the hosted service: same three phases as the local one, stored in Qdrant.

Qdrant URL and key come from `VectorConfig.provider_url` / `provider_api_key`, else from `.env`
(`QDRANT_URL`, `QDRANT_API_KEY`). Embeddings are computed on this machine with fastembed, so only
vectors and the stored texts travel to Qdrant.
"""

from __future__ import annotations

from ...repositories.factory import get_vector_repository
from ..common.config import VectorConfig
from ..common.embedder import Embedder, FastEmbedder
from ..local.vector_learning_catch import (
    NAME,
    VectorDetection,
    VectorIngestError,
    VectorLearningCatcher,
)

__all__ = [
    "NAME",
    "VectorDetection",
    "VectorIngestError",
    "VectorLearningCatcher",
    "hosted_vector_catcher",
]


def hosted_vector_catcher(
    config: VectorConfig, embedder: Embedder | None = None, location: str | None = None
) -> VectorLearningCatcher:
    """Qdrant at `config.provider_url` (or `location`, e.g. ":memory:") + fastembed."""
    embedder = embedder or FastEmbedder(config.embedder)
    repo = get_vector_repository(
        "qdrant",
        dimension=embedder.dimension,
        collection=config.collection,
        url=None if location else config.provider_url,
        api_key=config.provider_api_key,
        location=location,
    )
    return VectorLearningCatcher(config, repo, embedder)
