"""Text embeddings for the vector layer. Default: fastembed (ONNX, CPU, no torch)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .config import DEFAULT_EMBEDDER

logger = logging.getLogger("havelai")


@runtime_checkable
class Embedder(Protocol):
    name: str

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedder:
    """`fastembed.TextEmbedding`, loaded on first use (downloads the model once, logged)."""

    def __init__(self, name: str = DEFAULT_EMBEDDER, cache_dir: str | None = None) -> None:
        self.name = name
        self.cache_dir = cache_dir
        self._model: Any = None
        self._dimension: int | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise ImportError(
                    "the vector layer needs fastembed: pip install havelai[local]"
                ) from exc
            logger.info("havelai: loading embedder %s (first use downloads it)", self.name)
            self._model = TextEmbedding(model_name=self.name, cache_dir=self.cache_dir)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            try:
                from fastembed import TextEmbedding

                for m in TextEmbedding.list_supported_models():
                    if m.get("model") == self.name and m.get("dim"):
                        self._dimension = int(m["dim"])
                        break
            except ImportError:  # pragma: no cover
                pass
            if self._dimension is None:
                self._dimension = len(self.embed(["dimension probe"])[0])
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._load().embed(list(texts))]
