"""The detector engine shared by `havelai.local` and `havelai.hosted`.

    prompt ─┬─> regex layer  ──> 0 or 1          ─┐
            ├─> model layer  ──> max chunk p_inj  ─┼─> weighted average ─> DetectionReport
            └─> vector layer ──> max similarity   ─┘          │
                                                              └─> learning: chunks whose
                                                                  regex+model score >= learn
                                                                  threshold go to the vector store

Subclasses only decide how the model and vector layers are built (`_build_model`,
`_build_vector`). Layers are built on first use, so creating a `Havel` downloads nothing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

from ... import __version__
from ...analytics.report import LAYERS, DetectionReport, LayerScore, VectorMatch
from ...data.taxonomy import AttackType
from ...repositories.base import VectorRepository
from ...repositories.factory import get_record_repository, get_vector_repository
from . import config as cfg
from .layer import Layer, LayerResult

logger = logging.getLogger("havelai")
LAYER_KEYS = {"regex": cfg.REGEX, "model": cfg.MODEL, "vector": cfg.VECTOR}
HOSTED = "hosted"


class HavelEngine:
    """Base class of `Havel`. Use `havelai.local.Havel` or `havelai.hosted.Havel`."""

    mode: ClassVar[str] = "abstract"

    def __init__(
        self,
        config: cfg.Config | None = None,
        vector_config: cfg.VectorConfig | None = None,
        model_config: cfg.ModelConfig | None = None,
        *,
        regex_layer: Layer | None = None,
        model_layer: Layer | None = None,
        vector_layer: Any = None,
    ) -> None:
        self.config = config or cfg.Config()
        self.vector_config = vector_config or cfg.VectorConfig()
        self.model_config = model_config or cfg.ModelConfig()
        self._regex = regex_layer
        self._model = model_layer
        self._vector = vector_layer
        self.history: list[DetectionReport] = []

    # ----------------------------------------------------------------------- fluent config

    def Config(self, **kwargs: Any) -> Self:  # noqa: N802 - name chosen for the public API
        """Which catchers run and their weights. See `havelai.services.common.config.Config`."""
        self.config = cfg.Config(**kwargs)
        return self

    def VectorConfig(self, **kwargs: Any) -> Self:  # noqa: N802
        self.vector_config = cfg.VectorConfig(**kwargs)
        self._vector = None
        return self

    def ModelConfig(self, **kwargs: Any) -> Self:  # noqa: N802
        self.model_config = cfg.ModelConfig(**kwargs)
        self._model = None
        return self

    configure = Config
    configure_vector = VectorConfig
    configure_model = ModelConfig

    # ----------------------------------------------------------------------- layers

    def _build_model(self) -> Layer:
        raise NotImplementedError

    def _build_vector(self) -> Any:
        raise NotImplementedError

    @property
    def regex_layer(self) -> Layer:
        if self._regex is None:
            from ..local.regex_catch import RegexCatcher

            self._regex = RegexCatcher()
        return self._regex

    @property
    def model_layer(self) -> Layer:
        if self._model is None:
            self._model = self._build_model()
        return self._model

    @property
    def vector_layer(self) -> Any:
        """The `VectorLearningCatcher` (initialize / learn / detect)."""
        if self._vector is None:
            self._vector = self._build_vector()
        return self._vector

    def initialize_vectors(
        self, path: str | Path, column: str, table: str | None = None, **kw: Any
    ) -> int:
        """Vector phase 1: load known attack texts from a csv / parquet / SQLite file."""
        return self.vector_layer.initialize(path, column, table, **kw)

    # ----------------------------------------------------------------------- detection

    def _learning_weights(self) -> tuple[float, float]:
        """Relative weights of regex and model for the learning score, from the config."""
        enabled = self.config.enabled()
        if self.config.detector_scoring is not None:
            w = self.config.detector_scoring
            r, m = w[cfg.REGEX], w[cfg.MODEL]
        else:
            r = cfg.WEIGHTS_WITHOUT_VECTORS[cfg.REGEX] if enabled[cfg.REGEX] else 0.0
            m = cfg.WEIGHTS_WITHOUT_VECTORS[cfg.MODEL] if enabled[cfg.MODEL] else 0.0
        total = r + m
        return (r / total, m / total) if total else (0.0, 0.0)

    @staticmethod
    def chunk_learning_scores(
        regex_score: float, chunk_scores: Sequence[float], weights: tuple[float, float]
    ) -> list[float]:
        """Per chunk: weighted average of the prompt's regex score and the chunk's model score."""
        wr, wm = weights
        return [wr * regex_score + wm * s for s in chunk_scores]

    def detect(self, prompt: str) -> DetectionReport:
        """Score one prompt with every switched-on layer and return the full report."""
        start = time.perf_counter()
        enabled = self.config.enabled()
        vector_empty = True
        if enabled[cfg.VECTOR]:
            vector_empty = self.vector_layer.is_empty()
        weights = cfg.resolve_weights(self.config, vector_empty)

        results: dict[str, LayerResult] = {}
        if enabled[cfg.REGEX]:
            results["regex"] = self.regex_layer.score(prompt)
        if enabled[cfg.MODEL]:
            results["model"] = self.model_layer.score(prompt)
        if enabled[cfg.VECTOR]:
            results["vector"] = self.vector_layer.score(prompt)

        layers: dict[str, LayerScore] = {}
        final = 0.0
        for name in LAYERS:
            w = weights[LAYER_KEYS[name]]
            r = results.get(name)
            s = r.score if r else 0.0
            final += w * s
            layers[name] = LayerScore(
                layer=name,
                enabled=r is not None,
                score=s,
                weight=w,
                contribution=w * s,
                fired=bool(r and r.fired),
                attack_type=r.attack_type.value if r and r.attack_type else None,
                seconds=r.seconds if r else 0.0,
            )
        final = min(max(final, 0.0), 1.0)
        is_injection = final >= self.config.detect_threshold
        attack_type = self._attack_type(results, is_injection)

        learned = 0
        model_r, regex_r, vector_r = (
            results.get("model"),
            results.get("regex"),
            results.get("vector"),
        )
        if enabled[cfg.VECTOR] and self.vector_config.learning and (model_r or regex_r):
            if model_r is not None:
                chunks = model_r.detail.get("_chunks") or [prompt]
                chunk_scores = model_r.detail.get("chunk_scores") or [model_r.score]
            else:
                chunks, chunk_scores = [prompt], [0.0]
            scores = self.chunk_learning_scores(
                regex_r.score if regex_r else 0.0, chunk_scores, self._learning_weights()
            )
            learned = self.vector_layer.learn(chunks, scores, attack_type if is_injection else None)

        md = model_r.detail if model_r else {}
        vd = vector_r.detail if vector_r else {}
        report = DetectionReport(
            score=final,
            is_injection=is_injection,
            attack_type=attack_type,
            detected_by=tuple(n for n in LAYERS if layers[n].fired),
            layers=layers,
            detect_threshold=self.config.detect_threshold,
            catch_threshold=self.vector_config.catch_threshold if enabled[cfg.VECTOR] else None,
            learn_threshold=self.vector_config.learn_threshold if enabled[cfg.VECTOR] else None,
            prompt_tokens=md.get("prompt_tokens"),
            chunk_size=md.get("chunk_size"),
            chunk_count=md.get("chunk_count"),
            detecting_chunk_tokens=md.get("detecting_chunk_tokens"),
            detecting_chunk_index=md.get("detecting_chunk_index"),
            vectors_in_store=vd.get("vectors_in_store"),
            vector_top=tuple(
                VectorMatch(i + 1, t["text"], t["similarity"], t["source"], t["attack_type"])
                for i, t in enumerate(vd.get("top_texts") or [])
            ),
            max_similarity=vd.get("max_similarity"),
            learned_chunks=learned,
            db_type=vd.get("db_type"),
            embedder_model=vd.get("embedder"),
            embedder_dimensions=vd.get("embedder_dimensions"),
            mode=self.mode,
            library_version=__version__,
            local_model_version=md.get("model_version") if self.mode != HOSTED else None,
            hosted_model=md.get("hosted_model") if self.mode == HOSTED else None,
            hosted_endpoint=md.get("hosted_endpoint") if self.mode == HOSTED else None,
            detection_ms=(time.perf_counter() - start) * 1000,
        )
        self.history.append(report)
        return report

    @staticmethod
    def _attack_type(results: dict[str, LayerResult], is_injection: bool) -> AttackType:
        """Benign below the threshold; otherwise the model's type, then regex's, then vector's."""
        if not is_injection:
            return AttackType.BENIGN
        for name in ("model", "regex", "vector"):
            r = results.get(name)
            if r and r.fired and r.attack_type and r.attack_type is not AttackType.BENIGN:
                return r.attack_type
        return AttackType.OTHER_INJECTION

    # ----------------------------------------------------------------------- saving vectors

    def _hosted_repository(self, dimension: int, **overrides: Any) -> VectorRepository:
        vc = self.vector_config
        url = overrides.get("url") or vc.provider_url
        if not url and not overrides.get("location"):
            raise cfg.ConfigError(
                "no Qdrant URL: pass url=..., set VectorConfig(provider_url) or QDRANT_URL"
            )
        return get_vector_repository(
            "qdrant",
            dimension=dimension,
            collection=overrides.get("collection") or vc.collection,
            url=None if overrides.get("location") else url,
            api_key=overrides.get("api_key") or vc.provider_api_key,
            location=overrides.get("location"),
        )

    def save(self, target: str | Path = HOSTED, **hosted: Any) -> int:
        """Copy the vector store. `save("path/to/file.db")`: to a Milvus Lite file.
        `save("hosted")` (or `save("hosted", url=..., api_key=..., collection=...)`): to Qdrant,
        with defaults from `VectorConfig` / `.env`. Returns how many vectors were written."""
        source: VectorRepository = self.vector_layer.repository
        dim = source.dimension
        if str(target) == HOSTED:
            dest = self._hosted_repository(dim, **hosted)
        else:
            path = Path(target)
            if (
                getattr(source, "path", None) is not None
                and Path(source.path).resolve() == path.resolve()
            ):
                return source.count()  # already there; Milvus Lite writes as it goes
            dest = get_vector_repository(
                "milvus_lite", path=path, dimension=dim, collection=source.collection
            )
        written = 0
        batch: list[Any] = []
        with dest:
            for record in source.iter_records():
                batch.append(record)
                if len(batch) >= 256:
                    written += dest.upsert(batch)
                    batch = []
            if batch:
                written += dest.upsert(batch)
        logger.info("havelai: saved %d vectors to %r", written, dest)
        return written

    # ----------------------------------------------------------------------- analytics

    def save_metrics(self, target: str | Path = HOSTED, **remote: Any) -> int:
        """Store the detection history. A path: a SQLite file with the tables of SCHEMA.sql
        (plus the vector store's texts in `learned_vectors`). `"hosted"`: POST each report to
        the analytics endpoint (`url=` / `api_key=` or HAVEL_ANALYTICS_URL / _API_KEY)."""
        if str(target) == HOSTED:
            repo = get_record_repository("remote_http", **remote)
        else:
            repo = get_record_repository("sqlite", path=target)
        with repo:
            written = repo.write([r.to_dict() for r in self.history])
            if self._vector is not None and hasattr(repo, "write_vectors"):
                source = self._vector.repository
                repo.write_vectors(list(source.iter_records()), source.dimension)
        return written

    def export_metrics_summary(self, path: str | Path) -> Path:
        """Markdown summary of the detection history with matplotlib plots next to it."""
        from ...analytics.export import export_summary

        return export_summary(self.history, path)
