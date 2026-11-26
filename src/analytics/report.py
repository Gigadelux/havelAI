"""The result of one `Havel.detect()` call, with everything needed for analytics.

Stdlib only. The prompt itself is never stored in a report. The only texts kept are the vector
layer's top matches, which come from the user's own vector store.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..data.taxonomy import AttackType
from ..detection.base import Detection

LAYERS = ("regex", "model", "vector")


@dataclass(frozen=True, slots=True)
class LayerScore:
    layer: str
    enabled: bool
    score: float  # the layer's own score, 0..1
    weight: float
    contribution: float  # weight x score
    fired: bool
    attack_type: str | None
    seconds: float


@dataclass(frozen=True, slots=True)
class VectorMatch:
    rank: int
    text: str
    similarity: float
    source: str
    attack_type: str | None


@dataclass(frozen=True)
class DetectionReport:
    # verdict
    score: float  # final weighted score
    is_injection: bool
    attack_type: AttackType
    detected_by: tuple[str, ...]  # layers that fired
    layers: dict[str, LayerScore]
    # thresholds
    detect_threshold: float
    catch_threshold: float | None
    learn_threshold: float | None
    # prompt and chunking
    prompt_tokens: int | None
    chunk_size: int | None  # target chunk size used for this prompt
    chunk_count: int | None
    detecting_chunk_tokens: int | None
    detecting_chunk_index: int | None
    # vector layer
    vectors_in_store: int | None  # at detection time
    vector_top: tuple[VectorMatch, ...]
    max_similarity: float | None
    learned_chunks: int
    db_type: str | None
    embedder_model: str | None
    embedder_dimensions: int | None
    # versions and providers
    mode: str  # "local" | "hosted"
    library_version: str
    local_model_version: str | None  # None when the hosted service is used
    hosted_model: str | None  # None when the local service is used
    hosted_endpoint: str | None  # None when the local service is used
    # bookkeeping
    detection_ms: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds")
    )

    # ----------------------------------------------------------------------- views

    @property
    def final_score(self) -> float:
        return self.score

    @property
    def weights(self) -> dict[str, float]:
        return {name: layer.weight for name, layer in self.layers.items()}

    @property
    def contributions(self) -> dict[str, float]:
        return {name: layer.contribution for name, layer in self.layers.items()}

    @property
    def scores(self) -> dict[str, float]:
        return {name: layer.score for name, layer in self.layers.items()}

    @property
    def top_texts(self) -> list[str]:
        return [m.text for m in self.vector_top]

    def to_detection(self, text: str = "") -> Detection:
        """As the library's basic `Detection`, so `analytics.summary.summarize` works on reports.

        `score` there is the confidence in `attack_type`: the final score for an injection,
        1 - final score for benign."""
        confidence = self.score if self.is_injection else 1.0 - self.score
        return Detection(text=text, attack_type=self.attack_type, score=confidence)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (the shape stored by the record repositories, see SCHEMA.sql)."""
        d = asdict(self)
        d["attack_type"] = self.attack_type.value
        d["detected_by"] = list(self.detected_by)
        d["vector_top"] = [asdict(m) for m in self.vector_top]
        d["layers"] = {k: asdict(v) for k, v in self.layers.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectionReport:
        data = dict(d)
        data["attack_type"] = AttackType(data["attack_type"])
        data["detected_by"] = tuple(data.get("detected_by") or ())
        data["vector_top"] = tuple(VectorMatch(**m) for m in data.get("vector_top") or ())
        data["layers"] = {k: LayerScore(**v) for k, v in (data.get("layers") or {}).items()}
        return cls(**data)
