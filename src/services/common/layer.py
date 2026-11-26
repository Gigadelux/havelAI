"""The contract between a detection layer (regex, model, vector) and the detector engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...data.taxonomy import AttackType


@dataclass(frozen=True, slots=True)
class LayerResult:
    """What one layer says about one prompt.

    `score` is in [0, 1] and is what the engine weights: 1/0 for the regex layer, the highest
    chunk `p_injection` for the model, the highest similarity for the vector layer.
    `detail` holds layer-specific analytics (matched patterns, chunk sizes, top texts...).
    """

    layer: str
    score: float
    fired: bool
    attack_type: AttackType | None = None
    seconds: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Layer(Protocol):
    name: str

    def score(self, text: str) -> LayerResult: ...
