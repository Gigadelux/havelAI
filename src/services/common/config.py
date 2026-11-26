"""Configuration shared by the local and hosted detector engines.

Stdlib only. The engines read these dataclasses; nothing here imports a heavy dependency.

Weights (`detector_scoring`) decide how much each layer adds to the final score:

- left as ``None`` (default), they follow the layers that are switched on and the vector store:
  regex 0.1 / model 0.9 while the vector layer is off or its store is empty, and
  regex 0.1 / model 0.65 / vector 0.25 once the store has vectors. Weights of layers that are
  off are dropped and the rest are rescaled to sum to 1.
- given explicitly, they are checked: only the three known keys, every value in [0, 1], a
  switched-off layer must weigh 0, a switched-on layer must weigh more than 0, and the sum must
  be 1. Explicit weights are used as they are, also while the vector store is empty.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from .env import get_env

REGEX = "regex_contribution"
MODEL = "model_contribution"
VECTOR = "vector_contribution"
WEIGHT_KEYS: tuple[str, ...] = (REGEX, MODEL, VECTOR)

WEIGHTS_WITHOUT_VECTORS: dict[str, float] = {REGEX: 0.1, MODEL: 0.9, VECTOR: 0.0}
WEIGHTS_WITH_VECTORS: dict[str, float] = {REGEX: 0.1, MODEL: 0.65, VECTOR: 0.25}
SUM_TOLERANCE = 1e-6

DEFAULT_HF_REPO = "Gigadelux/havel_base"
DEFAULT_EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_HOSTED_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_HOSTED_MODEL = "meta-llama/llama-guard-4-12b"


class ConfigError(ValueError):
    """The configuration is inconsistent (weights, thresholds, switched-off layers)."""


SECRET_FIELDS = frozenset({"hf_token", "provider_api_key"})


def safe_repr(obj: object) -> str:
    """repr without secrets, so configs can be logged."""
    fields = ", ".join(
        f"{k}={'***' if k in SECRET_FIELDS and v else repr(v)}" for k, v in vars(obj).items()
    )
    return f"{type(obj).__name__}({fields})"


def _check_unit(name: str, value: float) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be a number between 0 and 1, got {value!r}")


@dataclass
class Config:
    """Which layers run and how their scores are combined."""

    regex_catcher: bool = True
    model_catcher: bool = True
    vector_catcher: bool = False
    detector_scoring: dict[str, float] | None = None  # None: automatic, see the module docstring
    detect_threshold: float = 0.5  # final score at or above this is an injection

    def __post_init__(self) -> None:
        _check_unit("detect_threshold", self.detect_threshold)
        if not (self.regex_catcher or self.model_catcher or self.vector_catcher):
            raise ConfigError("at least one catcher must be switched on")
        if self.detector_scoring is not None:
            self.detector_scoring = validate_weights(self.detector_scoring, self.enabled())

    def enabled(self) -> dict[str, bool]:
        return {REGEX: self.regex_catcher, MODEL: self.model_catcher, VECTOR: self.vector_catcher}


def validate_weights(weights: Mapping[str, float], enabled: Mapping[str, bool]) -> dict[str, float]:
    """Check explicit weights against the switched-on layers. Returns all three keys."""
    unknown = set(weights) - set(WEIGHT_KEYS)
    if unknown:
        raise ConfigError(f"unknown detector_scoring keys {sorted(unknown)}; use {WEIGHT_KEYS}")
    full = {key: float(weights.get(key, 0.0)) for key in WEIGHT_KEYS}
    for key, value in full.items():
        _check_unit(key, value)
        if not enabled[key] and value > 0:
            raise ConfigError(f"{key} is {value} but that catcher is switched off; set it to 0")
        if enabled[key] and value == 0:
            raise ConfigError(f"{key} is 0 but that catcher is switched on; give it a weight")
    total = sum(full.values())
    if not math.isclose(total, 1.0, abs_tol=SUM_TOLERANCE):
        raise ConfigError(f"detector_scoring must sum to 1, got {total:.6f}")
    return full


def resolve_weights(config: Config, vector_store_empty: bool) -> dict[str, float]:
    """The weights to use for one detection."""
    if config.detector_scoring is not None:
        return dict(config.detector_scoring)
    enabled = config.enabled()
    use_vectors = enabled[VECTOR] and not vector_store_empty
    base = WEIGHTS_WITH_VECTORS if use_vectors else WEIGHTS_WITHOUT_VECTORS
    active = {
        k: (v if enabled[k] and (k != VECTOR or use_vectors) else 0.0) for k, v in base.items()
    }
    total = sum(active.values())
    if total == 0:
        # only the vector layer is on and its store is empty: nothing can score yet
        return {REGEX: 0.0, MODEL: 0.0, VECTOR: 1.0 if enabled[VECTOR] else 0.0}
    return {k: v / total for k, v in active.items()}


@dataclass
class VectorConfig:
    """The vector layer: thresholds, store and embedder.

    `provider_url` / `provider_api_key` point at the hosted vector store (Qdrant). They are used
    by the hosted engine and by `Havel.save("hosted")`; empty values are read from `.env`
    (`QDRANT_URL`, `QDRANT_API_KEY`).
    """

    catch_threshold: float = 0.8  # max similarity at or above this flags the text
    learn_threshold: float = 0.7  # regex+model score at or above this stores the chunk
    learning: bool = True
    top_k: int = 10
    db_path: str = "havel_vectors.db"  # local Milvus Lite file
    collection: str = field(
        default_factory=lambda: get_env("HAVEL_VECTOR_COLLECTION", "havel_vectors")
    )
    embedder: str = DEFAULT_EMBEDDER
    provider_url: str | None = None
    provider_api_key: str | None = None

    def __post_init__(self) -> None:
        _check_unit("catch_threshold", self.catch_threshold)
        _check_unit("learn_threshold", self.learn_threshold)
        if self.top_k < 1:
            raise ConfigError(f"top_k must be at least 1, got {self.top_k}")
        self.provider_url = self.provider_url or get_env("QDRANT_URL")
        self.provider_api_key = self.provider_api_key or get_env("QDRANT_API_KEY")

    def __repr__(self) -> str:  # never print secrets
        return safe_repr(self)


Precision = Literal["int8", "fp32"]


@dataclass
class ModelConfig:
    """The model layer.

    Local engine: the havel_base classifier from Hugging Face (`repo_id`, `precision`,
    `revision`; `hf_token` from `HF_TOKEN` while the repo is private).
    Hosted engine: a guard model behind an OpenAI-compatible endpoint (`hosted_model`,
    `provider_url`, `provider_api_key`, from `HAVEL_HOSTED_*` or `OPENROUTER_API_KEY`).
    """

    repo_id: str = DEFAULT_HF_REPO
    precision: Precision = "int8"
    revision: str | None = None
    cache_dir: str | None = None
    hf_token: str | None = None
    min_chunk_tokens: int = 64
    max_chunk_tokens: int = 512
    batch_size: int = 16
    hosted_model: str | None = None
    provider_url: str | None = None
    provider_api_key: str | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.precision not in ("int8", "fp32"):
            raise ConfigError(f"precision must be 'int8' or 'fp32', got {self.precision!r}")
        if not 1 <= self.min_chunk_tokens <= self.max_chunk_tokens:
            raise ConfigError("need 1 <= min_chunk_tokens <= max_chunk_tokens")
        self.hf_token = self.hf_token or get_env("HF_TOKEN")
        self.hosted_model = self.hosted_model or get_env("HAVEL_HOSTED_MODEL", DEFAULT_HOSTED_MODEL)
        self.provider_url = self.provider_url or get_env(
            "HAVEL_HOSTED_BASE_URL", DEFAULT_HOSTED_BASE_URL
        )
        self.provider_api_key = (
            self.provider_api_key
            or get_env("HAVEL_HOSTED_API_KEY")
            or get_env("OPENROUTER_API_KEY")
        )

    def __repr__(self) -> str:  # never print secrets
        return safe_repr(self)
