"""Hosted detector engine: regex (here) + a guard model at a provider + Qdrant.

    from havelai.hosted import Havel

    h = Havel().ModelConfig(hosted_model="openai/gpt-oss-safeguard-20b")
    report = h.detect("...")

Provider URL and key: `ModelConfig(provider_url=..., provider_api_key=...)` or `.env`
(`HAVEL_HOSTED_BASE_URL`, `HAVEL_HOSTED_API_KEY` / `OPENROUTER_API_KEY`). Qdrant: `VectorConfig`
or `QDRANT_URL` / `QDRANT_API_KEY`. Prompts are sent to the provider. Needs `havelai[hosted]`.
"""

from __future__ import annotations

from typing import Any

from ..common.engine import HavelEngine
from ..common.layer import Layer


class Havel(HavelEngine):
    mode = "hosted"

    def _build_model(self) -> Layer:
        from .model_catch import HostedModelCatcher

        return HostedModelCatcher(self.model_config)

    def _build_vector(self) -> Any:
        from .vector_learning_catch import hosted_vector_catcher

        return hosted_vector_catcher(self.vector_config)
