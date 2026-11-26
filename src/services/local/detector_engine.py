"""Local detector engine: regex + havel_base (ONNX) + Milvus Lite, all on this machine.

    from havelai.local import Havel

    h = Havel()
    h.Config(regex_catcher=True, model_catcher=True, vector_catcher=False)
    report = h.detect("...")

The model is downloaded from Hugging Face on first use (logged). Needs `pip install havelai[local]`.
"""

from __future__ import annotations

from typing import Any

from ..common.engine import HavelEngine
from ..common.layer import Layer


class Havel(HavelEngine):
    mode = "local"

    def _build_model(self) -> Layer:
        from .model_catch import LocalModelCatcher

        return LocalModelCatcher(self.model_config)

    def _build_vector(self) -> Any:
        from .vector_learning_catch import local_vector_catcher

        return local_vector_catcher(self.vector_config)
