"""Local detection: `from havelai.local import Havel, Config` (needs `havelai[local]`)."""

from .analytics.report import DetectionReport
from .services.common.config import Config, ConfigError, ModelConfig, VectorConfig
from .services.local.detector_engine import Havel
from .services.local.vector_learning_catch import VectorIngestError

__all__ = [
    "Config",
    "ConfigError",
    "DetectionReport",
    "Havel",
    "ModelConfig",
    "VectorConfig",
    "VectorIngestError",
]
