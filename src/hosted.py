"""Hosted detection: `from havelai.hosted import Havel, Config` (needs `havelai[hosted]`)."""

from .analytics.report import DetectionReport
from .services.common.config import Config, ConfigError, ModelConfig, VectorConfig
from .services.hosted.detector_engine import Havel
from .services.hosted.model_catch import HostedModelError
from .services.hosted.vector_learning_catch import VectorIngestError

__all__ = [
    "Config",
    "ConfigError",
    "DetectionReport",
    "Havel",
    "HostedModelError",
    "ModelConfig",
    "VectorConfig",
    "VectorIngestError",
]
