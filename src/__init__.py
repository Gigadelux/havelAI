"""havelai: detect and classify prompt-injection attacks.

The repo folder is `src/`; the installed package is `havelai` (see pyproject.toml).
"""

from .data.taxonomy import ATTACK_CRITERIA, AttackType
from .detection.base import Detection, Detector

__version__ = "0.1.0"

__all__ = ["ATTACK_CRITERIA", "AttackType", "Detection", "Detector", "__version__"]
