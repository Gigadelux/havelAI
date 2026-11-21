"""Detector interface. Concrete detectors (the local mmBERT classifier, heuristics) come later."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..data.taxonomy import AttackType


@dataclass(frozen=True, slots=True)
class Detection:
    """Result of scanning one text."""

    text: str
    attack_type: AttackType
    score: float  # confidence in `attack_type`, 0..1

    @property
    def is_injection(self) -> bool:
        return self.attack_type is not AttackType.BENIGN


@runtime_checkable
class Detector(Protocol):
    def detect(self, text: str) -> Detection: ...
