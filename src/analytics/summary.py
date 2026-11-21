"""Aggregate detection results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from ..detection.base import Detection


@dataclass(frozen=True, slots=True)
class Summary:
    total: int
    injections: int
    by_attack_type: dict[str, int]

    @property
    def injection_rate(self) -> float:
        return self.injections / self.total if self.total else 0.0


def summarize(detections: Iterable[Detection]) -> Summary:
    """Count detections per attack type and the share that are injections."""
    counts: Counter[str] = Counter()
    injections = 0
    for d in detections:
        counts[d.attack_type.value] += 1
        injections += d.is_injection
    return Summary(
        total=sum(counts.values()),
        injections=injections,
        by_attack_type=dict(counts),
    )
