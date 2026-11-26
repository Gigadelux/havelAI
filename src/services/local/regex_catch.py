"""Regex layer: the cheapest check. Binary: 1.0 if any static pattern matches, else 0.0.

Patterns live in `src/data/regex_patterns.py`. Stdlib only, so it also runs without any extra.
"""

from __future__ import annotations

import time
from collections import Counter

from ...data.regex_patterns import PATTERNS, RegexPattern
from ...data.taxonomy import AttackType
from ..common.layer import LayerResult

NAME = "regex"


def regex_matches(text: str, patterns: tuple[RegexPattern, ...] = PATTERNS) -> list[RegexPattern]:
    """Every pattern that matches somewhere in `text`."""
    return [p for p in patterns if p.regex.search(text)]


def regex_catch(text: str) -> bool:
    """True when any pattern matches: `catch = re.search(pattern); if catch: detected = True`."""
    return any(p.regex.search(text) for p in PATTERNS)


def dominant_type(matches: list[RegexPattern]) -> AttackType | None:
    """The attack type with the most matching patterns; ties go to taxonomy order."""
    if not matches:
        return None
    counts = Counter(m.attack_type for m in matches)
    order = list(AttackType)
    return max(counts, key=lambda t: (counts[t], -order.index(t)))


class RegexCatcher:
    """`Layer` wrapper used by the detector engines."""

    name = NAME

    def __init__(self, patterns: tuple[RegexPattern, ...] = PATTERNS) -> None:
        self.patterns = patterns

    def score(self, text: str) -> LayerResult:
        start = time.perf_counter()
        matches = regex_matches(text, self.patterns)
        detected = bool(matches)
        return LayerResult(
            layer=NAME,
            score=1.0 if detected else 0.0,
            fired=detected,
            attack_type=dominant_type(matches),
            seconds=time.perf_counter() - start,
            detail={
                "matched_patterns": [m.id for m in matches],
                "pattern_count": len(self.patterns),
            },
        )
