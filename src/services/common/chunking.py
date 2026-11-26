"""Split a prompt into sentence-aligned chunks for the model layer.

The chunk size grows with the input: `clamp(min, max, 32 * ceil(sqrt(total_tokens)))`.

| prompt tokens | chunk tokens (64..512) |
|---|---|
| up to 4 | 64 (the whole prompt is one chunk) |
| 100 | 320 (one chunk) |
| 256 | 512 |
| 10,000 | 512 (about 20 chunks) |

Short prompts stay whole; long ones are cut into chunks of up to 512 tokens, the length the model
was trained on. Each chunk is scored on its own, so one malicious sentence in a long benign text
is not averaged away. Stdlib only; the token counter is passed in.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

TokenCounter = Callable[[str], int]

_SENTENCE_END = re.compile(r"(?<=[.!?。！？;])\s+|\n+")
_APPROX_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SIZE_FACTOR = 32


def approx_tokens(text: str) -> int:
    """Words and punctuation marks: a tokenizer-free estimate, used by the hosted service."""
    return len(_APPROX_TOKEN.findall(text))


def chunk_size(total_tokens: int, min_tokens: int = 64, max_tokens: int = 512) -> int:
    grown = SIZE_FACTOR * math.ceil(math.sqrt(max(total_tokens, 0)))
    return max(min_tokens, min(max_tokens, grown))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


@dataclass(frozen=True, slots=True)
class Chunking:
    chunks: list[str]
    chunk_tokens: int  # the target size used
    total_tokens: int


def _split_long(sentence: str, size: int, count: TokenCounter) -> list[str]:
    """A sentence longer than `size`: cut on whitespace into pieces of at most `size` tokens."""
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for word in sentence.split():
        n = max(count(word), 1)
        if current and used + n > size:
            pieces.append(" ".join(current))
            current, used = [], 0
        current.append(word)
        used += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    text: str,
    count: TokenCounter = approx_tokens,
    min_tokens: int = 64,
    max_tokens: int = 512,
) -> Chunking:
    """Pack sentences greedily into chunks of at most `chunk_size(total)` tokens."""
    sentences = split_sentences(text)
    counts = [count(s) for s in sentences]
    total = sum(counts)
    size = chunk_size(total, min_tokens, max_tokens)
    if not sentences:
        return Chunking([], size, 0)
    chunks: list[str] = []
    current: list[str] = []
    used = 0
    for sentence, n in zip(sentences, counts, strict=True):
        if n > size:
            if current:
                chunks.append(" ".join(current))
                current, used = [], 0
            chunks.extend(_split_long(sentence, size, count))
            continue
        if current and used + n > size:
            chunks.append(" ".join(current))
            current, used = [], 0
        current.append(sentence)
        used += n
    if current:
        chunks.append(" ".join(current))
    return Chunking(chunks, size, total)
