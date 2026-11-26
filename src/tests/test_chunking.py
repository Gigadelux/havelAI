import pytest

from src.services.common.chunking import (
    approx_tokens,
    chunk_size,
    chunk_text,
    split_sentences,
)


@pytest.mark.parametrize(
    "total, expected",
    [(0, 64), (4, 64), (5, 96), (100, 320), (256, 512), (10_000, 512)],
)
def test_chunk_size_grows_with_input(total, expected):
    assert chunk_size(total) == expected


def test_chunk_size_respects_bounds():
    assert chunk_size(100, min_tokens=8, max_tokens=128) == 128
    assert chunk_size(0, min_tokens=8, max_tokens=128) == 8


def test_split_sentences():
    assert split_sentences("One. Two!  Three?\n\nFour") == ["One.", "Two!", "Three?", "Four"]
    assert split_sentences("   ") == []


def test_short_prompt_is_one_chunk():
    c = chunk_text("Hello there. How are you?")
    assert c.chunks == ["Hello there. How are you?"]
    assert c.total_tokens == approx_tokens("Hello there. How are you?")


def test_long_text_is_chunked_within_size_and_keeps_every_word():
    sentence = "This is a perfectly ordinary sentence about the weather today."
    text = " ".join([sentence] * 400)
    c = chunk_text(text)
    assert c.chunk_tokens == 512
    assert len(c.chunks) > 1
    assert all(approx_tokens(ch) <= c.chunk_tokens for ch in c.chunks)
    assert " ".join(c.chunks).split() == text.split()


def test_oversized_sentence_is_split_on_words():
    text = " ".join(["word"] * 3000)  # one "sentence", 3000 tokens
    c = chunk_text(text, max_tokens=512)
    assert all(approx_tokens(ch) <= 512 for ch in c.chunks)
    assert sum(approx_tokens(ch) for ch in c.chunks) == 3000


def test_custom_counter_and_empty_text():
    c = chunk_text("a b c. d e f.", count=lambda s: len(s.split()), min_tokens=1, max_tokens=3)
    assert c.chunks == ["a b c.", "d e f."]
    assert chunk_text("").chunks == []
