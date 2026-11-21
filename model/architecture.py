"""mmBERT sequence classifier. Placeholder: the architecture is decided when training starts.

Base checkpoints: `jhu-clsp/mmBERT-base` (307M params) or `jhu-clsp/mmBERT-small` (140M), both
ModernBERT-based with an 8192-token context and MIT licensed. They load through
`AutoModelForSequenceClassification`.
"""

from __future__ import annotations

from src.data.taxonomy import ID2LABEL, LABEL2ID, LABELS

BASE_MODEL = "jhu-clsp/mmBERT-base"


def build_classifier(base_model: str = BASE_MODEL):
    """Return (tokenizer, model) with one output per class in `src.data.taxonomy.LABELS`."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )  # lazy: `model` extra

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    return tokenizer, model
