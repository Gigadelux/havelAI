"""mmBERT models for prompt-injection detection.

Two models live here. See model/docs/ARCHITECTURE.md for the design and the reasoning.

1. `build_classifier`: the single-head baseline. `AutoModelForSequenceClassification` with one
   output per class in `src.data.taxonomy.LABELS`. Kept so the two-head model can be compared
   against it.
2. `HavelForInjectionDetection` / `build_two_head`: the target model. One shared mmBERT encoder
   and one pooled vector, read by two small linear heads:
     - a binary head: one logit, "is this an injection?";
     - a type head: one logit per attack type (the 9 classes that are not `benign`).
   The heads never read each other. They meet in the loss (`two_head_loss`) and in the
   decision rule (`decide`, `class_probabilities`).

Everything that needs torch or transformers is imported lazily, so importing this module (and
`model.train.train --dry-run`) works without the `model` extra.

Not done here: vocabulary pruning (32k tokens) and the int8 ONNX export. They are separate
steps; this file only defines the architecture.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Any

from src.data.taxonomy import ID2LABEL, LABEL2ID, LABELS, AttackType

# The one encoder checkpoint. mmBERT-small: 140M parameters, 98M of them in the embedding table.
# mmBERT-base (307M) is not used: it cannot meet the 256 MB model limit, even in int8.
ENCODER_MODEL = "jhu-clsp/mmBERT-small"

BENIGN = AttackType.BENIGN.value
# The type head's classes: every label except `benign`, in taxonomy order. LABELS is
# (benign, *TYPE_LABELS), so class id k of the type head is class id k + 1 of LABELS.
TYPE_LABELS: tuple[str, ...] = tuple(label for label in LABELS if label != BENIGN)
TYPE2ID: dict[str, int] = {label: i for i, label in enumerate(TYPE_LABELS)}
assert LABELS[0] == BENIGN and LABELS[1:] == TYPE_LABELS

# --------------------------------------------------------------------------- baseline


def build_classifier(base_model: str = ENCODER_MODEL):
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


# --------------------------------------------------------------------------- targets (no torch)


def type_target_from_record(
    attack_type: str | None, probs: dict[str, float] | None = None
) -> list[float] | None:
    """Soft target over `TYPE_LABELS` for the type head, from one enhanced-dataset row.

    Returns None for rows the type head must not learn from: `benign`, `unlabeled`, or any label
    that is not an attack type. With `probs` (the labeler's `attack_type_probs`) the target is
    that distribution restricted to the nine types and renormalized, so an ambiguous row pushes
    the head less sharply. Without usable probabilities it is one-hot on `attack_type`.
    """
    if attack_type not in TYPE2ID:
        return None
    one_hot = [0.0] * len(TYPE_LABELS)
    one_hot[TYPE2ID[attack_type]] = 1.0
    if not probs:
        return one_hot
    weights = [max(0.0, float(probs.get(label, 0.0))) for label in TYPE_LABELS]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return one_hot
    return [w / total for w in weights]


# --------------------------------------------------------------------------- model


@functools.cache
def _define_classes() -> tuple[Any, Any]:
    """Define the output type and the model class. Deferred so torch is not needed at import."""
    from dataclasses import dataclass

    import torch
    from torch import nn
    from transformers import initialization as init
    from transformers.modeling_outputs import ModelOutput
    from transformers.models.modernbert.modeling_modernbert import (
        ModernBertModel,
        ModernBertPredictionHead,
        ModernBertPreTrainedModel,
    )

    @dataclass
    class HavelOutput(ModelOutput):
        """binary_logit: shape (batch,). type_logits: shape (batch, len(TYPE_LABELS))."""

        binary_logit: torch.Tensor | None = None
        type_logits: torch.Tensor | None = None

    class HavelForInjectionDetection(ModernBertPreTrainedModel):
        """Shared mmBERT encoder, pooling, shared projection, then a binary head and a type head.

        The config is a plain `ModernBertConfig` (so the encoder loads from any mmBERT
        checkpoint) with `classifier_pooling` ("mean" or "cls") and `num_attack_types` set.
        """

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            n_types = getattr(config, "num_attack_types", len(TYPE_LABELS))
            if n_types != len(TYPE_LABELS):
                raise ValueError(
                    f"config.num_attack_types={n_types}, but the taxonomy has "
                    f"{len(TYPE_LABELS)} attack types"
                )
            self.model = ModernBertModel(config)
            self.head = ModernBertPredictionHead(config)  # shared: dense + GELU + LayerNorm
            self.drop = nn.Dropout(config.classifier_dropout)
            self.binary_head = nn.Linear(config.hidden_size, 1)
            self.type_head = nn.Linear(config.hidden_size, n_types)
            self.post_init()

        def _init_weights(self, module: nn.Module) -> None:
            super()._init_weights(module)
            if module is getattr(self, "binary_head", None) or module is getattr(
                self, "type_head", None
            ):
                std = self.config.hidden_size**-0.5  # same as the library's classifier layers
                cutoff = (self.config.initializer_cutoff_factor or 3) * std
                init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff, b=cutoff)
                init.zeros_(module.bias)

        def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
            if self.config.classifier_pooling == "cls":
                return hidden[:, 0]
            if attention_mask is None:
                attention_mask = torch.ones(hidden.shape[:2], device=hidden.device)
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
        ) -> HavelOutput:
            hidden = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
            )[0]
            z = self.drop(self.head(self._pool(hidden, attention_mask)))
            return HavelOutput(
                binary_logit=self.binary_head(z).squeeze(-1), type_logits=self.type_head(z)
            )

    return HavelOutput, HavelForInjectionDetection


def __getattr__(name: str) -> Any:
    # `from model.architecture import HavelForInjectionDetection` works without importing torch
    # until the name is actually used.
    if name == "HavelOutput":
        return _define_classes()[0]
    if name == "HavelForInjectionDetection":
        return _define_classes()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_config(base_model: str = ENCODER_MODEL, pooling: str = "mean") -> Any:
    """ModernBertConfig of a base checkpoint, set up for the two-head model."""
    from transformers import ModernBertConfig

    if pooling not in ("mean", "cls"):
        raise ValueError("pooling must be 'mean' or 'cls'")
    config = ModernBertConfig.from_pretrained(base_model)
    return configure(config, pooling)


def configure(config: Any, pooling: str = "mean") -> Any:
    """Set the two-head fields on a ModernBertConfig and return it."""
    config.classifier_pooling = pooling
    config.num_attack_types = len(TYPE_LABELS)
    config.type_labels = list(TYPE_LABELS)  # makes a saved checkpoint self-describing
    config.architectures = ["HavelForInjectionDetection"]
    return config


def build_two_head(base_model: str = ENCODER_MODEL, pooling: str = "mean"):
    """Return (tokenizer, model): the two-head model on a pretrained mmBERT encoder.

    The encoder and the shared projection start from the checkpoint (the projection from its
    masked-language-model head, as the library's own classifier does); the two output heads
    start random. Same return shape as `build_classifier`.
    """
    from transformers import AutoTokenizer

    config = make_config(base_model, pooling)
    model_cls = _define_classes()[1]
    model = model_cls.from_pretrained(base_model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return tokenizer, model


# --------------------------------------------------------------------------- loss + decision


def two_head_loss(
    binary_logit: Any,
    type_logits: Any,
    is_injection: Any,
    type_targets: Any,
    type_mask: Any,
    *,
    type_class_weights: Sequence[float] | None = None,
    type_weight: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """L = L_binary + type_weight * L_type. Returns (total, {"binary": ..., "type": ...}).

    - L_binary: binary cross-entropy against the dataset's original label (`is_injection`, 0 or
      1) on ALL rows.
    - L_type: soft-label cross-entropy against `type_targets` (shape (batch, 9), a probability
      distribution per row) on the rows where `type_mask` is true, normally the attack rows
      that have a type label. Averaged over those rows; 0 when there are none.
    - `type_class_weights` (length 9) scales each class's term, for the rare attack types.
    """
    import torch
    import torch.nn.functional as F

    binary = F.binary_cross_entropy_with_logits(binary_logit.float(), is_injection.float())

    mask = type_mask.bool()
    if mask.any():
        log_q = F.log_softmax(type_logits.float()[mask], dim=-1)
        targets = type_targets.float()[mask]
        if type_class_weights is not None:
            w = torch.as_tensor(type_class_weights, dtype=log_q.dtype, device=log_q.device)
            log_q = log_q * w
        type_loss = -(targets * log_q).sum(dim=-1).mean()
    else:
        type_loss = type_logits.float().sum() * 0.0  # keeps the graph, contributes nothing

    return binary + type_weight * type_loss, {"binary": binary, "type": type_loss}


def class_probabilities(binary_logit: Any, type_logits: Any) -> Any:
    """Probabilities over `LABELS` (benign first), shape (batch, len(LABELS)), rows sum to 1.

    P(benign) = 1 - p_inj and P(type k) = p_inj * q_k.
    """
    import torch

    p = torch.sigmoid(binary_logit.float()).unsqueeze(-1)
    q = torch.softmax(type_logits.float(), dim=-1)
    return torch.cat([1.0 - p, p * q], dim=-1)


def decide(binary_logit: Any, type_logits: Any, threshold: float = 0.5) -> tuple[Any, Any]:
    """Return (label ids into `LABELS`, scores).

    The binary head decides first: if p_inj >= threshold the text is an attack, and the type head
    picks the most likely type (score = p_inj * q_max); otherwise it is `benign`
    (score = 1 - p_inj). The score is the probability of the reported class under
    `class_probabilities`.
    """
    import torch

    p = torch.sigmoid(binary_logit.float())
    q = torch.softmax(type_logits.float(), dim=-1)
    q_max, top = q.max(dim=-1)
    attack = p >= threshold
    ids = torch.where(attack, top + 1, torch.zeros_like(top))  # +1: LABELS[0] is benign
    scores = torch.where(attack, p * q_max, 1.0 - p)
    return ids, scores


# --------------------------------------------------------------------------- size


def parameter_counts(model: Any) -> dict[str, int]:
    """Parameters per part of a `HavelForInjectionDetection`."""

    def count(module: Any) -> int:
        return sum(p.numel() for p in module.parameters())

    embeddings = count(model.model.embeddings.tok_embeddings)
    total = count(model)
    shared = count(model.head)
    binary, type_head = count(model.binary_head), count(model.type_head)
    return {
        "embeddings": embeddings,
        "encoder": total - embeddings - shared - binary - type_head,
        "shared_projection": shared,
        "binary_head": binary,
        "type_head": type_head,
        "total": total,
    }


def estimated_size_mb(n_params: int, bytes_per_param: float) -> float:
    """Model size in MB (1 MB = 1e6 bytes): 1 byte per parameter for int8, 2 fp16, 4 fp32."""
    return n_params * bytes_per_param / 1e6
