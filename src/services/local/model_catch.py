"""Model layer (local): the havel_base classifier, run with ONNX Runtime on this machine.

On first use the model is downloaded from Hugging Face (`ModelConfig.repo_id`, default
`Gigadelux/havel_base`) and cached by `huggingface_hub`; the download is logged on the
`havelai` logger. Default file: `onnx/model_int8.onnx` (about 142 MB); `precision="fp32"` uses
`onnx/model.onnx`. Needs the `local` extra (onnxruntime, tokenizers, huggingface_hub, numpy);
no torch.

Each prompt is split into sentence-aligned chunks whose size grows with the prompt
(`common/chunking.py`). Every chunk is scored on its own; the prompt's score is the highest chunk
`p_injection = 1 - P(benign)`, and the attack type is the most likely non-benign class of that
chunk.

Note: the current model was trained on mostly short standalone prompts. A data-augmentation
training batch (injections inserted into benign text in different ways) is planned so the model
gets better at spotting an injection inside a chunk of a longer text.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...data.taxonomy import LABELS, AttackType
from ..common.chunking import Chunking, chunk_text
from ..common.config import ModelConfig
from ..common.layer import LayerResult

NAME = "model"
FILES = {"int8": "onnx/model_int8.onnx", "fp32": "onnx/model.onnx"}
TOKENIZER_FILE = "tokenizer.json"
META_FILE = "havel_meta.json"
FIRE_THRESHOLD = 0.5  # a chunk with p_injection at or above this counts as detected
logger = logging.getLogger("havelai")


def softmax(logits: Any) -> Any:
    import numpy as np

    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def snapshot_revision(path: str | Path) -> str | None:
    """The commit sha from a huggingface_hub cache path (`.../snapshots/<sha>/...`)."""
    m = re.search(r"snapshots[/\\]([0-9a-f]{7,64})[/\\]", str(path))
    return m.group(1) if m else None


def read_probabilities(
    probs: Sequence[float], labels: Sequence[str] = LABELS
) -> tuple[float, AttackType, float]:
    """(p_injection, most likely attack type, its probability) from one softmax row."""
    benign = labels.index(AttackType.BENIGN.value)
    p_inj = 1.0 - float(probs[benign])
    best = max((i for i in range(len(labels)) if i != benign), key=lambda i: probs[i])
    return p_inj, AttackType(labels[best]), float(probs[best])


def truncate_ids(ids: Sequence[int], max_length: int) -> list[int]:
    """Cut to `max_length`, keeping the closing special token (as the tokenizer's truncation does)."""
    ids = list(ids)
    if len(ids) <= max_length:
        return ids
    return ids[: max_length - 1] + [ids[-1]]


class LocalModelCatcher:
    """`Layer` that scores prompts with havel_base. Pass `session`/`tokenizer` to skip the download."""

    name = NAME

    def __init__(
        self,
        config: ModelConfig | None = None,
        session: Any = None,
        tokenizer: Any = None,
        labels: Sequence[str] = LABELS,
        max_length: int = 512,
        model_version: str | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self._session = session
        self._tokenizer = tokenizer
        self.labels = tuple(labels)
        self.max_length = max_length
        self.model_version = model_version

    # ----------------------------------------------------------------------- loading

    def load(self) -> None:
        if self._session is not None and self._tokenizer is not None:
            return
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("the local model layer needs: pip install havelai[local]") from exc
        cfg = self.config
        kwargs = {"revision": cfg.revision, "cache_dir": cfg.cache_dir, "token": cfg.hf_token}
        filename = FILES[cfg.precision]
        logger.info(
            "havelai: loading model %s (%s, %s) from Hugging Face; first use downloads it",
            cfg.repo_id,
            cfg.precision,
            filename,
        )
        start = time.perf_counter()
        model_path = hf_hub_download(cfg.repo_id, filename, **kwargs)
        tok_path = hf_hub_download(cfg.repo_id, TOKENIZER_FILE, **kwargs)
        try:
            meta = json.loads(Path(hf_hub_download(cfg.repo_id, META_FILE, **kwargs)).read_text())
            self.labels = tuple(meta.get("labels", self.labels))
            self.max_length = int(meta.get("max_length", self.max_length))
        except Exception:  # meta file is optional; fall back to the taxonomy order
            logger.warning(
                "havelai: %s has no readable %s; using built-in labels", cfg.repo_id, META_FILE
            )
        if list(self.labels) != list(LABELS):
            raise RuntimeError(
                f"{cfg.repo_id} labels {self.labels} do not match this havelai version's taxonomy"
            )
        sha = snapshot_revision(model_path)
        size_mb = Path(model_path).stat().st_size / 1e6
        self.model_version = (
            f"{cfg.repo_id}@{(sha or cfg.revision or 'unknown')[:12]}:{cfg.precision}"
        )
        logger.info(
            "havelai: model ready: %s, %.1f MB, %.1f s",
            self.model_version,
            size_mb,
            time.perf_counter() - start,
        )
        tokenizer = Tokenizer.from_file(tok_path)
        # no truncation here: chunking needs real token counts; predict() truncates the ids
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    @property
    def tokenizer(self) -> Any:
        self.load()
        return self._tokenizer

    @property
    def session(self) -> Any:
        self.load()
        return self._session

    # ----------------------------------------------------------------------- inference

    def count_tokens(self, text: str) -> int:
        """Tokens without the special tokens the model adds around each input."""
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def chunk(self, text: str) -> Chunking:
        cfg = self.config
        return chunk_text(text, self.count_tokens, cfg.min_chunk_tokens, cfg.max_chunk_tokens)

    def predict(self, chunks: Sequence[str]) -> list[list[float]]:
        """Softmax probabilities per chunk, in input order."""
        import numpy as np

        encodings = [
            truncate_ids(e.ids, self.max_length) for e in self.tokenizer.encode_batch(list(chunks))
        ]
        out: list[list[float]] = [[] for _ in chunks]
        order = sorted(range(len(encodings)), key=lambda i: len(encodings[i]))
        bs = max(1, self.config.batch_size)
        for start in range(0, len(order), bs):
            batch = order[start : start + bs]
            width = max(len(encodings[i]) for i in batch)
            ids = np.zeros((len(batch), width), dtype=np.int64)
            mask = np.zeros((len(batch), width), dtype=np.int64)
            for row, i in enumerate(batch):
                ids[row, : len(encodings[i])] = encodings[i]
                mask[row, : len(encodings[i])] = 1
            logits = self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
            for row, i in enumerate(batch):
                out[i] = softmax(logits[row]).tolist()
        return out

    def score(self, text: str) -> LayerResult:
        start = time.perf_counter()
        chunking = self.chunk(text)
        if not chunking.chunks:
            return LayerResult(
                NAME, 0.0, False, None, time.perf_counter() - start, {"chunk_count": 0}
            )
        probs = self.predict(chunking.chunks)
        read = [read_probabilities(p, self.labels) for p in probs]
        best = max(range(len(read)), key=lambda i: read[i][0])
        p_inj, attack_type, type_prob = read[best]
        detail = {
            "model_version": self.model_version,
            "prompt_tokens": chunking.total_tokens,
            "chunk_size": chunking.chunk_tokens,
            "chunk_count": len(chunking.chunks),
            "detecting_chunk_index": best,
            "detecting_chunk_tokens": self.count_tokens(chunking.chunks[best]),
            "chunk_scores": [round(r[0], 6) for r in read],
            "attack_type_probability": type_prob,
            # chunk texts for the vector layer's learning step; never written to analytics
            "_chunks": list(chunking.chunks),
        }
        fired = p_inj >= FIRE_THRESHOLD
        return LayerResult(
            layer=NAME,
            score=p_inj,
            fired=fired,
            attack_type=attack_type if fired else AttackType.BENIGN,
            seconds=time.perf_counter() - start,
            detail=detail,
        )
