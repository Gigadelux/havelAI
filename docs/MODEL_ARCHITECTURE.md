# Model architecture

_Last updated: 2026-11-21_

**Status: the model is implemented, nothing is trained.** `model/architecture.py` defines the two-head model (mmBERT-small, binary + attack-type heads), the loss and the decision rule. The 32k vocabulary, the training loop for two heads and the int8 export are not built yet (see [What is built and what is not](#what-is-built-and-what-is-not)). Nothing here has been trained or measured on real data. The decision is recorded in [`../DECISIONS.md`](../DECISIONS.md).

## Requirements

- Runs on a laptop CPU on every dev machine.
- The model (weights) is at most **256 MB**.
- No second model is trained for memory reasons: the only training run is the fine-tuning itself. No distillation into a smaller student, no quantization-aware training, no pruning followed by a separate retraining. Compression is done once before fine-tuning (vocabulary) and once after (int8 export).
- Output: is it an injection, and which attack type ([`TAXONOMY.md`](TAXONOMY.md)).

## Overview

One shared encoder produces one pooled vector. Two small linear heads read it.

```mermaid
flowchart TB
    TXT["text"] --> TOK["Pruned tokenizer<br/>BPE, 32k tokens"]
    TOK --> EMB["Embedding table<br/>32k x 384"]
    EMB --> ENC["mmBERT-small encoder<br/>22 layers, hidden 384<br/>(ModernBERT)"]
    ENC --> POOL["Masked mean pooling<br/>over tokens"]
    POOL --> PROJ["Shared projection<br/>Linear 384x384, GELU, LayerNorm"]
    PROJ --> Z(["z : 384 numbers"])
    Z --> HB["Binary head<br/>Linear 384 to 1"]
    Z --> HT["Type head<br/>Linear 384 to 9"]
    HB --> PINJ["p_inj = sigmoid(logit)"]
    HT --> Q["q = softmax(logits)<br/>over the 9 attack types"]
    PINJ --> COMB["Combine"]
    Q --> COMB
    COMB --> OUT["Detection<br/>attack_type + score"]
```

The nine attack types are every class in [`src/data/taxonomy.py`](../src/data/taxonomy.py) except `benign`: `instruction_override`, `jailbreak_persona`, `prompt_leaking`, `sensitive_data_exfiltration`, `coercion_social_engineering`, `context_manipulation`, `obfuscation_encoding`, `harmful_task_request`, `other_injection`.

## How the two heads interact

The heads do not read each other's output. They interact in three places: the **shared encoder** (training), the **loss** (training) and the **combination rule** (inference).

### 1. Training: one encoder, two losses

```mermaid
flowchart LR
    ROWS["Batch of rows<br/>text, original label,<br/>Jev type probabilities"] --> ENC["Shared encoder<br/>+ projection"]
    ENC --> HB["Binary head"]
    ENC --> HT["Type head"]
    HB --> LB["Binary loss<br/>BCE vs original label<br/>ALL rows"]
    HT --> LT["Type loss<br/>soft-label cross-entropy<br/>vs Jev probabilities<br/>ATTACK rows only"]
    LB --> SUM["Total loss<br/>L = L_bin + lambda x L_type"]
    LT --> SUM
    SUM -->|"gradients from both"| ENC
```

- **Binary head** learns from every row, using the dataset's own `label`. This is the only ground truth in the data, so it is never filtered by Jev's confidence: rows whose type label was dropped (unlabeled, low confidence) still train this head.
- **Type head** learns only from rows with `label = 1`. Its target is Jev's probability distribution over the nine types (`attack_type_probs`, stored by the enhancement pipeline), not a hard label. An ambiguous row therefore pushes the head less sharply than a confident one. Benign rows are never shown to this head, so it never has to decide "benign vs attack"; that is the binary head's job.
- **Shared encoder** receives gradients from both. The binary task teaches it what an attack looks like at all; the type task teaches it to tell attacks apart. Noisy type labels can still affect the encoder, which is why `lambda` (default 1.0, untuned) exists: lower it if the binary metric drops compared with a binary-only run.
- Class weights (`sqrt_inverse`, as in [`TRAINING.md`](TRAINING.md)) apply to the type loss only, because the rare classes are all attack types.

### 2. Inference: combining the heads

```mermaid
flowchart TB
    PINJ["p_inj from binary head"] --> GATE{"p_inj >= threshold?"}
    GATE -->|no| BEN["attack_type = benign<br/>score = 1 - p_inj"]
    GATE -->|yes| ARG["attack_type = argmax of q<br/>score = p_inj x q_max"]
    Q["q from type head"] --> ARG
    BEN --> DET["Detection"]
    ARG --> DET
```

- **The binary head decides whether it is an attack.** The type head is consulted only after that.
- **The full distribution is consistent**: `P(benign) = 1 - p_inj` and `P(type k) = p_inj x q_k`, which sums to 1 over the ten classes. `Detection.score` is the probability of the reported class under this rule, so the existing field keeps its meaning ("confidence in `attack_type`") and `Detection` does not need to change. A flagged text can have a low `score` when the type is uncertain (for example `p_inj` 0.95 and `q_max` 0.4 gives 0.38). Adding `p_inj` and `q` as optional fields to `Detection` would make that visible; that is a separate library change, not done.
- **The threshold** (default 0.5) is tuned on validation data to reach a target false-positive rate. It is a deployment setting, not part of the weights.
- **Calibration** (optional): temperature scaling fits one scalar per head on the validation set (Guo et al., 2017). It does not retrain the model.

### Why this shape

| Alternative | Why not |
|---|---|
| One 10-way softmax (today's code) | Mixes trusted binary labels with noisy type labels in one loss, and confidence-filtering the type labels also throws away binary training signal. Kept as the baseline to compare against. |
| Type head also learns `benign` | Gives the noisy head a say in the binary decision. |
| Type head fed with the binary logit | Extra coupling with no evidence it helps; the combination rule already uses both. |
| Two separate models | Doubles the memory and breaks the "no second model" rule. |

## Components and size

Computed from the model configuration (22 layers, hidden size 384, intermediate size 1152, 6 attention heads), not measured from files.

| Part | Parameters |
|---|---|
| Embedding table, 32,000 x 384 | 12.29 M |
| Encoder, 22 layers | 42.19 M |
| Shared projection (dense + LayerNorm) | 147,840 |
| Binary head | 385 |
| Type head | 3,465 |
| **Total** | **about 54.6 M** |

| Format | Size |
|---|---|
| fp32 | 219 MB |
| fp16 | 109 MB |
| **int8 (target)** | **about 55 MB** |

That is about a fifth of the 256 MB limit. Both heads together add 3,850 parameters on top of the shared projection; the second head costs 3,465 parameters (14 KB in fp32).

The tokenizer file is separate and small (the crude 32k test file was 1.3 MB).

## Vocabulary pruning (32k)

mmBERT's tokenizer has 256,000 tokens and 580,604 merges; 98M of mmBERT-small's 140M parameters are the embedding table. Keeping 32,000 tokens removes about 86M parameters.

1. **Select tokens before fine-tuning**, from the training corpus: every token that appears when the corpus is tokenized with the original tokenizer, ranked by frequency, plus the special tokens and the 256 byte-fallback tokens. The selection must be closed under the BPE merges that remain, so words still segment properly.
2. **Build the pruned tokenizer**: keep only merges whose two parts and result are all kept, and renumber the ids.
3. **Slice the embedding matrix** to the kept rows (in the new order) and start fine-tuning from that. This is not a second model: the pruned model is the one model we fine-tune.
4. **Check before training**: on held-out text, compare the pruned tokenizer's token count with the original's. A large increase means the selection is wrong. A first crude test (keeping the lowest 32k ids) produced 83 tokens where the original produced 12, so id order is not a usable selection.

Consequence: tokens not in the kept set are split into smaller pieces or bytes. The training data appears to be English, so a corpus-based selection keeps English well and other languages poorly; mmBERT's multilingual advantage is reduced. If specific languages matter, add sample text in those languages to the selection step. **This language question is still open.**

## int8 export

After fine-tuning, the model is exported once, with no further training:

1. Export to ONNX with two outputs: the binary logit and the nine type logits. Pooling and the heads are part of the graph.
2. Quantize the matrix multiplications to int8 (ONNX Runtime dynamic quantization). Keep LayerNorms and the heads in fp32; together they are well under 1 MB.
3. **Embedding table:** at 32k x 384 it is 49 MB in fp32 and 12 MB in int8, so it fits the limit even if the quantizer leaves it alone. Whether ONNX Runtime quantizes it by default has not been verified here; the fallback is to quantize it ourselves (int8 with per-row scales).
4. **Evaluate the quantized model against the float one** on the binary metric (the headline) and the type metric, on held-out data. If int8 costs too much accuracy, the options that need no training are fp16 embeddings (+12 MB), per-channel quantization, or leaving the most sensitive layers in fp32. With about 200 MB of headroom, a larger format is affordable.

PyTorch is not the inference runtime: importing `torch` and `transformers` alone peaks at 231 MB of process memory on the test machine. Inference uses ONNX Runtime, `tokenizers` and `numpy`, which together measured 48 MB. That would be an optional extra of the library (for example `havelai[onnx]`); the library core stays stdlib-only, and the model file is downloaded on first use, not shipped in the wheel.

## Evaluation

- **Headline:** injection vs benign (`p_inj >= threshold`) against the dataset's original `label`: precision, recall, F1, false-positive rate.
- **Type:** accuracy and macro-F1 of the type head on rows that are truly attacks, against Jev's labels. These measure agreement with the labeler, not true accuracy.
- **End to end:** the combined 10-class confusion matrix.
- **Generalization:** the dataset looks short, templated and English-only, so its own test split will overestimate real performance. Evaluate on a second public prompt-injection dataset, and on a few hand-written attacks in the target languages.
- **Long inputs:** attacks hidden in long documents are not in the training data. The detector should split long text into windows and take the maximum `p_inj`. That belongs in `src/detection/`, not in the model.
- **Compare against the baseline:** a flat 10-way head with the same encoder, on the same data, judged on the binary metric.

## What is built and what is not

**Built, in `model/architecture.py`** (tested offline in `model/tests/test_architecture.py`):

| Piece | What it is |
|---|---|
| `HavelForInjectionDetection` | the model: `ModernBertModel` encoder, masked mean pooling (or `cls`, a config setting), shared projection (dense + GELU + LayerNorm), `binary_head` (`Linear(384, 1)`), `type_head` (`Linear(384, 9)`). Returns `binary_logit` and `type_logits`. Config is a plain `ModernBertConfig` plus `classifier_pooling`, `num_attack_types`, `type_labels`, so it loads from any mmBERT checkpoint and saves/loads with `save_pretrained`/`from_pretrained`. |
| `build_two_head(base_model, pooling)` | returns `(tokenizer, model)` on a pretrained encoder; same shape as the baseline's `build_classifier`. |
| `two_head_loss(...)` | `L_binary` on all rows plus `type_weight` x soft-label `L_type` on the masked attack rows, optional per-class weights. |
| `type_target_from_record(attack_type, probs)` | turns a row of the enhanced dataset into the type head's soft target (or None for benign/unlabeled). |
| `class_probabilities`, `decide` | the combination rule above. |
| `parameter_counts`, `estimated_size_mb` | size reporting. |
| `build_classifier` | unchanged single-head baseline, still what `train.py` uses. |

The module imports without torch; the model class is created on first use.

**Checked against the real checkpoint** (`jhu-clsp/mmBERT-small`, loaded once into the two-head class): the encoder output is identical to a plain `ModernBertModel` loaded from the same checkpoint; the shared projection is loaded from the checkpoint's masked-language-model head (as the library's own classifier does); the only newly initialized tensors are the four head tensors; the checkpoint's pooling setting is already `mean`. Full-vocabulary parameter counts: 98,304,000 embeddings + 42,189,696 encoder + 147,840 shared projection + 385 + 3,465 = 140,645,386 (about 141 MB in int8). With a 32k vocabulary the unit test reproduces the 54.63M figure above.

**Not built:**

- **Vocabulary pruning:** token selection from the corpus, the pruned tokenizer, slicing the embedding matrix, and the tokenization check. The config's special-token ids (`pad_token_id`, `bos`, `eos`, `cls`, `sep`) must be remapped to the new ids, or the model will not build.
- **Two-head training in `train.py`:** it still trains the single-head baseline on hard labels. It needs the original label and `attack_type_probs` per row, the combined loss, class weights on the type loss, and the metrics above.
- **Calibration** (temperature per head) and threshold tuning.
- **ONNX export and int8 quantization**, with the accuracy comparison.
- **Inference code in `src/detection/`** (ONNX Runtime, windowing for long text) behind an optional extra.

## Process memory (measured, for reference)

The 256 MB limit applies to the model, not the process. Measured on a MacBook (macOS, CPU), 2026-11-21:

| Component | Memory |
|---|---|
| `import torch, transformers` | 231 MB peak |
| Python + `onnxruntime` + `tokenizers` + `numpy` | 48 MB |
| mmBERT tokenizer, full 256k vocab (580k merges) | +174 MB |
| Tokenizer with a pruned vocabulary | +23 MB (32k), +43 MB (64k); from the crude test above, so only the memory figure is valid |

## About the base model

From the model card and paper (arXiv 2509.06888): encoder-only, ModernBERT architecture, Gemma 2 tokenizer (256k vocabulary), 8192-token context, MIT license. `small`: 140M parameters (42M non-embedding), hidden size 384, 22 layers. `base`: 307M (110M non-embedding), hidden size 768. Configs checked on Hugging Face on 2026-11-21.

## Untested choices

Masked mean pooling over CLS (the library default is CLS; both are one setting, compare them), `lambda` = 1.0, the 0.5 threshold, the 32k vocabulary size, and int8 for every layer. None has been tried on real data.
