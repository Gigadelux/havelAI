# Decisions

_Last updated: 2026-11-21_

A log of design choices, written as **"X over Y because Z"**. Add an entry whenever a design choice is made, so the reasoning survives after the code changes. Newest entries go on top.

## Entry template

### YYYY-MM-DD — Short title
- **Decision:** X
- **Over:** Y (and any other option considered)
- **Because:** Z
- **Consequences:** what this makes easier, what it makes harder, and when to revisit it

---

## Entries

### 2026-11-21 — tqdm bars on the console, logging to stderr and a log file
- **Decision:** training shows tqdm progress bars (epoch, validation, test) and logs through a `havelai.train` logger to stderr and `<run>/train.log`: start config, warnings, per-epoch lines and errors with tracebacks. Log lines go through `logging_redirect_tqdm`. A non-finite loss raises `FloatingPointError`; Ctrl+C exits 130 and still exports partial metrics.
- **Over:** `print` statements; logging every 100 steps; leaving a NaN run to finish.
- **Because:** a run can take hours, so the progress and loss trend must be visible live, and the start configuration plus any failure must survive in a file next to the run. The bar already shows the step-level numbers, so the old every-100-steps lines are gone. Training on NaN wastes the run silently.
- **Consequences:** two outputs to keep consistent (the data report is still `print`ed to stdout). The log file holds no progress bars. tqdm is MPL-2.0; fine as an unmodified dependency, not to be vendored.

### 2026-11-21 — One `TrainingMetrics` class that records during training and exports with `flush()`
- **Decision:** all metric math, tracking and plotting live in one class in `model/train/metrics.py`. `train.py` calls `log_baseline`, `log_step`, `log_epoch`, `set_test` and, at the end, `flush(run_dir / "metrics")`, which writes the JSON tables first and then the PNG plots. Improvement is reported as the change in validation accuracy per epoch (measured against an untrained epoch 0) and as the least-squares slope of windowed training accuracy over batches.
- **Over:** plotting from `train.py` directly; plotting live during training; differencing neighbouring training-accuracy points; reading the improvement from the loss only.
- **Because:** one object keeps the numbers and the plots from drifting apart and keeps `train.py` about the loop. Plots at the end need no display and cost nothing during training. A least-squares slope is much steadier than differencing noisy per-window accuracy, which a first render showed to be mostly noise. The epoch-0 pass costs one validation run and makes epoch 1's improvement meaningful.
- **Consequences:** no live view while training runs (the JSON and plots appear at the end, or on an interrupted run). The class is deliberately monolithic, so it will grow when the two-head model adds binary-head metrics and calibration; split it then if it gets unwieldy. matplotlib is a new `model` dependency. Static PNGs have no hover, so `history.json` and `test_metrics.json` are the table view of every plot.

### 2026-11-21 — mmBERT-small, 32k vocabulary, int8, two heads
- **Decision:** the model is mmBERT-small with the vocabulary pruned to 32k tokens before the single fine-tuning run, exported to int8 afterwards, with one shared encoder and two heads: a binary head (trained on all rows with the dataset's original label) and a nine-way attack-type head (trained on attack rows with Jev's probabilities as soft targets). At inference the binary head decides whether it is an attack and the type head only picks the type.
- **Over:** mmBERT-base; the full 256k vocabulary; a single 10-way softmax (kept as the comparison baseline); distillation, quantization-aware training or a second retrained model to meet the memory limit.
- **Because:** the model file must be at most 256 MB and run on laptop CPUs. Computed size is about 55 MB in int8 (base would not fit). Two heads keep the only ground truth (binary) separate from Jev's noisier type labels, and the second head costs 3,465 parameters. Pruning before fine-tuning stays within the "no second model" rule.
- **Consequences:** English-heavy token selection weakens other languages (open question). Needs a pruning tool, a two-head model class, a new loss and an ONNX int8 export, none built yet. Compare against the single-head baseline on the binary metric and revisit if the two-head model does not win. Sizes are computed, not measured; quantization accuracy loss is untested. Details in `docs/MODEL_ARCHITECTURE.md`.

### 2026-11-21 — Plain PyTorch loop, keep `other_injection`, report binary metrics first
- **Decision:** a hand-written training loop; drop only `unlabeled` and low-confidence rows (keep `other_injection`); weight the loss by the square root of balanced class weights; headline metric is injection vs benign against the original dataset label.
- **Over:** Hugging Face `Trainer`; dropping every `needs_review` row; unweighted loss; reporting attack-type accuracy as the main number.
- **Because:** the loop is small, easy to test offline, and does not depend on `Trainer` API changes between transformers versions. `needs_review` includes every `other_injection` row, so dropping it would delete that class. Rare attack types would otherwise barely affect the loss. Attack-type labels come from Jev, so only the binary label is real ground truth.
- **Consequences:** no built-in multi-GPU, resume-from-checkpoint or logging integrations (add `accelerate` or switch to `Trainer` if needed). Class weights, learning rate and the 0.6 confidence cut are untuned. Revisit after a first real run.

### 2026-11-21 — Python 3.12, requires >=3.11
- **Decision:** `.python-version` 3.12, `requires-python >=3.11`.
- **Over:** Python 3.14 (the template's choice).
- **Because:** torch and transformers wheels lag new Python releases, and the training extra must install. 3.11 is the floor because the code uses `StrEnum`.
- **Consequences:** raise the pin once the `model` extra resolves on a newer version.

### 2026-11-21 — Hatch renames `src` to `havelai`
- **Decision:** the repo folder is `src/{analytics,data,detection}` and Hatch's wheel `sources` mapping publishes it as `havelai`.
- **Over:** a literal top-level `src` package; the standard `src/havelai/...` layout.
- **Because:** the owner wants the path `src/<subpackage>` without an extra folder, and a public package named `src` collides with other projects.
- **Consequences:** two names for one tree (`src.*` in the repo, `havelai.*` for users). Hatch refuses editable installs with a renamed prefix, so `dev-mode-dirs = ["."]` is set (dev imports `src.*` from the repo root), and library code must use relative imports or the wheel breaks. Checked by building a wheel and installing it, and by `pip install -e .`. If this keeps causing friction, move to `src/havelai/`.

### 2026-11-21 — Merged attack taxonomy with an `other_injection` catch-all
- **Decision:** nine attack classes plus `benign` (see `docs/TAXONOMY.md`), defined once in `src/data/taxonomy.py`.
- **Over:** the surveys' direct/indirect split, or goal hijacking vs prompt leaking only; the full HackAPrompt ontology (29 techniques).
- **Because:** the dataset rows seen so far are short, standalone texts, so direct/indirect (a delivery route) does not separate them, while technique classes do. Surveys disagree on the jailbreak vs injection boundary, so jailbreak personas are kept as one class instead of a separate problem. The full ontology is probably too fine-grained for a single-label classifier (not tested). Jev has no abstain option, so a catch-all is needed.
- **Consequences:** labels are our own merge, not a standard. Revisit after reading the `needs_review` and `other_injection` rates on a real sample.

### 2026-11-21 — Jev over a generative LLM as labeler; positives only
- **Decision:** classify label-1 rows with `typesafe/jev-1.14` (a decision model) through OpenRouter. Label-0 rows become `benign` in code.
- **Over:** a general chat LLM with a JSON prompt; labeling all rows.
- **Because:** Jev returns a typed choice with probabilities, is very cheap ($0.042/M input tokens, output free) and fast, and gives a confidence to drive review. Benign rows need no model call.
- **Consequences:** no explanations, no abstain option, and the endpoint is alpha. A benign-labeled row that is really an attack is not caught. Pin the version and revisit if the API changes.

<!-- Add entries above this line. -->
