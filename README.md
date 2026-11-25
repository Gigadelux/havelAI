# havelAI

_Last updated: 2026-11-25_

A small Python library for detecting and classifying **prompt-injection attacks**, plus the workspace used to train a local classifier based on [mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base).

> **Status:** pre-alpha. The taxonomy, result types and the dataset-enhancement pipeline exist. The trained detector does not yet.

## What it does

- `havelai.data`: the attack taxonomy (instruction override, jailbreak persona, prompt leaking, data exfiltration, coercion, context manipulation, obfuscation, harmful task request, other).
- `havelai.detection`: the `Detection` result and the `Detector` protocol.
- `havelai.analytics`: summaries over detections.
- `model/`: not shipped. Relabels [`jayavibhav/prompt-injection`](https://huggingface.co/datasets/jayavibhav/prompt-injection) (binary) with an attack type using the Jev decision model on OpenRouter, then trains the classifier. See [`docs/TAXONOMY.md`](docs/TAXONOMY.md).

```python
from havelai import AttackType, Detection
from havelai.analytics.summary import summarize

print(summarize([Detection("hi", AttackType.BENIGN, 0.99)]))
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
uv sync --extra model      # library + training/data tooling
cp example.env .env        # add OPENROUTER_API_KEY; never commit .env
uv run pytest
```

Without uv: `pip install -r requirements.txt` (pinned versions of the `model` extra). The library alone has no runtime dependencies. It is not published yet; build a wheel with `uv build` and install it to get `import havelai`.

## Model workspace

`model/` (not shipped) relabels the dataset and trains the classifier. Quick commands:

```bash
uv run python -m model.train.enhance_pipeline --split train --limit 200 --dry-run   # print a request, call nothing
uv run python -m model.train.train --dry-run                                         # data report, no torch
uv run python -m model.train.runpod_launch run                                       # RunPod training, dry run (needs --group cloud)
```

Training runs on a rented RunPod GPU (it ran out of memory locally on MPS). Pod size and training memory are set in `.env`; see [`docs/TRAINING.md`](docs/TRAINING.md#training-on-runpod).

Positive dataset rows are sent to OpenRouter and TypeSafe for labeling: do not use private text. Check the measured cost of a 200-row trial before a full run. See [`docs/ENHANCEMENT_PIPELINE.md`](docs/ENHANCEMENT_PIPELINE.md) and [`docs/TRAINING.md`](docs/TRAINING.md).

## Documentation

| File | What it covers |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The Python library: modules, detector protocol, packaging |
| [`docs/TAXONOMY.md`](docs/TAXONOMY.md) | Attack classes, literature, caveats |
| [`docs/ENHANCEMENT_PIPELINE.md`](docs/ENHANCEMENT_PIPELINE.md) | Relabeling the dataset with Jev |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Fine-tuning, metrics and plots, outputs |
| [`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md) | mmBERT-small with binary + attack-type heads, 32k vocabulary, int8 (model defined, training and export not built) |
| [`DECISIONS.md`](DECISIONS.md) | Design choices as "X over Y because Z" |
| [`SECURITY.md`](SECURITY.md) | Security model, reporting a vulnerability |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable changes |
| [`CLAUDE.md`](CLAUDE.md) | Project rules and conventions for Claude Code |

## Notable dependencies and licenses

New dependencies must be license-compatible with Apache 2.0. The library itself has none.

| Dependency | Purpose | License |
|---|---|---|
| httpx (`model` extra) | Calls to OpenRouter | BSD-3-Clause |
| matplotlib (`model` extra) | Metric plots | Matplotlib license (permissive, PSF-based) |
| runpod (`cloud` dependency group, not shipped) | Creates and terminates the RunPod training pod (`model/train/runpod_launch.py`). Brings its own server stack (aiohttp, fastapi, boto3, sentry-sdk...), which is why it is kept out of the `model` extra and off the pod | MIT |
| tqdm (`model` extra) | Training progress bars | MPL-2.0 and MIT (used unmodified as a dependency; MPL-2.0 is file-level copyleft, so do not vendor and edit its files) |
| datasets, transformers, torch, accelerate, scikit-learn (`model` extra) | Data loading and training | Apache-2.0 (datasets, transformers, accelerate), BSD (torch, scikit-learn) |
| mmBERT (`jhu-clsp/mmBERT-small`) | Base model | MIT |
| pytest, ruff (dev) | Tests, lint and format | MIT |

The dataset `jayavibhav/prompt-injection` shows no license on its card. Check with the author before redistributing it or anything derived from it.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
