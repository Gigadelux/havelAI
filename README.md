# havelAI

_Last updated: 2026-11-21_

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

## Enhance the dataset

```bash
uv run python -m model.train.enhance_pipeline --split train --limit 200 --dry-run   # print a request, call nothing
uv run python -m model.train.enhance_pipeline --split train --limit 200             # small paid trial
uv run python -m model.train.enhance_pipeline --stats                               # class counts, review rate, cost
```

Only label-1 rows are sent to Jev, so the cost scales with the positives. Check the measured cost of the 200-row trial before a full run, and pass `--max-cost` when you do one. Output is append-only JSONL in `model/train/data/enhanced/` and a stopped run resumes where it left off. Positive rows are sent to OpenRouter and TypeSafe: do not use private text.

## Documentation

| File | What it covers |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Layout, data flow, pipeline steps |
| [`docs/TAXONOMY.md`](docs/TAXONOMY.md) | Attack classes, literature, caveats |
| [`DECISIONS.md`](DECISIONS.md) | Design choices as "X over Y because Z" |
| [`SECURITY.md`](SECURITY.md) | Security model, reporting a vulnerability |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable changes |
| [`CLAUDE.md`](CLAUDE.md) | Project rules and conventions for Claude Code |

## Notable dependencies and licenses

New dependencies must be license-compatible with Apache 2.0. The library itself has none.

| Dependency | Purpose | License |
|---|---|---|
| httpx (`model` extra) | Calls to OpenRouter | BSD-3-Clause |
| datasets, transformers, torch, accelerate, scikit-learn (`model` extra) | Data loading and training | Apache-2.0 (datasets, transformers, accelerate), BSD (torch, scikit-learn) |
| mmBERT (`jhu-clsp/mmBERT-base`) | Base model | MIT |
| pytest, ruff (dev) | Tests, lint and format | MIT |

The dataset `jayavibhav/prompt-injection` shows no license on its card. Check with the author before redistributing it or anything derived from it.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
