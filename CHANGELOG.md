# Changelog

_Last updated: 2026-11-21_

All notable changes to this project are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

Add one line per meaningful change under `[Unreleased]`. Security-relevant changes are also noted in `SECURITY.md`.

## [Unreleased]

### Added
- 2026-11-21: project scaffold from the mrkter AI template: uv project, `src/{analytics,data,detection}` published as `havelai`, `model/` workspace, docs.
- 2026-11-21: attack taxonomy (`src/data/taxonomy.py`) and `docs/TAXONOMY.md`.
- 2026-11-21: `Detection` and `Detector` (`src/detection/base.py`), `summarize` (`src/analytics/summary.py`).
- 2026-11-21: `model/train/enhance_pipeline.py`: adds an attack type to `jayavibhav/prompt-injection` with Jev via OpenRouter; resumable, budgeted, with `--dry-run` and `--stats`.
- 2026-11-21: placeholders `model/architecture.py` (mmBERT classifier) and `model/train/train.py`.
- 2026-11-21: `requirements.txt`: pinned `model` extra for pip users (resolved with pip; regenerate with `uv export` once `uv.lock` exists).
- 2026-11-21: offline unit tests in `src/tests/` and `model/tests/`.

### Fixed
- 2026-11-21: relative imports in `src/` and `dev-mode-dirs` so the `src` → `havelai` rename works for built wheels and editable installs.

### Security
- 2026-11-21: key from env only, no raw text in logs, data and checkpoints gitignored.
