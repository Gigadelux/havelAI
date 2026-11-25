# Changelog

_Last updated: 2026-11-25_

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
- 2026-11-21: `model/train/train.py`: mmBERT fine-tuning (weighted loss, validation-based checkpointing, test evaluation, `--dry-run`) and `model/tests/test_train.py`.
- 2026-11-21: `requirements.txt`: pinned `model` extra for pip users (resolved with pip; regenerate with `uv export` once `uv.lock` exists).
- 2026-11-21: offline unit tests in `src/tests/` and `model/tests/`.
- 2026-11-25: `model/train/runpod_launch.py`: trains on a RunPod GPU pod over SSH, with no network volume (`up`, `push`, `train`, `status`, `logs`, `pull`, `down`, and `run` for all of it, which always terminates the pod). Pod size (RUNPOD_*) and training memory (TRAIN_*) are set in `.env`; `example.env` documents them. Dry run unless `--yes`; hourly-rate limit; training time limit. Tests in `model/tests/test_runpod_launch.py` (offline, fake SDK and ssh).
- 2026-11-25: the launcher creates its own SSH key (`gitignore/temp/runpod_ed25519`) on the first `--yes` when `RUNPOD_SSH_KEY` is empty, so `RUNPOD_API_KEY` is the only value to fill in. A key set in `RUNPOD_SSH_KEY` is never generated.
- 2026-11-25: optional RunPod network volume for download caches only (`RUNPOD_NETWORK_VOLUME_ID`, mounted at `/workspace`; uv, pip and Hugging Face caches in `/workspace/havelai-cache/`), so later pods skip the ~3.5 GB package download. Requires `RUNPOD_DATA_CENTER`. Data, venv and checkpoints stay on the container disk.
- 2026-11-25: `train.py --gradient-checkpointing` to trade compute for activation memory.
- 2026-11-25: `runpod>=2.4.0` (MIT) added in a new `cloud` dependency group. Resolving it with the rest moved `python-dateutil` from 4.0.0 to 2.9.0.post0 in `uv.lock` and `requirements.txt` (runpod's boto3/botocore needs `<3`).

### Changed
- 2026-11-21: `train.py` now shows tqdm progress bars (per epoch, validation and test; `--no-progress` to turn off) and logs through the `havelai.train` logger to stderr and `<run>/train.log`: start configuration, data warnings, per-epoch progress, and errors with tracebacks. A NaN or infinite training loss now stops the run with a clear error; an interrupted run exits 130 and still exports its partial metrics. `tqdm` is now an explicit `model` dependency.
- 2026-11-21: metrics split out of `train.py` into `model/train/metrics.py`: one class, `TrainingMetrics`, that computes the metrics, records training, and exports JSON tables plus six matplotlib plots into `<run>/metrics/` with `flush()`. New: a validation pass before training and the improvement metrics (change in validation accuracy / macro-F1 / loss per epoch, and the slope of training accuracy over batches). `matplotlib` added to the `model` extra; `uv.lock` and `requirements.txt` regenerated.
- 2026-11-21: one encoder constant, `ENCODER_MODEL` (mmBERT-small), replaces `BASE_MODEL`/`SMALL_MODEL`/`DEFAULT_ENCODER`; `train.py` and the baseline now default to mmBERT-small, since mmBERT-base cannot meet the 256 MB limit.
- 2026-11-21: `model/architecture.py` now defines the two-head model (`HavelForInjectionDetection`, `build_two_head`, `two_head_loss`, `decide`, `class_probabilities`, `type_target_from_record`); the single-head `build_classifier` is unchanged. Tests in `model/tests/test_architecture.py`. Vocabulary pruning, two-head training and the int8 export are not built yet.
- 2026-11-21: `docs/MODEL_ARCHITECTURE.md` now describes the target design (mmBERT-small, 32k vocabulary, int8, binary + attack-type heads) with diagrams, size budget and the list of code still to write; decision logged in `DECISIONS.md`.
- 2026-11-21: root `ARCHITECTURE.md` now covers only the Python library; model material moved to `docs/ENHANCEMENT_PIPELINE.md`, `docs/TRAINING.md` and `docs/MODEL_ARCHITECTURE.md`. `CLAUDE.md` and `README.md` keep short overviews that link there.

### Fixed
- 2026-11-21: relative imports in `src/` and `dev-mode-dirs` so the `src` → `havelai` rename works for built wheels and editable installs.

### Security
- 2026-11-21: key from env only, no raw text in logs, data and checkpoints gitignored.
