# Security Policy

_Last updated: 2026-11-25_

## Supported versions

| Version | Supported |
|---|---|
| `main` (pre-release) | Yes |

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's **private vulnerability reporting** (Security tab → "Report a vulnerability") on this repository. Include what you found, how to reproduce it, and the impact you expect. Never include live credentials in a report.

## Security model

- **Attack text is data.** The dataset contains prompt-injection and jailbreak text. It is never executed and never given to a generative model as instructions. The labeler (Jev) is a classifier, and its instructions say to treat the text as data.
- **Third-party processing.** Positive dataset rows are sent to OpenRouter and TypeSafe (Jev) for labeling. Do not send private or customer text through `enhance_pipeline.py`.
- **No raw text in training logs.** `model/train/train.py` logs settings, counts, progress and errors to stderr and `<run>/train.log`, never dataset text (a test checks this).
- **No raw text in logs.** Progress output has counts and cost only. Error records hold the row id and the error, not the text. Raw text exists only in the dataset output files.
- **Secrets.** `OPENROUTER_API_KEY` is read from the environment or `.env`, sent only in the `Authorization` header, and never logged or written to outputs. `.env` is gitignored.
- **RunPod training.** `RUNPOD_API_KEY` is read from `.env`, used only on this machine by the SDK, and never put in the pod's environment or the pod script. The code is uploaded with `git archive HEAD`, so `.env`, `gitignore/` and uncommitted files never leave the machine. SSH is key-only (`BatchMode=yes`); only the public key is sent, as `PUBLIC_KEY`. By default the launcher uses its own ed25519 key, `gitignore/temp/runpod_ed25519` (created on the first `--yes`, mode 600, no passphrase, gitignored, used for nothing else), so your personal keys are not involved; delete it to rotate it. Anyone with that file can log in to a running pod, so never copy it out of `gitignore/`. Host keys are pinned on first use in `gitignore/temp/runpod_known_hosts` (`StrictHostKeyChecking=accept-new`). The enhanced data is copied only to the owner's own pod (owner approved, 2026-11-25), on container disk that is destroyed with the pod. The optional network volume (`RUNPOD_NETWORK_VOLUME_ID`) holds only public download caches (uv, pip, the Hugging Face model), never dataset rows, checkpoints or secrets, so nothing derived from the dataset outlives the pod.
- **Spending control.** Runs support `--limit`, `--dry-run` and `--max-cost`. 401/402/403 abort the run instead of retrying.
- **Reproducibility.** The labeler model is pinned (`typesafe/jev-1.14`) and each row stores the dated snapshot that served it. The endpoint is alpha and may change.
- **Data and licenses.** Raw and enhanced data and checkpoints are gitignored. The dataset's license is not stated on its card; do not redistribute it or derived data without checking.
- **Detection is not a guarantee.** A detector reduces risk, it does not remove it. Do not use it as the only control for an agent with access to sensitive tools or data.

## Security-relevant changes

Record changes to data handling, secrets, dependencies with network access, and detection behavior here, newest first.

| Date | Change | Area |
|---|---|---|
| 2026-11-25 | RunPod launcher: API key stays local, `git archive HEAD` upload, key-only SSH with pinned host keys, data only on the owner's pod, `--yes` + hourly-rate and time limits, pod always terminated by `run`. New network-facing dependency `runpod` (cloud group) | Secrets, data handling, spending |
| 2026-11-21 | Labeling pipeline: key from env only, no raw text in logs or errors, budget cap, fatal stop on 401/402/403 | Secrets, data handling |
| 2026-11-21 | Raw and enhanced data, checkpoints and `.env` gitignored | Data retention |
