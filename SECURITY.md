# Security Policy

_Last updated: 2026-11-21_

## Supported versions

| Version | Supported |
|---|---|
| `main` (pre-release) | Yes |

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's **private vulnerability reporting** (Security tab → "Report a vulnerability") on this repository. Include what you found, how to reproduce it, and the impact you expect. Never include live credentials in a report.

## Security model

- **Attack text is data.** The dataset contains prompt-injection and jailbreak text. It is never executed and never given to a generative model as instructions. The labeler (Jev) is a classifier, and its instructions say to treat the text as data.
- **Third-party processing.** Positive dataset rows are sent to OpenRouter and TypeSafe (Jev) for labeling. Do not send private or customer text through `enhance_pipeline.py`.
- **No raw text in logs.** Progress output has counts and cost only. Error records hold the row id and the error, not the text. Raw text exists only in the dataset output files.
- **Secrets.** `OPENROUTER_API_KEY` is read from the environment or `.env`, sent only in the `Authorization` header, and never logged or written to outputs. `.env` is gitignored.
- **Spending control.** Runs support `--limit`, `--dry-run` and `--max-cost`. 401/402/403 abort the run instead of retrying.
- **Reproducibility.** The labeler model is pinned (`typesafe/jev-1.14`) and each row stores the dated snapshot that served it. The endpoint is alpha and may change.
- **Data and licenses.** Raw and enhanced data and checkpoints are gitignored. The dataset's license is not stated on its card; do not redistribute it or derived data without checking.
- **Detection is not a guarantee.** A detector reduces risk, it does not remove it. Do not use it as the only control for an agent with access to sensitive tools or data.

## Security-relevant changes

Record changes to data handling, secrets, dependencies with network access, and detection behavior here, newest first.

| Date | Change | Area |
|---|---|---|
| 2026-11-21 | Labeling pipeline: key from env only, no raw text in logs or errors, budget cap, fatal stop on 401/402/403 | Secrets, data handling |
| 2026-11-21 | Raw and enhanced data, checkpoints and `.env` gitignored | Data retention |
