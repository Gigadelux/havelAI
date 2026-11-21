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
