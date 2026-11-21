"""Fine-tune mmBERT on the enhanced dataset. Not implemented yet.

Input: model/train/data/enhanced/<split>.jsonl from enhance_pipeline.py. Rows with
`needs_review` should be reviewed or dropped before training, and texts present in both splits
should be removed from train to avoid leakage.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Training is not implemented yet (see model/architecture.py).")


if __name__ == "__main__":
    main()
