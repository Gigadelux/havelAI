"""Fine-tune mmBERT to classify the attack type of a text.

Input: the JSONL written by enhance_pipeline.py (model/train/data/enhanced/{train,test}.jsonl).
Output: model/train/checkpoints/<run-name>/ with `best/` (model + tokenizer), `run.json` and a
`metrics/` folder: `history.json`, `test_metrics.json` and the plots (loss, accuracy, accuracy
gradient, confusion matrix, per-class F1, binary metrics). The metrics live in metrics.py.

    uv run python -m model.train.train --dry-run                  # data report only, no torch
    uv run python -m model.train.train --epochs 1 --limit-train 2000
    uv run python -m model.train.train                            # mmBERT-small, 3 epochs

Data rules
- Rows with attack_type `unlabeled`, or an attack-type confidence below --min-confidence, are
  dropped. `other_injection` rows are kept: the pipeline flags them for review only because the
  class is a catch-all.
- Texts that appear in both splits are removed from train (exact match after whitespace
  normalization; near-duplicates are not detected).
- Validation is a stratified slice of train. If there is no test.jsonl, a slice of train is held
  out as test instead.

What the metrics mean
- Attack-type metrics compare the model with Jev's labels, not with human labels. They measure
  agreement with the labeler, not true accuracy.
- Binary metrics (injection vs benign) compare with the dataset's original `label`, which is the
  only ground truth in the data. Report these as the headline numbers.

Console output: a tqdm progress bar per epoch (loss, accuracy, learning rate) and one log line
per epoch. Logging goes to stderr and to <run>/train.log: the start configuration, data warnings,
progress per epoch and any error with its traceback. Logs never contain dataset text. Pass
--no-progress to turn the bars off (for example when output goes to a file).

The training loop is plain PyTorch (AdamW, linear warmup and decay, weighted cross-entropy,
gradient accumulation, bf16 on CUDA) so it does not depend on Trainer API details. Defaults are
common starting points and have not been tuned.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model.architecture import ENCODER_MODEL, build_classifier
from model.train.metrics import TrainingMetrics
from src.data.taxonomy import ID2LABEL, LABEL2ID, LABELS

DEFAULT_DATA_DIR = Path("model/train/data/enhanced")
DEFAULT_OUT_DIR = Path("model/train/checkpoints")
UNLABELED = "unlabeled"
MIN_EXAMPLES_WARNING = 20  # warn when a class has fewer training examples than this

logger = logging.getLogger("havelai.train")

# --------------------------------------------------------------------------- logging


def close_logging() -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def setup_logging(log_file: Path | None = None) -> None:
    """Log to stderr and, when `log_file` is given, to that file with timestamps.

    Replaces any handlers from an earlier call. Messages are configuration, counts, progress and
    errors: never dataset text."""
    close_logging()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)
    if log_file is not None:
        file = logging.FileHandler(log_file, encoding="utf-8")
        file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file)


# --------------------------------------------------------------------------- config


@dataclass
class TrainConfig:
    model: str = ENCODER_MODEL
    epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 64
    grad_accum: int = 1
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = 512
    seed: int = 13
    val_fraction: float = 0.05
    test_fraction: float = 0.10  # only used when there is no test.jsonl
    min_confidence: float = 0.6
    class_weights: str = "sqrt_inverse"  # none | sqrt_inverse
    limit_train: int | None = None


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label_id: int  # index into LABELS (attack type)
    is_injection: int  # original dataset label, 0 or 1


@dataclass
class LoadStats:
    read: int = 0
    unlabeled: int = 0
    low_confidence: int = 0
    unknown_label: int = 0
    invalid: int = 0
    duplicates: int = 0
    kept: int = 0


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by an interrupted run
            if isinstance(rec, dict):
                yield rec


def load_examples(path: Path, min_confidence: float) -> tuple[list[Example], LoadStats]:
    stats = LoadStats()
    seen: set[str] = set()
    out: list[Example] = []
    for rec in read_jsonl(path):
        stats.read += 1
        text, rid, label = rec.get("text"), rec.get("id"), rec.get("label")
        if not isinstance(text, str) or not text.strip() or not isinstance(rid, str):
            stats.invalid += 1
            continue
        if label not in (0, 1):
            stats.invalid += 1
            continue
        attack = rec.get("attack_type")
        if attack == UNLABELED:
            stats.unlabeled += 1
            continue
        if attack not in LABEL2ID:
            stats.unknown_label += 1
            continue
        conf = rec.get("attack_type_confidence")
        if isinstance(conf, int | float) and conf < min_confidence:
            stats.low_confidence += 1
            continue
        if rid in seen:
            stats.duplicates += 1
            continue
        seen.add(rid)
        out.append(Example(rid, text, LABEL2ID[attack], int(label)))
    stats.kept = len(out)
    return out, stats


def stratified_split(
    examples: Sequence[Example], fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Split into (rest, held) keeping class proportions. Deterministic for a given seed.

    A class with fewer than 2 examples stays entirely in `rest`; otherwise at least one example
    is held out and at least one stays.
    """
    by_class: dict[int, list[Example]] = {}
    for ex in sorted(examples, key=lambda e: e.id):
        by_class.setdefault(ex.label_id, []).append(ex)
    rng = random.Random(seed)
    rest: list[Example] = []
    held: list[Example] = []
    for label_id in sorted(by_class):
        group = by_class[label_id]
        rng.shuffle(group)
        n = len(group)
        k = 0 if n < 2 else min(n - 1, max(1, round(n * fraction)))
        held.extend(group[:k])
        rest.extend(group[k:])
    return rest, held


@dataclass
class Splits:
    train: list[Example]
    val: list[Example]
    test: list[Example]
    leaked_removed: int = 0
    test_source: str = "test.jsonl"


def prepare_splits(train: list[Example], test: list[Example] | None, cfg: TrainConfig) -> Splits:
    leaked = 0
    source = "test.jsonl"
    if test is None:
        train, test = stratified_split(train, cfg.test_fraction, cfg.seed)
        source = f"held out {cfg.test_fraction:.0%} of train"
    else:
        test_ids = {e.id for e in test}
        before = len(train)
        train = [e for e in train if e.id not in test_ids]
        leaked = before - len(train)
    train, val = stratified_split(train, cfg.val_fraction, cfg.seed + 1)
    if cfg.limit_train is not None and len(train) > cfg.limit_train:
        rng = random.Random(cfg.seed)
        train = sorted(train, key=lambda e: e.id)
        rng.shuffle(train)
        train = train[: cfg.limit_train]
    return Splits(train, val, test, leaked, source)


def class_counts(examples: Iterable[Example]) -> list[int]:
    counter = Counter(e.label_id for e in examples)
    return [counter.get(i, 0) for i in range(len(LABELS))]


def class_weights(counts: Sequence[int], mode: str) -> list[float]:
    """Per-class loss weights. `sqrt_inverse` is the square root of balanced weights.

    Absent classes get 1.0 (they never appear as a target, so it has no effect).
    """
    if mode == "none":
        return [1.0] * len(counts)
    if mode != "sqrt_inverse":
        raise ValueError(f"unknown class weight mode: {mode}")
    present = [c for c in counts if c > 0]
    if not present:
        return [1.0] * len(counts)
    total, k = sum(present), len(present)
    return [math.sqrt(total / (k * c)) if c > 0 else 1.0 for c in counts]


# --------------------------------------------------------------------------- torch pieces


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextDataset:
    def __init__(self, examples: Sequence[Example]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


def make_collate(tokenizer: Any, max_length: int) -> Callable[[list[Example]], dict[str, Any]]:
    """Tokenize a batch with dynamic padding. Only input_ids and attention_mask are passed on."""

    def collate(batch: list[Example]) -> dict[str, Any]:
        import torch

        enc = tokenizer(
            [e.text for e in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        out = {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}
        out["labels"] = torch.tensor([e.label_id for e in batch], dtype=torch.long)
        return out

    return collate


def predict(
    model: Any, loader: Any, device: Any, use_bf16: bool, desc: str | None = None
) -> tuple[list[int], list[int], float]:
    """Return (golds, predictions, mean unweighted loss) for a loader.

    With `desc`, a tqdm progress bar with that label is shown."""
    import torch
    import torch.nn.functional as F

    model.eval()
    golds: list[int] = []
    preds: list[int] = []
    total_loss, n = 0.0, 0
    batches = loader
    if desc:
        from tqdm.auto import tqdm

        batches = tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True)
    with torch.inference_mode():
        for batch in batches:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**inputs).logits
            total_loss += F.cross_entropy(logits.float(), labels, reduction="sum").item()
            n += labels.numel()
            preds.extend(logits.argmax(dim=-1).tolist())
            golds.extend(labels.tolist())
    return golds, preds, total_loss / max(n, 1)


def train_model(
    model: Any,
    train_ds: TextDataset,
    val_ds: TextDataset,
    collate: Callable[[list[Example]], dict[str, Any]],
    cfg: TrainConfig,
    device: Any,
    weights: Sequence[float],
    on_best: Callable[[Any], None],
    metrics: TrainingMetrics,
    log: Callable[[str], None] = logger.info,
    progress: bool = True,
) -> None:
    """Train, evaluate on the validation set each epoch, and call `on_best(model)` on improvement.

    Everything measured goes into `metrics` (see metrics.py): a validation pass before training
    (epoch 0), per-batch training stats, and a validation pass after every epoch. The best epoch
    is chosen by validation macro-F1 over attack types; epoch 0 is never the best.

    `progress` shows a tqdm bar per epoch and per validation pass; `log` receives one line per
    evaluation. Raises FloatingPointError if the training loss stops being finite, so a diverged
    run fails at once instead of training on NaN.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, generator=gen
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    no_decay = ("bias", "norm")
    decay_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not any(k in n.lower() for k in no_decay)
    ]
    plain_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and any(k in n.lower() for k in no_decay)
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": plain_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_weights = torch.tensor(list(weights), dtype=torch.float32, device=device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model.to(device)
    golds, preds, val_loss = predict(
        model, val_loader, device, use_bf16, "validation (untrained)" if progress else None
    )
    start = metrics.log_baseline(val_loss, golds, preds)
    log(
        f"before training: val loss {start['val_loss']:.4f}, val macro-F1 "
        f"{start['val_macro_f1']:.4f}, val acc {start['val_accuracy']:.4f}"
    )
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        started = time.time()
        running, seen, correct = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        with tqdm(
            train_loader,
            desc=f"epoch {epoch}/{cfg.epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not progress,
        ) as bar:
            for i, batch in enumerate(bar):
                labels = batch.pop("labels").to(device)
                inputs = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    logits = model(**inputs).logits
                loss = F.cross_entropy(logits.float(), labels, weight=loss_weights)
                batch_loss = loss.item()
                if not math.isfinite(batch_loss):
                    raise FloatingPointError(
                        f"training loss is {batch_loss} at epoch {epoch}, batch {i + 1}: "
                        "the run diverged. Lower --lr or check the data."
                    )
                (loss / cfg.grad_accum).backward()
                n_correct = int((logits.argmax(dim=-1) == labels).sum().item())
                running += batch_loss * labels.numel()
                seen += labels.numel()
                correct += n_correct
                metrics.log_step(batch_loss, n_correct, labels.numel())
                if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                bar.set_postfix(
                    loss=f"{running / seen:.4f}",
                    acc=f"{correct / seen:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.1e}",
                    refresh=False,
                )

        golds, preds, val_loss = predict(
            model, val_loader, device, use_bf16, f"validation (epoch {epoch})" if progress else None
        )
        entry = metrics.log_epoch(
            epoch, running / max(seen, 1), val_loss, golds, preds, time.time() - started
        )
        log(
            f"epoch {epoch}/{cfg.epochs}: train loss {entry['train_loss']:.4f}, val loss "
            f"{val_loss:.4f}, val macro-F1 {entry['val_macro_f1']:.4f}, val acc "
            f"{entry['val_accuracy']:.4f} ({entry['val_accuracy_delta']:+.4f} vs previous)"
        )
        if entry["is_best"]:
            on_best(model)
            log(f"  new best (epoch {epoch}), checkpoint saved")


# --------------------------------------------------------------------------- reporting + CLI


def data_report(splits: Splits, train_stats: LoadStats, test_stats: LoadStats | None) -> str:
    lines = [
        f"train file: read {train_stats.read}, kept {train_stats.kept} (dropped: "
        f"{train_stats.unlabeled} unlabeled, {train_stats.low_confidence} low confidence, "
        f"{train_stats.invalid + train_stats.unknown_label} invalid, "
        f"{train_stats.duplicates} duplicate ids)",
    ]
    if test_stats is not None:
        lines.append(
            f"test file: read {test_stats.read}, kept {test_stats.kept} (dropped: "
            f"{test_stats.unlabeled} unlabeled, {test_stats.low_confidence} low confidence, "
            f"{test_stats.invalid + test_stats.unknown_label} invalid)"
        )
    lines.append(
        f"splits: train {len(splits.train)}, val {len(splits.val)}, test {len(splits.test)} "
        f"(test source: {splits.test_source}); removed {splits.leaked_removed} train texts "
        f"that also appear in test"
    )
    train_counts = class_counts(splits.train)
    test_counts = class_counts(splits.test)
    lines.append(f"{'class':32} {'train':>8} {'test':>8}")
    for i, label in enumerate(LABELS):
        lines.append(f"{label:32} {train_counts[i]:>8} {test_counts[i]:>8}")
    return "\n".join(lines)


def data_warnings(splits: Splits) -> list[str]:
    warnings: list[str] = []
    counts = class_counts(splits.train)
    for i, label in enumerate(LABELS):
        if counts[i] == 0:
            warnings.append(
                f"class '{label}' has no training examples: the model cannot predict it"
            )
        elif counts[i] < MIN_EXAMPLES_WARNING:
            warnings.append(f"class '{label}' has only {counts[i]} training examples")
    if not splits.val:
        warnings.append("validation set is empty: the best epoch cannot be selected")
    if not splits.test:
        warnings.append("test set is empty")
    return warnings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--run-name", help="default: <model>-<UTC timestamp>")
    p.add_argument("--model", default=d.model, help="mmBERT checkpoint or a local path")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--eval-batch-size", type=int, default=d.eval_batch_size)
    p.add_argument("--grad-accum", type=int, default=d.grad_accum)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup-ratio", type=float, default=d.warmup_ratio)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--val-fraction", type=float, default=d.val_fraction)
    p.add_argument("--test-fraction", type=float, default=d.test_fraction)
    p.add_argument("--min-confidence", type=float, default=d.min_confidence)
    p.add_argument("--class-weights", choices=["none", "sqrt_inverse"], default=d.class_weights)
    p.add_argument("--limit-train", type=int, help="use at most N training rows (quick trials)")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--dry-run", action="store_true", help="print the data report and stop")
    p.add_argument("--no-progress", action="store_true", help="turn off the tqdm progress bars")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    names = TrainConfig.__dataclass_fields__
    return TrainConfig(**{k: v for k, v in vars(args).items() if k in names})


def _log_start(
    cfg: TrainConfig,
    splits: Splits,
    weights: Sequence[float],
    warnings: list[str],
    device: Any = None,
    run_dir: Path | None = None,
) -> None:
    """Log the start configuration. Counts and settings only, never dataset text."""
    logger.info("config: %s", json.dumps(asdict(cfg), sort_keys=True))
    logger.info(
        "data: train %d, val %d, test %d (test source: %s); %d train texts also in test removed",
        len(splits.train),
        len(splits.val),
        len(splits.test),
        splits.test_source,
        splits.leaked_removed,
    )
    counts = class_counts(splits.train)
    logger.info(
        "train class counts: %s",
        {label: n for label, n in zip(LABELS, counts, strict=True) if n},
    )
    logger.info(
        "class weights (%s): %s",
        cfg.class_weights,
        {label: round(w, 3) for label, w, n in zip(LABELS, weights, counts, strict=True) if n},
    )
    for message in warnings:
        logger.warning(message)
    if device is not None:
        import torch
        import transformers

        logger.info(
            "device: %s | torch %s | transformers %s | run dir: %s",
            device,
            torch.__version__,
            transformers.__version__,
            run_dir,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    try:
        return _run(args, config_from_args(args))
    finally:
        close_logging()


def _run(args: argparse.Namespace, cfg: TrainConfig) -> int:
    train_path = args.data_dir / "train.jsonl"
    test_path = args.data_dir / "test.jsonl"
    if not train_path.is_file():
        logger.error("%s not found. Run model.train.enhance_pipeline first.", train_path)
        return 2
    train_ex, train_stats = load_examples(train_path, cfg.min_confidence)
    test_ex, test_stats = (None, None)
    if test_path.is_file():
        test_ex, test_stats = load_examples(test_path, cfg.min_confidence)
    if not train_ex:
        logger.error("No usable training rows after filtering.")
        return 2

    splits = prepare_splits(train_ex, test_ex, cfg)
    print(data_report(splits, train_stats, test_stats))
    warnings = data_warnings(splits)
    weights = class_weights(class_counts(splits.train), cfg.class_weights)
    if args.dry_run:
        _log_start(cfg, splits, weights, warnings)
        print(
            "class weights:", {label: round(w, 3) for label, w in zip(LABELS, weights, strict=True)}
        )
        return 0
    if not splits.train or not splits.val:
        logger.error("Need non-empty train and validation sets to train.")
        return 2

    try:
        TrainingMetrics.require_plotting()  # fail now, not after hours of training
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    import torch  # lazy: needs the `model` extra

    set_seed(cfg.seed)
    device = pick_device(args.device)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{cfg.model.rstrip('/').split('/')[-1]}-{stamp}"
    run_dir = args.out_dir / run_name
    best_dir = run_dir / "best"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")  # from here on, everything is also written to the file
    logger.info("training run %s", run_name)
    _log_start(cfg, splits, weights, warnings, device, run_dir)

    from tqdm.contrib.logging import logging_redirect_tqdm

    tracker = TrainingMetrics()
    metrics_dir = run_dir / "metrics"
    finished = False
    with logging_redirect_tqdm(loggers=[logger]):  # log lines do not break the progress bars
        try:
            tokenizer, model = build_classifier(cfg.model)

            def save_best(m: Any) -> None:
                m.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

            collate = make_collate(tokenizer, cfg.max_length)
            train_model(
                model,
                TextDataset(splits.train),
                TextDataset(splits.val),
                collate,
                cfg,
                device,
                weights,
                save_best,
                tracker,
                progress=not args.no_progress,
            )

            # Evaluate the best epoch, reloaded from disk, on the test set.
            from torch.utils.data import DataLoader
            from transformers import AutoModelForSequenceClassification

            best = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
            test_loader = DataLoader(
                TextDataset(splits.test),
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                collate_fn=collate,
            )
            use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
            golds, preds, test_loss = predict(
                best, test_loader, device, use_bf16, None if args.no_progress else "test"
            )
            metrics = tracker.set_test(
                golds, preds, [e.is_injection for e in splits.test], test_loss
            )
            finished = True
        except KeyboardInterrupt:
            logger.warning("Interrupted. Exporting the metrics measured so far to %s", metrics_dir)
        except Exception:
            logger.exception("Training failed")
            raise
        finally:
            if not finished:  # interrupted or failed: still export what was measured so far
                try:
                    tracker.flush(metrics_dir)
                    logger.info("Partial metrics exported to %s", metrics_dir)
                except Exception:
                    logger.exception("Could not export the partial metrics")
    if not finished:
        return 130

    run_info = {
        "config": asdict(cfg),
        "device": str(device),
        "class_weights": dict(zip(LABELS, weights, strict=True)),
        "train_counts": dict(zip(LABELS, class_counts(splits.train), strict=True)),
        "test_source": splits.test_source,
        "leaked_removed": splits.leaked_removed,
        "labels": dict(ID2LABEL),
        "torch": torch.__version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))
    written = tracker.flush(metrics_dir)  # JSON tables first, then every plot

    b = metrics["binary_vs_original_label"]
    logger.info(
        "test (%d rows, best epoch %d): injection vs benign (vs dataset label): acc %.4f, "
        "precision %.4f, recall %.4f, F1 %.4f, false-positive rate %.4f",
        metrics["n"],
        tracker.best_epoch,
        b["accuracy"],
        b["precision"],
        b["recall"],
        b["f1"],
        b["false_positive_rate"],
    )
    logger.info(
        "test: attack type (vs Jev labels): acc %.4f, macro-F1 %.4f",
        metrics["attack_type"]["accuracy"],
        metrics["attack_type"]["macro_f1"],
    )
    logger.info("saved to %s; exported %d metric files to %s", run_dir, len(written), metrics_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
