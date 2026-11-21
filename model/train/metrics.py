"""Training metrics: computation, tracking, and plot export, in one class.

`TrainingMetrics` is the only thing train.py needs. It

- computes the metrics (confusion matrix, per-class precision/recall/F1, macro-F1, the binary
  injection-vs-benign metrics),
- records what happens during training (per-batch windows, a validation pass before training and
  after every epoch),
- derives the improvement metrics: the change in validation accuracy / macro-F1 / loss from one
  evaluation to the next (the gradient of accuracy over epochs), and the slope of training
  accuracy over batches,
- and `flush(out_dir)` writes everything at the end: JSON tables (history.json,
  test_metrics.json) first, then the PNG plots.

    metrics = TrainingMetrics()
    metrics.log_baseline(val_loss, golds, preds)           # before training, epoch 0
    metrics.log_step(loss, correct, n)                     # every training batch
    metrics.log_epoch(epoch, train_loss, val_loss, golds, preds)
    metrics.set_test(golds, preds, is_injection)           # final evaluation
    metrics.flush(run_dir / "metrics")

The math is pure Python (no numpy); matplotlib is imported only inside `flush`, so the class
works without it until export. Plots are static PNGs on a light surface.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.data.taxonomy import LABELS, AttackType

BENIGN_ID = LABELS.index(AttackType.BENIGN.value)


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


class TrainingMetrics:
    # ---- chart design (see the dataviz palette: validated light-mode values) ----
    DPI = 150
    SURFACE = "#fcfcfb"
    INK = "#0b0b0b"  # titles
    INK2 = "#52514e"  # subtitles, axis titles, legend text, value labels
    MUTED = "#898781"  # tick labels
    GRID = "#e1e0d9"  # hairline gridlines
    AXIS = "#c3c2b7"  # baselines and axes
    SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (blue, orange, aqua)
    POS, NEG = "#2a78d6", "#e34948"  # diverging poles: improved / got worse
    RAMP = (  # sequential blue, steps 100 to 700
        "#cde2fb",
        "#b7d3f6",
        "#9ec5f4",
        "#86b6ef",
        "#6da7ec",
        "#5598e7",
        "#3987e5",
        "#2a78d6",
        "#256abf",
        "#1c5cab",
        "#184f95",
        "#104281",
        "#0d366b",
    )

    def __init__(
        self,
        labels: Sequence[str] = LABELS,
        benign_id: int = BENIGN_ID,
        step_window: int = 50,
    ) -> None:
        self.labels = tuple(labels)
        self.benign_id = benign_id
        self.step_window = step_window  # batches averaged into one training-curve point
        self.baseline: dict[str, Any] | None = None  # evaluation before any training (epoch 0)
        self.epochs: list[dict[str, Any]] = []  # epochs 1..N
        self.steps: list[dict[str, Any]] = []  # windowed training loss/accuracy
        self.test: dict[str, Any] | None = None
        self.best_epoch = 0
        self.best_macro_f1 = -1.0
        self._batch = 0
        self._window = [0.0, 0, 0, 0]  # loss sum, correct, examples, batches

    # ================================================================== pure math

    @staticmethod
    def confusion_matrix(golds: Sequence[int], preds: Sequence[int], n: int) -> list[list[int]]:
        """Rows are gold labels, columns are predictions."""
        cm = [[0] * n for _ in range(n)]
        for g, p in zip(golds, preds, strict=True):
            cm[g][p] += 1
        return cm

    @staticmethod
    def accuracy(cm: list[list[int]]) -> float:
        return _div(sum(cm[i][i] for i in range(len(cm))), sum(map(sum, cm)))

    def per_class(self, cm: list[list[int]]) -> dict[str, dict[str, float]]:
        """Precision, recall, F1 and support per class."""
        out: dict[str, dict[str, float]] = {}
        for i, label in enumerate(self.labels):
            tp = cm[i][i]
            support = sum(cm[i])
            predicted = sum(row[i] for row in cm)
            precision, recall = _div(tp, predicted), _div(tp, support)
            out[label] = {
                "precision": precision,
                "recall": recall,
                "f1": _div(2 * precision * recall, precision + recall),
                "support": support,
            }
        return out

    @staticmethod
    def macro_f1(per_class: dict[str, dict[str, float]]) -> float:
        """Mean F1 over the classes that have gold examples."""
        scores = [m["f1"] for m in per_class.values() if m["support"] > 0]
        return sum(scores) / len(scores) if scores else 0.0

    def binary(self, preds: Sequence[int], is_injection: Sequence[int]) -> dict[str, float]:
        """Injection vs benign. A prediction counts as injection when it is not `benign`."""
        tp = fp = fn = tn = 0
        for p, y in zip(preds, is_injection, strict=True):
            flagged = p != self.benign_id
            if flagged and y:
                tp += 1
            elif flagged and not y:
                fp += 1
            elif not flagged and y:
                fn += 1
            else:
                tn += 1
        precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
        return {
            "accuracy": _div(tp + tn, tp + fp + fn + tn),
            "precision": precision,
            "recall": recall,
            "f1": _div(2 * precision * recall, precision + recall),
            "false_positive_rate": _div(fp, fp + tn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    @staticmethod
    def local_slope(ys: Sequence[float], xs: Sequence[float], width: int = 7) -> list[float]:
        """Slope dy/dx at each point: the least-squares line through a sliding window of `width`
        points (fewer at the ends). Far less noisy than differencing neighbouring points."""
        n = len(ys)
        half = width // 2
        out: list[float] = []
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            x, y = xs[lo:hi], ys[lo:hi]
            if len(x) < 2:
                out.append(0.0)
                continue
            mx, my = sum(x) / len(x), sum(y) / len(y)
            out.append(
                _div(
                    sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)),
                    sum((a - mx) ** 2 for a in x),
                )
            )
        return out

    def evaluate(
        self, golds: Sequence[int], preds: Sequence[int], is_injection: Sequence[int]
    ) -> dict[str, Any]:
        """Full evaluation: attack-type metrics (vs Jev's labels) and binary metrics (vs the
        dataset's original label)."""
        cm = self.confusion_matrix(golds, preds, len(self.labels))
        per_class = self.per_class(cm)
        return {
            "n": len(golds),
            "attack_type": {
                "accuracy": self.accuracy(cm),
                "macro_f1": self.macro_f1(per_class),
                "per_class": per_class,
                "labels": list(self.labels),
                "confusion_matrix": cm,
                "note": "Compared with Jev's labels, not human labels.",
            },
            "binary_vs_original_label": self.binary(preds, is_injection),
        }

    # ================================================================== recording

    def log_step(self, loss: float, correct: int, n: int) -> None:
        """Record one training batch: mean loss, number correct, number of examples."""
        self._batch += 1
        w = self._window
        w[0] += loss * n
        w[1] += correct
        w[2] += n
        w[3] += 1
        if w[3] >= self.step_window:
            self._close_window()

    def _close_window(self) -> None:
        loss_sum, correct, n, _ = self._window
        if n:
            self.steps.append(
                {"batch": self._batch, "train_loss": loss_sum / n, "train_accuracy": correct / n}
            )
        self._window = [0.0, 0, 0, 0]

    def _validation_entry(
        self, epoch: int, val_loss: float, golds: Sequence[int], preds: Sequence[int]
    ) -> dict[str, Any]:
        cm = self.confusion_matrix(golds, preds, len(self.labels))
        return {
            "epoch": epoch,
            "train_loss": None,
            "val_loss": val_loss,
            "val_accuracy": self.accuracy(cm),
            "val_macro_f1": self.macro_f1(self.per_class(cm)),
            "val_loss_delta": None,
            "val_accuracy_delta": None,
            "val_macro_f1_delta": None,
            "seconds": 0.0,
            "is_best": False,
        }

    def log_baseline(self, val_loss: float, golds: Sequence[int], preds: Sequence[int]) -> dict:
        """Validation pass before any training (epoch 0). It is the starting point for the
        epoch-1 improvement and is never chosen as the best epoch."""
        self.baseline = self._validation_entry(0, val_loss, golds, preds)
        return self.baseline

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        golds: Sequence[int],
        preds: Sequence[int],
        seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Record an epoch. Returns its entry; `entry["is_best"]` says whether validation
        macro-F1 improved on every earlier epoch."""
        self._close_window()
        entry = self._validation_entry(epoch, val_loss, golds, preds)
        entry["train_loss"] = train_loss
        entry["seconds"] = round(seconds, 1)
        previous = self.epochs[-1] if self.epochs else self.baseline
        if previous is not None:
            # Improvement since the previous evaluation, i.e. d(metric)/d(epoch) with a step of 1.
            for key in ("val_loss", "val_accuracy", "val_macro_f1"):
                entry[f"{key}_delta"] = entry[key] - previous[key]
        if entry["val_macro_f1"] > self.best_macro_f1:
            self.best_macro_f1, self.best_epoch = entry["val_macro_f1"], epoch
            entry["is_best"] = True
        self.epochs.append(entry)
        return entry

    def set_test(
        self,
        golds: Sequence[int],
        preds: Sequence[int],
        is_injection: Sequence[int],
        loss: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate on the test set, keep the result for the export, and return it."""
        self.test = self.evaluate(golds, preds, is_injection)
        self.test["test_loss"] = loss
        self.test["best_epoch"] = self.best_epoch
        return self.test

    def step_accuracy_gradient(self) -> list[float]:
        """Slope of training accuracy over batches, in accuracy points per 100 batches.

        Each point is the least-squares slope through a sliding window of 7 windowed-accuracy
        points (`step_window` batches each)."""
        if len(self.steps) < 2:
            return [0.0] * len(self.steps)
        slope = self.local_slope(
            [s["train_accuracy"] for s in self.steps], [s["batch"] for s in self.steps], 7
        )
        return [g * 100 * 100 for g in slope]  # fraction per batch -> points per 100 batches

    def history(self) -> dict[str, Any]:
        """Everything recorded during training, as plain data (the table view of the plots)."""
        grads = self.step_accuracy_gradient()
        return {
            "baseline": self.baseline,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_val_macro_f1": self.best_macro_f1 if self.epochs else None,
            "step_window_batches": self.step_window,
            "steps": [
                {**s, "train_accuracy_gradient_points_per_100_batches": g}
                for s, g in zip(self.steps, grads, strict=True)
            ],
        }

    # ================================================================== export

    @staticmethod
    def require_plotting() -> None:
        """Fail early (before a long training run) if matplotlib is missing."""
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "matplotlib is needed to export the metric plots: uv sync --extra model"
            ) from exc

    def flush(self, out_dir: str | Path) -> list[Path]:
        """Write history.json, test_metrics.json (if a test set was evaluated) and the plots.

        The JSON tables are written first, so the numbers survive even if a plot fails. A plot
        is skipped when its data is missing (for example no test set). If any plot raises, the
        others still render and a RuntimeError listing the failures is raised at the end.
        Returns the files written."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        path = out / "history.json"
        path.write_text(json.dumps(self.history(), indent=2))
        written.append(path)
        if self.test is not None:
            path = out / "test_metrics.json"
            path.write_text(json.dumps(self.test, indent=2))
            written.append(path)

        self.require_plotting()
        import matplotlib

        failures: list[str] = []
        plots = (
            self._plot_loss,
            self._plot_accuracy,
            self._plot_accuracy_gradient,
            self._plot_confusion_matrix,
            self._plot_per_class,
            self._plot_binary,
        )
        with matplotlib.rc_context(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                "axes.unicode_minus": False,
            }
        ):
            for plot in plots:
                try:
                    made = plot(out)
                except Exception as exc:  # keep going so one bad plot loses only itself
                    failures.append(f"{plot.__name__}: {type(exc).__name__}: {exc}")
                else:
                    if made is not None:
                        written.append(made)
        if failures:
            raise RuntimeError("some metric plots failed: " + "; ".join(failures))
        return written

    # ---------------------------------------------------------------- plot helpers
    #
    # Layout is in inches from the top-left of the figure, so titles, axes and legends never
    # overlap whatever the figure size: header, then the axes, then a footer for tick labels
    # and the legend.

    @staticmethod
    def _pt(px: float) -> float:
        """CSS pixels to points (a 2 px line is 1.5 pt, whatever the export dpi)."""
        return px * 0.75

    @staticmethod
    def _header_in(lines: int) -> float:
        """Height reserved for a title plus a subtitle of `lines` lines."""
        return (24 + 14.5 * lines) / 72 + 0.3

    def _figure(self, width: float, height: float):
        from matplotlib.figure import Figure

        return Figure(figsize=(width, height), dpi=self.DPI, facecolor=self.SURFACE)

    @staticmethod
    def _axes(fig, left: float, top: float, width: float, height: float):
        """Axes whose box is `width` x `height` inches, `left` from the left edge and `top` from
        the top edge of the figure."""
        w, h = fig.get_size_inches()
        return fig.add_axes([left / w, 1 - (top + height) / h, width / w, height / h])

    def _head(self, fig, left: float, top: float, title: str, subtitle: str = "") -> None:
        """Title (and subtitle) with their top-left corner `left`, `top` inches from the corner."""
        w, h = fig.get_size_inches()
        fig.text(
            left / w, 1 - top / h, title, fontsize=13, fontweight="bold", color=self.INK,
            va="top", ha="left",
        )  # fmt: skip
        if subtitle:
            fig.text(
                left / w, 1 - (top + 24 / 72) / h, subtitle, fontsize=9, color=self.INK2,
                va="top", ha="left", linespacing=1.5,
            )  # fmt: skip

    def _style(self, ax, grid: str | None = "y") -> None:
        """Recessive chrome: no top/right spines, solid hairline grid, muted ticks."""
        ax.set_facecolor(self.SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(self.AXIS)
            ax.spines[side].set_linewidth(self._pt(1))
        ax.tick_params(colors=self.MUTED, labelsize=9, length=0)
        ax.grid(False)
        if grid:
            ax.grid(axis=grid, color=self.GRID, linewidth=self._pt(1), linestyle="-")
            ax.set_axisbelow(True)

    def _line(self, ax, xs, ys, color: str, label: str, marker: str = "o") -> None:
        ax.plot(
            xs, ys, color=color, label=label, marker=marker, zorder=3,
            linewidth=self._pt(2), solid_capstyle="round", solid_joinstyle="round",
            markersize=self._pt(8), markerfacecolor=color,
            markeredgecolor=self.SURFACE, markeredgewidth=self._pt(2),
        )  # fmt: skip

    def _legend(self, fig, ax, bottom_in: float = 0.12) -> None:
        """One legend for two or more series, centered under the axes."""
        w, h = fig.get_size_inches()
        pos = ax.get_position()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=((pos.x0 + pos.x1) / 2, bottom_in / h),
            ncol=len(labels), frameon=False, fontsize=9, labelcolor=self.INK2,
            handlelength=2.2, columnspacing=1.8,
        )  # fmt: skip

    def _end_labels(self, ax, items: list[tuple[float, float, str]]) -> None:
        """Direct labels at line ends. Skipped when two labels would collide (the legend
        carries identity then); stacking or nudging them would detach them from their lines."""
        lo, hi = ax.get_ylim()
        ys = sorted(y for _, y, _ in items)
        if any((b - a) / (hi - lo) < 0.07 for a, b in zip(ys, ys[1:], strict=False)):
            return
        for x, y, text in items:
            ax.annotate(
                text, (x, y), xytext=(9, 0), textcoords="offset points", va="center",
                fontsize=9, color=self.INK2,
            )  # fmt: skip

    def _epoch_axis(self, ax, first: int, last: int) -> None:
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        # Room on the right for end labels, but no tick labels beyond the last epoch.
        ax.set_xlim(first - 0.35, last + 0.35 + 0.28 * max(1, last - first))

        def label(v, _):
            if v == 0:
                return "start"
            return f"{int(v)}" if v == int(v) and v <= last else ""

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(label))
        ax.set_xlabel("epoch", color=self.INK2, fontsize=9)

    def _px_per_unit(self, ax) -> tuple[float, float]:
        """Rendered pixels per data unit along x and y (limits must already be set)."""
        fig, pos = ax.figure, ax.get_position()
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        return (
            pos.width * fig.get_figwidth() * fig.dpi / (x1 - x0),
            pos.height * fig.get_figheight() * fig.dpi / (y1 - y0),
        )

    def _column(self, ax, x: float, value: float, color: str, base: float = 0.0) -> None:
        """A column at most 24 px wide, square at the baseline, 4 px rounded at the data end."""
        from matplotlib.patches import Polygon

        if value == base:
            return
        css = ax.figure.dpi / 96  # CSS px -> rendered px
        ppx, ppy = self._px_per_unit(ax)
        half = 24 * css / ppx / 2
        rx = min(4 * css / ppx, half)
        ry = min(4 * css / ppy, abs(value - base))
        sign = 1 if value > base else -1
        pts = [(x - half, base), (x + half, base)]
        cy = value - sign * ry
        n = 8
        for k in range(n + 1):  # right-hand corner, angle 0 to 90 degrees
            a = math.pi / 2 * k / n
            pts.append((x + half - rx + rx * math.cos(a), cy + sign * ry * math.sin(a)))
        for k in range(n + 1):  # left-hand corner, 90 to 180 degrees
            a = math.pi / 2 + math.pi / 2 * k / n
            pts.append((x - half + rx + rx * math.cos(a), cy + sign * ry * math.sin(a)))
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=3))

    def _heatmap(self, ax, counts: list[list[int]], rows: list[str], cols: list[str], size: float):
        """Row-normalized one-hue heatmap with the counts printed in the cells that have any.
        Cells with no examples stay blank, and so do rows with no examples."""
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap

        arr = np.array(counts, dtype=float)
        totals = arr.sum(axis=1, keepdims=True)
        share = np.divide(arr, totals, out=np.zeros_like(arr), where=totals > 0)
        cmap = LinearSegmentedColormap.from_list("blue", list(self.RAMP))
        mesh = ax.pcolormesh(
            np.ma.masked_where(share == 0, share), cmap=cmap, vmin=0, vmax=1,
            edgecolors=self.SURFACE, linewidth=self._pt(2),
        )  # fmt: skip
        n_rows, n_cols = arr.shape
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(n_cols) + 0.5, labels=cols)
        ax.set_yticks(np.arange(n_rows) + 0.5, labels=rows)
        ax.tick_params(colors=self.MUTED, labelsize=size + 1, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        for i in range(n_rows):
            for j in range(n_cols):
                if arr[i, j] > 0:
                    ax.text(
                        j + 0.5, i + 0.5, f"{int(arr[i, j])}", ha="center", va="center",
                        fontsize=size, color="#ffffff" if share[i, j] > 0.5 else self.INK,
                    )  # fmt: skip
        return mesh

    def _save(self, fig, out: Path, name: str) -> Path:
        path = out / name
        fig.savefig(path, dpi=self.DPI, facecolor=self.SURFACE)
        return path

    def _curve_figure(self, title: str, subtitle: str):
        """Shared frame of the two per-epoch line charts: header, axes, legend footer."""
        width, axes_h, left, right = 7.4, 3.3, 0.85, 0.4
        top = 0.25
        height = top + self._header_in(subtitle.count("\n") + 1) + axes_h + 1.15
        fig = self._figure(width, height)
        self._head(fig, left, top, title, subtitle)
        ax = self._axes(
            fig, left, top + self._header_in(subtitle.count("\n") + 1), width - left - right, axes_h
        )
        self._style(ax)
        return fig, ax

    # ---------------------------------------------------------------- plots

    def _plot_loss(self, out: Path) -> Path | None:
        if not self.epochs:
            return None
        fig, ax = self._curve_figure(
            "Loss per epoch",
            "Lower is better. Training loss is class-weighted and validation loss is not,\n"
            "so the two are not on the same scale.",
        )
        train_x = [e["epoch"] for e in self.epochs]
        train_y = [e["train_loss"] for e in self.epochs]
        val = ([self.baseline] if self.baseline else []) + self.epochs
        val_x, val_y = [e["epoch"] for e in val], [e["val_loss"] for e in val]
        ax.set_ylim(0, max(train_y + val_y) * 1.12 or 1.0)
        self._epoch_axis(ax, min(val_x + train_x), max(val_x + train_x))
        self._line(ax, train_x, train_y, self.SERIES[0], "training loss", "o")
        self._line(ax, val_x, val_y, self.SERIES[1], "validation loss", "s")
        self._end_labels(
            ax, [(train_x[-1], train_y[-1], "training"), (val_x[-1], val_y[-1], "validation")]
        )
        ax.set_ylabel("cross-entropy loss", color=self.INK2, fontsize=9)
        self._legend(fig, ax)
        return self._save(fig, out, "loss.png")

    def _plot_accuracy(self, out: Path) -> Path | None:
        if not self.epochs:
            return None
        from matplotlib.ticker import PercentFormatter

        fig, ax = self._curve_figure(
            "Validation accuracy and macro-F1",
            "Higher is better. 'start' is the model before any training. Macro-F1 averages the\n"
            "F1 of every class that has validation examples, so rare classes count equally.",
        )
        rows = ([self.baseline] if self.baseline else []) + self.epochs
        xs = [e["epoch"] for e in rows]
        acc, f1 = [e["val_accuracy"] for e in rows], [e["val_macro_f1"] for e in rows]
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        self._epoch_axis(ax, min(xs), max(xs))
        self._line(ax, xs, acc, self.SERIES[0], "validation accuracy", "o")
        self._line(ax, xs, f1, self.SERIES[1], "validation macro-F1", "s")
        self._end_labels(ax, [(xs[-1], acc[-1], "accuracy"), (xs[-1], f1[-1], "macro-F1")])
        if self.best_epoch:  # label the epoch whose checkpoint was kept
            best = next(e for e in self.epochs if e["epoch"] == self.best_epoch)
            ax.annotate(
                f"best epoch {self.best_epoch}", (self.best_epoch, best["val_macro_f1"]),
                xytext=(0, -22), textcoords="offset points", ha="center", fontsize=9,
                color=self.INK2,
            )  # fmt: skip
        self._legend(fig, ax)
        return self._save(fig, out, "accuracy.png")

    def _plot_accuracy_gradient(self, out: Path) -> Path | None:
        """Improvement: how fast accuracy is changing. Top: change in validation accuracy per
        epoch. Bottom (when there are enough batches): slope of training accuracy."""
        if not self.epochs:
            return None
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        grads = self.step_accuracy_gradient()
        show_steps = len(self.steps) >= 7
        width, left, right, axes_h, top = 7.4, 0.85, 0.4, 2.35, 0.25
        head = self._header_in(2)
        panel = head + axes_h + 0.65  # header + axes + room for the x-axis label
        fig = self._figure(width, top + panel * (2 if show_steps else 1) + 0.15)

        # ---- panel A: change in validation accuracy at each epoch
        self._head(
            fig, left, top, "Improvement per epoch: change in validation accuracy",
            "Gradient of accuracy over epochs. Epoch 1 is measured against the untrained model.\n"
            "Blue: accuracy rose. Red: accuracy fell.",
        )  # fmt: skip
        ax = self._axes(fig, left, top + head, width - left - right, axes_h)
        self._style(ax)
        deltas = [(e["epoch"], (e["val_accuracy_delta"] or 0.0) * 100) for e in self.epochs]
        values = [d for _, d in deltas]
        span = (max(max(values), 0.0) - min(min(values), 0.0)) or 1.0
        ax.set_ylim(min(min(values), 0.0) - 0.25 * span, max(max(values), 0.0) + 0.25 * span)
        ax.set_xlim(0.4, len(deltas) + 0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axhline(0, color=self.AXIS, linewidth=self._pt(1), zorder=2)
        for x, v in deltas:
            self._column(ax, x, v, self.POS if v >= 0 else self.NEG)
            if len(deltas) <= 12:  # a value at every bar tip is fine for a handful of bars
                ax.annotate(
                    f"{v:+.2f}", (x, v), xytext=(0, 5 if v >= 0 else -5),
                    textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9, color=self.INK2,
                )  # fmt: skip
        ax.set_xlabel("epoch", color=self.INK2, fontsize=9)
        ax.set_ylabel("accuracy points", color=self.INK2, fontsize=9)

        # ---- panel B: slope of training accuracy over batches
        if show_steps:
            top_b = top + panel
            self._head(
                fig, left, top_b, "Training accuracy: slope per 100 batches",
                f"Least-squares slope over a sliding window of 7 points, each the accuracy of\n"
                f"{self.step_window} batches. Above zero: still learning. Near zero: plateau.",
            )  # fmt: skip
            ax2 = self._axes(fig, left, top_b + head, width - left - right, axes_h)
            self._style(ax2)
            xs = [s["batch"] for s in self.steps]
            lim = max(abs(g) for g in grads) * 1.2 or 1.0
            ax2.set_ylim(-lim, lim)
            ax2.set_xlim(xs[0], xs[-1])
            ax2.axhline(0, color=self.AXIS, linewidth=self._pt(1), zorder=2)
            ax2.plot(
                xs, grads, color=self.SERIES[0], linewidth=self._pt(2), zorder=3,
                solid_capstyle="round", solid_joinstyle="round",
            )  # fmt: skip
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
            ax2.set_xlabel("training batch", color=self.INK2, fontsize=9)
            ax2.set_ylabel("points per 100 batches", color=self.INK2, fontsize=9)
        return self._save(fig, out, "accuracy_gradient.png")

    def _plot_confusion_matrix(self, out: Path) -> Path | None:
        if self.test is None:
            return None
        from matplotlib.ticker import PercentFormatter

        at = self.test["attack_type"]
        cm, labels = at["confusion_matrix"], at["labels"]
        names = [label.replace("_", " ") for label in labels]
        rows = [f"{n} (n=0)" if sum(cm[i]) == 0 else n for i, n in enumerate(names)]
        side, left, top = 5.6, 2.45, 0.25
        head = self._header_in(2)
        fig = self._figure(left + side + 1.35, top + head + side + 2.1)
        self._head(
            fig, 0.85, top, "Attack type: confusion matrix on the test set",
            "Each row adds up to 100%. The number in a cell is a count; empty cells have none.\n"
            "A dark diagonal is good. Labels are Jev's, so this is agreement with the labeler.",
        )  # fmt: skip
        ax = self._axes(fig, left, top + head, side, side)
        mesh = self._heatmap(ax, cm, rows, names, size=8)
        ax.tick_params(axis="x", labelrotation=40)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation_mode("anchor")
        ax.set_xlabel("predicted", color=self.INK2, fontsize=9, labelpad=10)
        ax.set_ylabel("Jev's label (true class)", color=self.INK2, fontsize=9)
        cax = self._axes(fig, left + side + 0.3, top + head, 0.18, side)
        cbar = fig.colorbar(mesh, cax=cax, format=PercentFormatter(1.0, decimals=0))
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=self.MUTED, labelsize=8, length=0)
        cbar.set_label("share of the true class", color=self.INK2, fontsize=9)
        return self._save(fig, out, "confusion_matrix.png")

    def _plot_per_class(self, out: Path) -> Path | None:
        if self.test is None:
            return None
        from matplotlib.ticker import PercentFormatter

        per_class = self.test["attack_type"]["per_class"]
        present = [(c, m) for c, m in per_class.items() if m["support"] > 0]
        if not present:
            return None
        missing = [c.replace("_", " ") for c, m in per_class.items() if m["support"] == 0]
        present.sort(key=lambda cm: cm[1]["f1"], reverse=True)
        sub = "F1 is the harmonic mean of precision and recall. Compared with Jev's labels."
        if missing:
            sub += f"\nNo test examples, so not shown: {', '.join(missing)}."
        left, right, top = 3.0, 0.4, 0.25
        axes_h = 0.46 * len(present) + 0.3
        head = self._header_in(sub.count("\n") + 1)
        width = 8.6
        fig = self._figure(width, top + head + axes_h + 1.2)
        self._head(fig, 0.4, top, "Per-class precision, recall and F1 on the test set", sub)
        ax = self._axes(fig, left, top + head, width - left - right, axes_h)
        self._style(ax, grid="x")
        ys = list(range(len(present)))
        ax.set_ylim(len(present) - 0.4, -0.6)
        ax.set_xlim(0, 1.03)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_yticks(ys, labels=[f"{c.replace('_', ' ')}  (n={m['support']})" for c, m in present])
        ax.tick_params(axis="y", labelsize=9, labelcolor=self.INK2)
        for y, (_, m) in zip(ys, present, strict=True):  # thin range line between the three dots
            vals = (m["precision"], m["recall"], m["f1"])
            ax.hlines(y, min(vals), max(vals), color=self.AXIS, linewidth=self._pt(2), zorder=2)
        for key, name, marker, color in (  # distinct shapes: color is never the only channel
            ("precision", "precision", "o", self.SERIES[0]),
            ("recall", "recall", "s", self.SERIES[1]),
            ("f1", "F1", "D", self.SERIES[2]),
        ):
            ax.plot(
                [m[key] for _, m in present], ys, linestyle="none", marker=marker, label=name,
                color=color, markersize=self._pt(8 if marker != "D" else 7.5), zorder=3,
                markeredgecolor=self.SURFACE, markeredgewidth=self._pt(2),
            )  # fmt: skip
        self._legend(fig, ax)
        return self._save(fig, out, "per_class_f1.png")

    def _plot_binary(self, out: Path) -> Path | None:
        if self.test is None:
            return None
        from matplotlib.ticker import PercentFormatter

        b = self.test["binary_vs_original_label"]
        width, top, head, axes_h = 10.4, 0.25, self._header_in(1) + 0.2, 3.3
        fig = self._figure(width, top + head + axes_h + 1.05)
        self._head(
            fig, 0.85, top, "Injection vs benign on the test set",
            "A text counts as an injection when the model predicts anything other than 'benign'.\n"
            "Compared with the dataset's own label, the only ground truth.",
        )  # fmt: skip
        head = self._header_in(2) + 0.2
        # Left: the four counts. Rows are the dataset's own label.
        ax = self._axes(fig, 0.95, top + head, 4.1, axes_h)
        self._heatmap(
            ax, [[b["tn"], b["fp"]], [b["fn"], b["tp"]]],
            ["benign", "injection"], ["benign", "injection"], size=11,
        )  # fmt: skip
        ax.set_xlabel("predicted", color=self.INK2, fontsize=9, labelpad=8)
        ax.set_ylabel("dataset label", color=self.INK2, fontsize=9)
        ax.set_title(
            "Counts (each row adds up to 100%)", loc="left", fontsize=10, color=self.INK2, pad=10
        )
        # Right: the five rates, one column each (one series, so one color).
        ax = self._axes(fig, 6.1, top + head, width - 6.1 - 0.4, axes_h)
        self._style(ax)
        ax.set_ylim(0, 1.12)
        ax.set_xlim(-0.6, 4.6)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        names = ["accuracy", "precision", "recall", "F1", "false-positive\nrate (lower is better)"]
        values = [b["accuracy"], b["precision"], b["recall"], b["f1"], b["false_positive_rate"]]
        ax.set_xticks(range(5), labels=names)
        ax.tick_params(axis="x", labelsize=9, labelcolor=self.INK2)
        for x, v in enumerate(values):
            self._column(ax, x, v, self.SERIES[0])
            ax.annotate(
                f"{v:.1%}", (x, v), xytext=(0, 5), textcoords="offset points", ha="center",
                va="bottom", fontsize=9, color=self.INK2,
            )  # fmt: skip
        ax.set_title("Rates", loc="left", fontsize=10, color=self.INK2, pad=10)
        return self._save(fig, out, "binary_metrics.png")
