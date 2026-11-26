"""`Havel.export_metrics_summary(path)`: a Markdown summary of the detection history, with plots.

Writes `path` (Markdown) and PNG plots in `<path stem>_plots/` next to it. The tables in the
Markdown are the table view of every plot. Plots need matplotlib (in the `local` and `hosted`
extras); without it the Markdown is still written, without images. Prompts are never written:
reports do not contain them.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .report import LAYERS, DetectionReport

# Validated light-mode values from the dataviz reference palette (same as model/train/metrics.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"
THRESHOLD = "#e34948"


def _pct(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _md_text(text: str, limit: int = 120) -> str:
    t = " ".join(text.split())
    t = t if len(t) <= limit else t[: limit - 1] + "…"
    return t.replace("|", "\\|").replace("`", "'")


def summary_tables(reports: Sequence[DetectionReport]) -> dict[str, object]:
    """The numbers behind the summary (also used by the tests)."""
    n = len(reports)
    injections = sum(r.is_injection for r in reports)
    ms = [r.detection_ms for r in reports]
    layer_rows = {}
    for name in LAYERS:
        enabled = [r.layers[name] for r in reports if name in r.layers and r.layers[name].enabled]
        if not enabled:
            continue
        layer_rows[name] = {
            "runs": len(enabled),
            "mean_score": statistics.fmean(x.score for x in enabled),
            "mean_weight": statistics.fmean(x.weight for x in enabled),
            "mean_contribution": statistics.fmean(x.contribution for x in enabled),
            "fire_rate": sum(x.fired for x in enabled) / len(enabled),
            "mean_ms": 1000 * statistics.fmean(x.seconds for x in enabled),
        }
    matched = Counter(m.text for r in reports for m in r.vector_top)
    best_sim: dict[str, float] = {}
    for r in reports:
        for m in r.vector_top:
            best_sim[m.text] = max(best_sim.get(m.text, -1.0), m.similarity)
    return {
        "total": n,
        "injections": injections,
        "injection_rate": injections / n if n else 0.0,
        "by_attack_type": Counter(r.attack_type.value for r in reports),
        "detected_by": Counter(layer for r in reports for layer in r.detected_by),
        "ms": {
            "mean": statistics.fmean(ms) if ms else 0.0,
            "p50": _pct(ms, 0.5),
            "p95": _pct(ms, 0.95),
            "max": max(ms, default=0.0),
        },
        "layers": layer_rows,
        "chunk_sizes": Counter(r.chunk_size for r in reports if r.chunk_size),
        "prompt_tokens": [r.prompt_tokens for r in reports if r.prompt_tokens is not None],
        "learned": sum(r.learned_chunks for r in reports),
        "vectors_in_store_last": next(
            (r.vectors_in_store for r in reversed(reports) if r.vectors_in_store is not None), None
        ),
        "top_matches": [(t, c, best_sim[t]) for t, c in matched.most_common(10)],
        "versions": sorted(
            {
                (
                    r.mode,
                    r.library_version,
                    r.local_model_version or "",
                    r.hosted_model or "",
                    r.hosted_endpoint or "",
                    r.embedder_model or "",
                    r.embedder_dimensions or 0,
                    r.db_type or "",
                )
                for r in reports
            }
        ),
        "thresholds": sorted(
            {(r.detect_threshold, r.catch_threshold, r.learn_threshold) for r in reports},
            key=str,
        ),
    }


# --------------------------------------------------------------------------- plots


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _hist(plt, values, path: Path, title: str, xlabel: str, line: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    ax.hist(
        values,
        bins=min(30, max(5, len(values) // 3 or 1)),
        color=SERIES_1,
        edgecolor=SURFACE,
        linewidth=2,
    )
    if line is not None:
        ax.axvline(line, color=THRESHOLD, linewidth=2, linestyle="--")
        ax.text(
            line, ax.get_ylim()[1] * 0.95, f"  threshold {line:g}", color=INK2, va="top", fontsize=9
        )
    _style(ax, title, xlabel, "detections")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _hbar(plt, labels, values, path: Path, title: str, xlabel: str, fmt: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 1.4), facecolor=SURFACE)
    y = range(len(labels))
    ax.barh(list(y), values, color=SERIES_1, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v, i, " " + format(v, fmt), va="center", color=INK2, fontsize=9)
    _style(ax, title, xlabel, "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(0, max(values) * 1.18 if values and max(values) > 0 else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plots(reports: Sequence[DetectionReport], tables: dict, out: Path) -> dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    out.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}
    threshold = reports[-1].detect_threshold
    plots["scores"] = _hist(
        plt,
        [r.score for r in reports],
        out / "final_scores.png",
        "Final score per detection",
        "final score",
        threshold,
    )
    plots["latency"] = _hist(
        plt,
        [r.detection_ms for r in reports],
        out / "detection_ms.png",
        "Detection time",
        "milliseconds",
    )
    by_type = tables["by_attack_type"].most_common()
    plots["types"] = _hbar(
        plt, [t for t, _ in by_type], [c for _, c in by_type], out / "attack_types.png",
        "Detections by attack type", "detections", "d",
    )  # fmt: skip
    layers = tables["layers"]
    if layers:
        plots["contributions"] = _hbar(
            plt, list(layers), [v["mean_contribution"] for v in layers.values()], out / "layer_contributions.png",
            "Mean contribution to the final score", "weight × score", ".3f",
        )  # fmt: skip
    if tables["prompt_tokens"]:
        plots["tokens"] = _hist(
            plt, tables["prompt_tokens"], out / "prompt_tokens.png", "Prompt length", "tokens"
        )
    return plots


# --------------------------------------------------------------------------- markdown


def export_summary(reports: Sequence[DetectionReport], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not reports:
        path.write_text("# havelAI detection summary\n\nNo detections recorded yet.\n")
        return path
    t = summary_tables(reports)
    plots = _plots(reports, t, path.parent / f"{path.stem}_plots")

    def img(key: str, alt: str) -> str:
        p = plots.get(key)
        return f"\n![{alt}]({p.relative_to(path.parent).as_posix()})\n" if p else ""

    first, last = reports[0].created_at, reports[-1].created_at
    lines = [
        "# 🛡️ havelAI detection summary",
        "",
        f"{t['total']:,} detections from {first} to {last}.",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Detections | {t['total']:,} |",
        f"| Flagged as injection | {t['injections']:,} ({t['injection_rate']:.1%}) |",
        f"| Detection time, mean / p50 / p95 / max | {t['ms']['mean']:.1f} / {t['ms']['p50']:.1f} / "
        f"{t['ms']['p95']:.1f} / {t['ms']['max']:.1f} ms |",
        f"| Chunks learned by the vector layer | {t['learned']:,} |",
        f"| Vectors in store (last detection) | {t['vectors_in_store_last'] if t['vectors_in_store_last'] is not None else '–'} |",
        "",
        img("scores", "Final score distribution"),
        "## Attack types",
        "",
        "| Attack type | Detections |",
        "|---|---|",
        *[f"| `{k}` | {v:,} |" for k, v in t["by_attack_type"].most_common()],
        img("types", "Detections by attack type"),
        "## Layers",
        "",
        "| Layer | Runs | Mean score | Mean weight | Mean contribution | Fired | Mean time |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {name} | {v['runs']:,} | {v['mean_score']:.3f} | {v['mean_weight']:.3f} | "
            f"{v['mean_contribution']:.3f} | {v['fire_rate']:.1%} | {v['mean_ms']:.1f} ms |"
            for name, v in t["layers"].items()
        ],
        img("contributions", "Mean contribution per layer"),
        img("latency", "Detection time distribution"),
        "## Prompts and chunking",
        "",
        "| Target chunk size (tokens) | Detections |",
        "|---|---|",
        *[f"| {k} | {v:,} |" for k, v in sorted(t["chunk_sizes"].items())],
        img("tokens", "Prompt length distribution"),
    ]
    if t["top_matches"]:
        lines += [
            "## Most matched vector-store texts",
            "",
            "From your own vector store, most frequently in a detection's top 10.",
            "",
            "| Text | Times in top 10 | Best similarity |",
            "|---|---|---|",
            *[
                f"| {_md_text(text)} | {count:,} | {sim:.3f} |"
                for text, count, sim in t["top_matches"]
            ],
            "",
        ]
    lines += [
        "## Thresholds",
        "",
        "| Detect | Vector catch | Vector learn |",
        "|---|---|---|",
        *[
            f"| {d} | {c if c is not None else '–'} | {lt if lt is not None else '–'} |"
            for d, c, lt in t["thresholds"]
        ],
        "",
        "## Versions",
        "",
        "| Mode | havelAI | Local model | Hosted model | Hosted endpoint | Embedder | Dimensions | Vector db |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {' | '.join(str(x) if x else '–' for x in row)} |" for row in t["versions"]],
        "",
    ]
    path.write_text("\n".join(lines))
    return path
