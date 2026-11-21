import json
import random
from pathlib import Path

import pytest

from model.train.metrics import TrainingMetrics
from src.data.taxonomy import LABEL2ID, LABELS

N = len(LABELS)
BENIGN, LEAK = LABEL2ID["benign"], LABEL2ID["prompt_leaking"]


# ---- math (moved from train.py, same assertions)


def test_confusion_and_per_class_metrics():
    m = TrainingMetrics()
    golds = [BENIGN, BENIGN, LEAK, LEAK, LEAK]
    preds = [BENIGN, LEAK, LEAK, LEAK, BENIGN]
    cm = m.confusion_matrix(golds, preds, N)
    assert cm[BENIGN][BENIGN] == 1 and cm[BENIGN][LEAK] == 1 and cm[LEAK][LEAK] == 2
    pc = m.per_class(cm)
    assert pc["prompt_leaking"]["precision"] == pytest.approx(2 / 3)
    assert pc["prompt_leaking"]["recall"] == pytest.approx(2 / 3)
    assert pc["prompt_leaking"]["support"] == 3
    assert pc["jailbreak_persona"]["support"] == 0 and pc["jailbreak_persona"]["f1"] == 0.0
    # macro-F1 averages only the classes with gold examples (benign and prompt_leaking)
    assert m.macro_f1(pc) == pytest.approx((pc["benign"]["f1"] + pc["prompt_leaking"]["f1"]) / 2)
    assert m.accuracy(cm) == pytest.approx(3 / 5)


def test_f1_is_the_harmonic_mean_of_precision_and_recall():
    # TP=90, FP=10, FN=30: precision 0.9, recall 0.75, F1 = 2TP / (2TP + FP + FN) = 180/220
    m = TrainingMetrics()
    cm = [[0] * N for _ in range(N)]
    cm[LEAK][LEAK], cm[BENIGN][LEAK], cm[LEAK][BENIGN] = 90, 10, 30
    pc = m.per_class(cm)["prompt_leaking"]
    assert (pc["precision"], pc["recall"]) == (pytest.approx(0.9), pytest.approx(0.75))
    assert pc["f1"] == pytest.approx(180 / 220)
    assert pc["f1"] == pytest.approx(2 * 0.9 * 0.75 / (0.9 + 0.75))
    assert pc["f1"] < (0.9 + 0.75) / 2  # the harmonic mean is below the arithmetic mean


def test_binary_metrics_use_original_label():
    preds = [LEAK, LEAK, BENIGN, BENIGN]
    truth = [1, 0, 1, 0]
    b = TrainingMetrics().binary(preds, truth)
    assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (1, 1, 1, 1)
    assert b["precision"] == b["recall"] == b["false_positive_rate"] == 0.5


def test_binary_metrics_empty_is_zero_not_error():
    b = TrainingMetrics().binary([], [])
    assert b["accuracy"] == 0.0 and b["f1"] == 0.0


def test_evaluate_shape():
    out = TrainingMetrics().evaluate([0, 1], [0, 1], [0, 1])
    assert out["n"] == 2 and out["attack_type"]["accuracy"] == 1.0
    assert "Jev" in out["attack_type"]["note"]
    json.dumps(out)


# ---- slopes


def test_local_slope_recovers_a_line():
    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert TrainingMetrics.local_slope([2 * x + 1 for x in xs], xs) == pytest.approx([2.0] * 10)
    assert TrainingMetrics.local_slope([5.0] * 10, xs) == pytest.approx([0.0] * 10)


def test_local_slope_handles_uneven_spacing_and_tiny_inputs():
    xs = [0, 1, 3, 7, 8]
    assert TrainingMetrics.local_slope([0.5 * x for x in xs], xs) == pytest.approx([0.5] * 5)
    assert TrainingMetrics.local_slope([1.0], [0]) == [0.0]
    assert TrainingMetrics.local_slope([], []) == []
    assert TrainingMetrics.local_slope([1.0, 2.0], [3, 3]) == [0.0, 0.0]  # no spread in x


def test_local_slope_is_steadier_than_neighbour_differences():
    rng = random.Random(0)
    xs = list(range(60))
    ys = [0.01 * x + rng.gauss(0, 0.02) for x in xs]  # true slope 0.01 with noise
    smooth = TrainingMetrics.local_slope(ys, xs, 7)
    raw = [b - a for a, b in zip(ys, ys[1:], strict=False)]
    spread = lambda v: max(v) - min(v)  # noqa: E731
    assert spread(smooth) < spread(raw) / 2


# ---- recording


def test_log_step_windows_and_the_epoch_flush():
    m = TrainingMetrics(step_window=3)
    for loss, correct in [(1.0, 2), (2.0, 4), (3.0, 6), (4.0, 8), (5.0, 10), (6.0, 10), (7.0, 10)]:
        m.log_step(loss, correct, 10)
    assert [s["batch"] for s in m.steps] == [3, 6]  # a window closes every 3 batches
    assert m.steps[0]["train_loss"] == pytest.approx(2.0)
    assert m.steps[0]["train_accuracy"] == pytest.approx(0.4)
    m.log_epoch(1, 1.0, 1.0, [0, 1], [0, 1])  # the open window (1 batch) closes with the epoch
    assert [s["batch"] for s in m.steps] == [3, 6, 7]
    assert m.steps[2]["train_loss"] == pytest.approx(7.0)


def test_log_step_weights_by_batch_size():
    m = TrainingMetrics(step_window=2)
    m.log_step(1.0, 10, 10)
    m.log_step(4.0, 0, 30)
    assert m.steps[0]["train_loss"] == pytest.approx((1 * 10 + 4 * 30) / 40)
    assert m.steps[0]["train_accuracy"] == pytest.approx(10 / 40)


def _val(acc, n=20):
    """Gold/prediction lists with the given accuracy over two classes."""
    golds = [BENIGN] * (n // 2) + [LEAK] * (n // 2)
    k = round(acc * n)
    preds = list(golds)
    for i in range(n - k):  # flip (n - k) predictions
        preds[i] = LEAK if preds[i] == BENIGN else BENIGN
    return golds, preds


def test_improvement_is_the_change_since_the_previous_evaluation():
    m = TrainingMetrics()
    m.log_baseline(2.0, *_val(0.5))
    e1 = m.log_epoch(1, 1.5, 1.4, *_val(0.75))
    e2 = m.log_epoch(2, 1.0, 1.0, *_val(0.70))
    assert e1["val_accuracy_delta"] == pytest.approx(0.25)  # epoch 1 vs the untrained model
    assert e2["val_accuracy_delta"] == pytest.approx(-0.05)
    assert e1["val_loss_delta"] == pytest.approx(-0.6)
    assert e1["val_macro_f1_delta"] is not None


def test_without_a_baseline_the_first_delta_is_unknown():
    m = TrainingMetrics()
    e1 = m.log_epoch(1, 1.0, 1.0, *_val(0.8))
    assert e1["val_accuracy_delta"] is None
    assert m.log_epoch(2, 1.0, 1.0, *_val(0.9))["val_accuracy_delta"] == pytest.approx(0.1)


def test_best_epoch_follows_macro_f1_and_never_picks_the_baseline():
    m = TrainingMetrics()
    base = m.log_baseline(2.0, *_val(1.0))  # a perfect untrained model must still not win
    assert base["is_best"] is False and m.best_epoch == 0
    flags = [
        m.log_epoch(i + 1, 1.0, 1.0, *_val(a))["is_best"] for i, a in enumerate((0.6, 0.8, 0.7))
    ]
    assert flags == [True, True, False] and m.best_epoch == 2
    assert m.best_macro_f1 == pytest.approx(m.epochs[1]["val_macro_f1"])


def test_step_accuracy_gradient_is_points_per_100_batches():
    m = TrainingMetrics(step_window=1)
    for k in range(1, 11):  # accuracy 0.1, 0.2, ... 1.0 over batches 1..10: +0.1 per batch
        m.log_step(1.0, k, 10)
    assert m.step_accuracy_gradient() == pytest.approx([0.1 * 100 * 100] * 10)
    assert TrainingMetrics().step_accuracy_gradient() == []


def test_history_is_plain_json_and_matches_the_steps():
    m = TrainingMetrics(step_window=2)
    m.log_baseline(2.0, *_val(0.5))
    for _ in range(6):
        m.log_step(1.0, 5, 10)
    m.log_epoch(1, 1.0, 1.0, *_val(0.8))
    h = json.loads(json.dumps(m.history()))
    assert h["baseline"]["epoch"] == 0 and h["best_epoch"] == 1
    assert len(h["steps"]) == 3
    assert all("train_accuracy_gradient_points_per_100_batches" in s for s in h["steps"])


def test_set_test_stores_the_evaluation():
    m = TrainingMetrics()
    m.log_epoch(1, 1.0, 1.0, *_val(0.8))
    out = m.set_test([BENIGN, LEAK], [BENIGN, LEAK], [0, 1], loss=0.3)
    assert m.test is out and out["test_loss"] == 0.3 and out["best_epoch"] == 1


# ---- export (needs matplotlib)


def _trained(with_test=True, batches=60):
    m = TrainingMetrics(step_window=5)
    rng = random.Random(1)
    sizes = [30, 12, 0, 8, 10, 0, 0, 0, 5, 3]  # some classes absent, as in real data
    golds = [i for i, s in enumerate(sizes) for _ in range(s)]

    def preds(acc):
        return [g if rng.random() < acc else rng.randrange(N) for g in golds]

    m.log_baseline(2.3, golds, preds(0.1))
    for epoch, acc in enumerate((0.6, 0.8, 0.85), 1):
        for _ in range(batches // 3):
            m.log_step(1.0 / epoch, rng.randint(4, 14), 16)
        m.log_epoch(epoch, 1.2 / epoch, 1.0 / epoch, golds, preds(acc), seconds=1)
    if with_test:
        tg = golds + [8] * 4
        tp = [g if rng.random() < 0.85 else rng.randrange(N) for g in tg]
        m.set_test(tg, tp, [0 if g == 0 else 1 for g in tg], loss=0.4)
    return m


PLOTS = {
    "loss.png",
    "accuracy.png",
    "accuracy_gradient.png",
    "confusion_matrix.png",
    "per_class_f1.png",
    "binary_metrics.png",
}


def test_flush_writes_tables_and_every_plot(tmp_path: Path):
    pytest.importorskip("matplotlib")
    written = {p.name for p in _trained().flush(tmp_path / "metrics")}
    assert written == PLOTS | {"history.json", "test_metrics.json"}
    for name in PLOTS:
        data = (tmp_path / "metrics" / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5_000
    json.loads((tmp_path / "metrics" / "history.json").read_text())
    assert "binary_vs_original_label" in json.loads(
        (tmp_path / "metrics" / "test_metrics.json").read_text()
    )


def test_plots_have_the_light_surface_background(tmp_path: Path):
    mpimg = pytest.importorskip("matplotlib.image")
    _trained().flush(tmp_path)
    img = mpimg.imread(tmp_path / "loss.png")
    assert img.shape[0] > 400 and img.shape[1] > 800
    assert img[2, 2, :3] == pytest.approx([0xFC / 255, 0xFC / 255, 0xFB / 255], abs=0.01)


def test_flush_without_a_test_set_skips_the_test_plots(tmp_path: Path):
    pytest.importorskip("matplotlib")
    written = {p.name for p in _trained(with_test=False).flush(tmp_path)}
    assert written == {"history.json", "loss.png", "accuracy.png", "accuracy_gradient.png"}


def test_flush_before_any_epoch_writes_only_the_table(tmp_path: Path):
    pytest.importorskip("matplotlib")
    assert [p.name for p in TrainingMetrics().flush(tmp_path)] == ["history.json"]


def test_gradient_plot_adds_the_batch_panel_only_with_enough_batches(tmp_path: Path):
    mpimg = pytest.importorskip("matplotlib.image")
    few, many = tmp_path / "few", tmp_path / "many"
    _trained(batches=6).flush(few)
    _trained(batches=60).flush(many)
    assert (
        mpimg.imread(many / "accuracy_gradient.png").shape[0]
        > 1.5 * (mpimg.imread(few / "accuracy_gradient.png").shape[0])
    )


def test_tables_are_written_even_if_plotting_is_unavailable(tmp_path: Path, monkeypatch):
    m = _trained()

    def no_matplotlib():
        raise RuntimeError("matplotlib is needed")

    monkeypatch.setattr(TrainingMetrics, "require_plotting", staticmethod(no_matplotlib))
    with pytest.raises(RuntimeError, match="matplotlib"):
        m.flush(tmp_path)
    assert (tmp_path / "history.json").exists() and (tmp_path / "test_metrics.json").exists()


def test_one_failing_plot_does_not_stop_the_others(tmp_path: Path, monkeypatch):
    pytest.importorskip("matplotlib")
    m = _trained()

    def _plot_loss(self, out):  # same name as the method it replaces: errors are reported by name
        raise ValueError("bad data")

    monkeypatch.setattr(TrainingMetrics, "_plot_loss", _plot_loss)
    with pytest.raises(RuntimeError, match="_plot_loss.*bad data"):
        m.flush(tmp_path)
    assert {p.name for p in tmp_path.glob("*.png")} == PLOTS - {"loss.png"}


def test_columns_respect_the_24px_cap(tmp_path: Path):
    pytest.importorskip("matplotlib")
    m = TrainingMetrics()
    fig = m._figure(7.4, 4.0)
    ax = m._axes(fig, 0.85, 0.25, 6.0, 3.0)
    ax.set_xlim(0.4, 3.6)
    ax.set_ylim(-10, 10)
    m._column(ax, 1, 8.0, m.POS)
    m._column(ax, 2, -4.0, m.NEG)
    patch = ax.patches[0]
    xs = [x for x, _ in patch.get_xy()]
    ppx, _ = m._px_per_unit(ax)
    width_css_px = (max(xs) - min(xs)) * ppx / (fig.dpi / 96)
    assert width_css_px == pytest.approx(24, abs=0.5)
    ys = [y for _, y in ax.patches[1].get_xy()]
    assert min(ys) == pytest.approx(-4.0) and max(ys) == pytest.approx(
        0.0
    )  # grows from the baseline
