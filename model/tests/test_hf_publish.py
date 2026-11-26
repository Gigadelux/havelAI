import json
from pathlib import Path

import pytest

from model.export import hf_publish as hp
from src.data.taxonomy import LABELS


def _test_metrics():
    per_class = {
        label: {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 10} for label in LABELS
    }
    return {
        "n": 100,
        "binary_vs_original_label": {
            "accuracy": 0.99,
            "precision": 0.98,
            "recall": 0.97,
            "f1": 0.975,
            "false_positive_rate": 0.01,
            "tp": 30,
            "fp": 1,
            "fn": 1,
            "tn": 68,
        },
        "attack_type": {
            "accuracy": 0.95,
            "macro_f1": 0.9,
            "labels": list(LABELS),
            "per_class": per_class,
        },
    }


def _run_info():
    return {
        "config": {
            "max_length": 512,
            "min_confidence": 0.6,
            "epochs": 3,
            "batch_size": 16,
            "lr": 3e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "class_weights": "sqrt_inverse",
        },
        "test_source": "held out 10% of train",
        "device": "cuda",
        "torch": "2.15.0",
    }


def _verify():
    model = {
        "size_bytes": 140_000_000,
        "seconds": 1.0,
        "binary_vs_original_label": {"accuracy": 0.991, "f1": 0.981, "false_positive_rate": 0.002},
        "attack_type_accuracy": 0.95,
        "attack_type_macro_f1": 0.88,
    }
    return {
        "n": 100,
        "models": {hp.ONNX_FP32: model, hp.ONNX_INT8: model},
        "int8_vs_fp32_prediction_agreement": 0.995,
    }


def test_card_contains_measured_numbers_and_warnings():
    card = hp.render_card(
        "run-x", _test_metrics(), _run_info(), _verify(), {hp.ONNX_INT8: 141_000_000}
    )
    assert card.startswith("---\nlicense: apache-2.0")
    assert "0.9750" in card and "0.0100" in card  # binary F1 and FPR from the arguments
    assert "agreement with the labels of an automatic labeler" in card
    assert "no license on its card" in card
    assert "data-augmentation" in card
    assert "99.50%" in card  # int8 vs fp32 agreement
    for label in LABELS:
        assert f"`{label}`" in card


def test_card_without_verify_says_so():
    card = hp.render_card("run-x", _test_metrics(), _run_info(), None, {})
    assert "have not been checked" in card


def test_meta_lists_labels_and_files():
    meta = hp.build_meta("run-x", _run_info())
    assert meta["labels"] == list(LABELS)
    assert meta["benign_label"] == "benign"
    assert meta["onnx"]["int8"] == hp.ONNX_INT8


def test_length_batches_cover_every_index_once():
    batches = hp.length_batches([5, 1, 3, 2, 4], 2)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == [0, 1, 2, 3, 4]
    assert batches[0] == [1, 3]  # shortest first


def test_predict_onnx_pads_and_keeps_order():
    import numpy as np

    class FakeSession:
        def run(self, names, feeds):
            ids, mask = feeds["input_ids"], feeds["attention_mask"]
            assert ids.shape == mask.shape
            # class = number of real tokens, so order and padding are visible
            n = mask.sum(axis=1)
            out = np.zeros((len(n), len(LABELS)))
            out[np.arange(len(n)), n] = 1.0
            return [out]

    preds = hp.predict_onnx(FakeSession(), [[7, 7, 7], [7], [7, 7]], batch_size=2)
    assert preds == [3, 1, 2]


def test_agreement():
    assert hp.agreement([1, 2, 3], [1, 2, 0]) == pytest.approx(2 / 3)
    assert hp.agreement([], []) == 0.0


def test_softmax_rows_sums_to_one():
    import numpy as np

    p = hp.softmax_rows(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def _fake_export(tmp_path: Path) -> Path:
    run = tmp_path / "run-x"
    out = run / "export"
    (out / "onnx").mkdir(parents=True)
    for name in (*hp.ROOT_FILES, hp.ONNX_FP32, hp.ONNX_INT8, "README.md", hp.META_FILE):
        (out / name).write_bytes(b"x" * 10)
    return run


def test_upload_dry_run_lists_files_and_uploads_nothing(tmp_path, capsys, monkeypatch):
    run = _fake_export(tmp_path)

    def boom(*a, **k):
        raise AssertionError("network used in a dry run")

    monkeypatch.setattr("huggingface_hub.HfApi", boom, raising=False)
    assert hp.upload(run, "someone/repo", yes=False) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "private" in out
    for name in hp.upload_files(run / "export"):
        assert name in out


def test_upload_refuses_when_files_are_missing(tmp_path):
    run = _fake_export(tmp_path)
    (run / "export" / hp.ONNX_INT8).unlink()
    assert hp.upload(run, "someone/repo", yes=False) == 2


def test_upload_with_yes_keeps_repo_private(tmp_path, monkeypatch):
    run = _fake_export(tmp_path)
    calls = []

    class FakeApi:
        def create_repo(self, repo_id, **kw):
            calls.append(("create", kw["private"]))

        def update_repo_settings(self, repo_id, **kw):
            calls.append(("settings", kw["private"]))

        def upload_folder(self, **kw):
            calls.append(("upload", sorted(kw["allow_patterns"])))

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert hp.upload(run, "someone/repo", yes=True) == 0
    assert ("create", True) in calls and ("settings", True) in calls
    assert calls[-1][0] == "upload"


def test_latest_run_picks_newest_with_best(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name / "best").mkdir(parents=True)
        (tmp_path / name / "run.json").write_text(json.dumps({}))
    (tmp_path / "no-best").mkdir()
    import os

    os.utime(tmp_path / "a" / "run.json", (1, 1))
    assert hp.latest_run(tmp_path).name == "b"


def test_verify_without_data_measures_nothing(tmp_path):
    run = _fake_export(tmp_path)
    assert hp.verify(run, tmp_path / "missing", None, 8) is None
