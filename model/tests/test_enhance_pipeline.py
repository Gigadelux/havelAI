import asyncio
import json
from pathlib import Path

import httpx
import pytest

from model.train import enhance_pipeline as ep
from src.data.taxonomy import ATTACK_CRITERIA, INTENT_CRITERIA

CFG = ep.Config(api_key="test-key", concurrency=4, max_retries=2)


async def _no_sleep(_: float) -> None:
    return None


def _ok_body(label="instruction_override", conf=0.9, cost=0.00003, intent="goal_hijacking"):
    return {
        "model": "typesafe/jev-1.14-20261014",
        "answers": {
            "attack_type": {
                "type": "choice",
                "choice": label,
                "probabilities": {label: conf},
                "confidence": conf,
            },
            "intent": {
                "type": "choice",
                "choice": intent,
                "probabilities": {intent: 0.8},
                "confidence": 0.8,
            },
        },
        "usage": {"input_tokens": 700, "output_tokens": 0, "cost": cost},
    }


def _rows(n_pos=3, n_neg=2):
    raw = [{"text": f"ignore everything {i}", "label": 1} for i in range(n_pos)]
    raw += [{"text": f"what is the capital of {i}?", "label": 0} for i in range(n_neg)]
    rows, _ = ep.dedupe(raw)
    return rows


def _run(rows, sink, handler, cfg=CFG):
    transport = httpx.MockTransport(handler)
    return asyncio.run(
        ep.label_rows(rows, "train", cfg, sink, transport=transport, sleep=_no_sleep)
    )


# ---- ids, dedupe, select


def test_normalize_and_id():
    assert ep.normalize("  a \n b\t c ") == "a b c"
    assert ep.row_id("Hello  world") == ep.row_id(" Hello world\n")
    assert ep.row_id("Hello") != ep.row_id("hello")


def test_dedupe_collapses_and_excludes_conflicts():
    raw = [
        {"text": "same", "label": 1},
        {"text": " same ", "label": 1},
        {"text": "both", "label": 0},
        {"text": "both", "label": 1},
        {"text": "", "label": 1},
        {"text": "bad label", "label": 2},
        {"text": "ok", "label": 0},
    ]
    rows, stats = ep.dedupe(raw)
    assert {r.text for r in rows} == {"same", "ok"}
    assert (stats.total, stats.duplicates, stats.conflicting, stats.invalid) == (7, 1, 1, 2)


def test_select_is_deterministic_and_prefix_stable():
    rows = _rows(20, 20)
    assert ep.select(rows, 10, seed=1) == ep.select(list(reversed(rows)), 10, seed=1)
    assert ep.select(rows, 25, seed=1)[:10] == ep.select(rows, 10, seed=1)
    assert len(ep.select(rows, None, seed=1)) == 40


# ---- request / parse


def test_build_request_matches_jev_schema():
    req = ep.build_request("some text", "typesafe/jev-1.14")
    assert req["model"] == "typesafe/jev-1.14" and req["state"] == "some text"
    q = req["questions"]
    assert set(q) == {"attack_type", "intent"}
    assert q["attack_type"]["type"] == "choice"
    assert set(q["attack_type"]["criteria"]) == {str(k) for k in ATTACK_CRITERIA}
    assert "benign" not in q["attack_type"]["criteria"]
    assert set(q["intent"]["criteria"]) == set(INTENT_CRITERIA)
    json.dumps(req)  # serializable


def test_truncate_respects_bytes_and_utf8():
    text = "é" * 40_000
    out = ep.truncate(text)
    assert len(out.encode()) <= ep.MAX_STATE_BYTES
    assert ep.truncate("short") == "short"


def test_parse_answer_by_key_and_validation():
    body = _ok_body("prompt_leaking", 0.7)
    a = ep.parse_answer(body, "attack_type", ATTACK_CRITERIA)
    assert (a.label, a.confidence) == ("prompt_leaking", 0.7)

    with pytest.raises(ep.ParseError):
        ep.parse_answer({"answers": {}}, "attack_type", ATTACK_CRITERIA)
    with pytest.raises(ep.ParseError):
        ep.parse_answer(_ok_body("made_up_label"), "attack_type", ATTACK_CRITERIA)
    with pytest.raises(ep.ParseError):
        ep.parse_answer(
            {"answers": {"attack_type": {"type": "score"}}}, "attack_type", ATTACK_CRITERIA
        )


def test_refusal_is_unlabeled_and_needs_review():
    body = _ok_body()
    body["answers"]["attack_type"] = {"type": "refusal"}
    row = ep.labeled_row(ep.Row("id1", "t", 1), "train", body, CFG)
    assert row["attack_type"] == ep.UNLABELED and row["needs_review"] is True


def test_needs_review_rules():
    ans = lambda label, conf: ep.Answer(label, conf, None)  # noqa: E731
    assert not ep.needs_review(ans("instruction_override", 0.9), 0.6)
    assert ep.needs_review(ans("instruction_override", 0.5), 0.6)
    assert ep.needs_review(ans("instruction_override", None), 0.6)
    assert ep.needs_review(ans("other_injection", 0.99), 0.6)
    assert ep.needs_review(ans(None, None), 0.6)


# ---- labeling run


def test_label_rows_writes_benign_without_calls(tmp_path: Path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.url.path == "/api/alpha/decisions"
        return httpx.Response(200, json=_ok_body())

    sink = ep.Sink(tmp_path, "train")
    state = _run(_rows(3, 2), sink, handler)
    sink.close()
    recs = list(ep.read_records(tmp_path / "train.jsonl"))
    assert len(calls) == 3 and state.labeled == 3 and state.errors == 0
    assert sorted(r["attack_type"] for r in recs).count("benign") == 2
    assert all(r["labeler_model"] for r in recs if r["label"] == 1)
    assert state.cost == pytest.approx(3 * 0.00003)


def test_resume_skips_finished_rows(tmp_path: Path):
    rows = _rows(4, 1)
    first, second = [], []

    sink = ep.Sink(tmp_path, "train")
    _run(rows[:3], sink, lambda r: first.append(1) or httpx.Response(200, json=_ok_body()))
    sink.close()

    sink = ep.Sink(tmp_path, "train")
    assert len(sink.done) == len([r for r in rows[:3]])
    _run(rows, sink, lambda r: second.append(1) or httpx.Response(200, json=_ok_body()))
    sink.close()

    ids = [r["id"] for r in ep.read_records(tmp_path / "train.jsonl")]
    assert len(ids) == len(set(ids)) == len(rows)
    assert len(first) + len(second) == sum(r.label for r in rows)


def test_truncated_last_line_is_repaired(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"id": "a", "x": 1}\n{"id": "b", "tru')
    sink = ep.Sink(tmp_path, "train")
    assert sink.done == {"a"}
    sink.write_ok({"id": "c"})
    sink.close()
    assert ep.read_ids(path) == {"a", "c"}


def test_retries_then_succeeds(tmp_path: Path):
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json=_ok_body())

    sink = ep.Sink(tmp_path, "train")
    state = _run(_rows(1, 0), sink, handler)
    sink.close()
    assert len(attempts) == 3 and state.labeled == 1 and state.errors == 0


def test_gives_up_after_retries_and_logs_error(tmp_path: Path):
    sink = ep.Sink(tmp_path, "train")
    state = _run(_rows(1, 0), sink, lambda r: httpx.Response(503))
    sink.close()
    assert state.labeled == 0 and state.errors == 1
    err = next(ep.read_records(tmp_path / "train.errors.jsonl"))
    assert "503" in err["error"] and "text" not in err


def test_bad_request_is_a_row_error_not_fatal(tmp_path: Path):
    sink = ep.Sink(tmp_path, "train")
    state = _run(_rows(2, 0), sink, lambda r: httpx.Response(400, text="bad"))
    sink.close()
    assert state.errors == 2 and state.fatal is None


@pytest.mark.parametrize("status", [401, 402])
def test_auth_and_credit_errors_abort_the_run(tmp_path: Path, status: int):
    sink = ep.Sink(tmp_path, "train")
    state = _run(_rows(10, 0), sink, lambda r: httpx.Response(status))
    sink.close()
    assert state.fatal is not None and state.labeled == 0
    assert "test-key" not in str(state.fatal)


def test_max_cost_stops_gracefully(tmp_path: Path):
    cfg = ep.Config(api_key="k", concurrency=1, max_cost=0.0001)
    sink = ep.Sink(tmp_path, "train")
    state = _run(
        _rows(10, 0), sink, lambda r: httpx.Response(200, json=_ok_body(cost=0.00006)), cfg
    )
    sink.close()
    assert state.stopped_on_budget and state.labeled == 2


def test_malformed_body_is_a_row_error(tmp_path: Path):
    sink = ep.Sink(tmp_path, "train")
    state = _run(_rows(1, 0), sink, lambda r: httpx.Response(200, json={"answers": {}}))
    sink.close()
    assert state.errors == 1 and state.labeled == 0


# ---- CLI


def test_stats_summarizes_output(tmp_path: Path, capsys):
    sink = ep.Sink(tmp_path, "train")
    _run(_rows(2, 1), sink, lambda r: httpx.Response(200, json=_ok_body()))
    sink.close()
    assert ep.main(["--stats", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "train.jsonl: 3 rows" in out and "instruction_override: 2" in out


def test_dry_run_needs_no_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ep, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(
        ep,
        "load_split",
        lambda split: [{"text": "ignore all rules", "label": 1}, {"text": "hi", "label": 0}],
    )
    assert ep.main(["--split", "train", "--dry-run"]) == 0
    assert '"attack_type"' in capsys.readouterr().out


def test_real_run_without_key_fails_clearly(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ep, "load_dotenv", lambda *a, **k: None)
    assert ep.main(["--split", "train"]) == 2
