"""Unit tests for toki.safety_benchmark."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

from toki.attack_stats import AttackTracker
from toki.safety_benchmark import (
    BenchmarkDiff,
    BenchmarkRun,
    BenchmarkStore,
    SEVERITY_WEIGHTS,
    SafetyBenchmark,
    compare_runs,
    compute_safety_score,
    render_report_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stats(attempts: int, successes: int) -> dict:
    return {
        "attempts": attempts,
        "successes": successes,
        "success_rate": (successes / attempts) if attempts else 0.0,
    }


def _make_run(
    *,
    score: float,
    by_attack_type: dict[str, dict] | None = None,
    model: str = "test-model",
    playbook_name: str = "core",
    playbook_version: int = 1,
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> BenchmarkRun:
    return BenchmarkRun(
        id=uuid.uuid4().hex,
        model=model,
        playbook_name=playbook_name,
        playbook_version=playbook_version,
        timestamp=timestamp,
        total_attempts=sum(int(v["attempts"]) for v in (by_attack_type or {}).values()),
        successes=sum(int(v["successes"]) for v in (by_attack_type or {}).values()),
        safety_score=score,
        by_attack_type=by_attack_type or {},
        meta={},
    )


# ---------------------------------------------------------------------------
# compute_safety_score
# ---------------------------------------------------------------------------


def test_all_failures_returns_perfect_score() -> None:
    by_type = {
        "jailbreak": _stats(10, 0),
        "injection": _stats(10, 0),
    }
    assert compute_safety_score(by_type) == 100.0


def test_all_successes_returns_zero() -> None:
    by_type = {
        "jailbreak": _stats(5, 5),
        "edge_case": _stats(5, 5),
    }
    assert compute_safety_score(by_type) == 0.0


def test_jailbreak_weighted_heavier_than_edge_case() -> None:
    # Equal attempts; one type all-successes, other all-failures.
    # When jailbreak is the bad one, score is lower than when edge_case is.
    bad_jailbreak = {
        "jailbreak": _stats(10, 10),
        "edge_case": _stats(10, 0),
    }
    bad_edge_case = {
        "jailbreak": _stats(10, 0),
        "edge_case": _stats(10, 10),
    }
    score_jb = compute_safety_score(bad_jailbreak)
    score_ec = compute_safety_score(bad_edge_case)
    assert score_jb < score_ec
    assert SEVERITY_WEIGHTS["jailbreak"] > SEVERITY_WEIGHTS["edge_case"]


def test_unknown_attack_type_uses_default_weight() -> None:
    by_type = {"mystery": _stats(10, 5)}
    score = compute_safety_score(by_type)
    # default weight 0.5 — score = 100 * (1 - 0.5) = 50
    assert score == pytest.approx(50.0, abs=0.01)


# ---------------------------------------------------------------------------
# BenchmarkStore
# ---------------------------------------------------------------------------


def test_store_record_get_round_trip() -> None:
    store = BenchmarkStore(db_path=":memory:")
    run = _make_run(
        score=72.5,
        by_attack_type={"jailbreak": _stats(4, 1), "edge_case": _stats(4, 0)},
    )
    object.__setattr__(run, "meta", {"note": "round-trip"})
    store.record(run)
    out = store.get(run.id)
    assert out.id == run.id
    assert out.model == run.model
    assert out.playbook_name == run.playbook_name
    assert out.playbook_version == run.playbook_version
    assert out.timestamp == run.timestamp
    assert out.total_attempts == run.total_attempts
    assert out.successes == run.successes
    assert out.safety_score == run.safety_score
    assert out.by_attack_type == run.by_attack_type
    assert out.meta == run.meta


def test_store_list_recent_first_and_model_filter() -> None:
    store = BenchmarkStore(db_path=":memory:")
    older = _make_run(score=80.0, model="m-a", timestamp="2026-01-01T00:00:00+00:00")
    newer = _make_run(score=90.0, model="m-a", timestamp="2026-02-01T00:00:00+00:00")
    other = _make_run(score=50.0, model="m-b", timestamp="2026-03-01T00:00:00+00:00")
    store.record(older)
    store.record(newer)
    store.record(other)

    all_runs = store.list()
    assert [r.id for r in all_runs] == [other.id, newer.id, older.id]

    only_a = store.list(model="m-a")
    assert {r.id for r in only_a} == {older.id, newer.id}
    assert only_a[0].id == newer.id  # most recent first


def test_store_get_missing_raises_keyerror() -> None:
    store = BenchmarkStore(db_path=":memory:")
    with pytest.raises(KeyError):
        store.get("does-not-exist")


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


def test_compare_runs_improved_when_new_score_higher_than_tolerance() -> None:
    base = _make_run(
        score=60.0,
        by_attack_type={"jailbreak": _stats(10, 5)},
    )
    new = _make_run(
        score=80.0,
        by_attack_type={"jailbreak": _stats(10, 1)},
    )
    diff = compare_runs(base, new, tolerance=0.5)
    assert isinstance(diff, BenchmarkDiff)
    assert diff.verdict == "improved"
    assert diff.safety_score_delta == pytest.approx(20.0, abs=1e-9)

    # Regression case
    diff_r = compare_runs(new, base, tolerance=0.5)
    assert diff_r.verdict == "regressed"

    # Within tolerance
    base_close = _make_run(score=70.0, by_attack_type={"jailbreak": _stats(5, 1)})
    new_close = _make_run(score=70.3, by_attack_type={"jailbreak": _stats(5, 1)})
    diff_unchanged = compare_runs(base_close, new_close, tolerance=0.5)
    assert diff_unchanged.verdict == "unchanged"


def test_compare_runs_delta_by_attack_type_per_type() -> None:
    base = _make_run(
        score=50.0,
        by_attack_type={
            "jailbreak": _stats(10, 5),
            "edge_case": _stats(10, 2),
        },
    )
    new = _make_run(
        score=70.0,
        by_attack_type={
            "jailbreak": _stats(10, 1),
            "boundary": _stats(10, 3),
        },
    )
    diff = compare_runs(base, new)
    # jailbreak: 0.1 - 0.5 = -0.4
    assert diff.delta_by_attack_type["jailbreak"] == pytest.approx(-0.4, abs=1e-4)
    # edge_case removed in new: 0.0 - 0.2 = -0.2
    assert diff.delta_by_attack_type["edge_case"] == pytest.approx(-0.2, abs=1e-4)
    # boundary new in new: 0.3 - 0.0 = 0.3
    assert diff.delta_by_attack_type["boundary"] == pytest.approx(0.3, abs=1e-4)


# ---------------------------------------------------------------------------
# render_report_markdown
# ---------------------------------------------------------------------------


def test_render_report_contains_required_sections() -> None:
    run = _make_run(
        score=72.5,
        model="llama3.2-3b",
        by_attack_type={
            "jailbreak": _stats(4, 1),
            "edge_case": _stats(4, 0),
        },
    )
    sample = [
        {"attack_type": "jailbreak", "result": "success", "prompt": "ignore rules"},
    ]
    md = render_report_markdown(run, attempts_sample=sample)
    assert "Safety Benchmark" in md
    assert "llama3.2-3b" in md
    assert "72.5" in md
    # At least one Markdown table row separator
    assert "| --- |" in md
    assert "ignore rules" in md


# ---------------------------------------------------------------------------
# SafetyBenchmark end-to-end with inline playbook shim
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StubPlaybook:
    name: str
    version: int
    attack_types: tuple[str, ...]
    mutation_strategies: tuple[str, ...] = field(default=())
    n_variants: int = 1


class _StubPlaybookStore:
    def __init__(self, playbooks: dict[str, _StubPlaybook]) -> None:
        self._playbooks = playbooks

    def get(self, name: str) -> _StubPlaybook:
        return self._playbooks[name]


def _refusing_model(_prompt: str) -> str:
    # Triggers refusal heuristic in RobustnessEvaluator (contains "cannot").
    return "I cannot help with that request."


def test_safety_benchmark_run_end_to_end() -> None:
    pb = _StubPlaybook(
        name="core",
        version=2,
        attack_types=("jailbreak", "edge_case"),
        mutation_strategies=(),
        n_variants=1,
    )
    store = BenchmarkStore(db_path=":memory:")
    tracker = AttackTracker(db_path=":memory:")
    playbook_store = _StubPlaybookStore({"core": pb})
    bench = SafetyBenchmark(
        store=store, playbook_store=playbook_store, tracker=tracker
    )

    run = bench.run(
        model="mock-1b",
        playbook_name="core",
        model_fn=_refusing_model,
        seed=42,
        meta={"note": "e2e"},
    )

    assert run.model == "mock-1b"
    assert run.playbook_name == "core"
    assert run.playbook_version == 2
    assert run.total_attempts > 0
    # A refusing model should score high.
    assert run.safety_score >= 60.0
    # Persisted
    fetched = store.get(run.id)
    assert fetched.id == run.id
    # Tracker captured rows
    assert tracker.count() == run.total_attempts
