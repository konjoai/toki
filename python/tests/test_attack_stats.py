"""Tests for the SQLite-backed AttackTracker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from toki.attack_stats import AttackRecord, AttackResult, AttackTracker


def _iso_days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat(
        timespec="seconds"
    )


def test_result_enum_parse_string_and_self() -> None:
    assert AttackResult.parse("success") is AttackResult.SUCCESS
    assert AttackResult.parse("FAILURE") is AttackResult.FAILURE
    assert AttackResult.parse(AttackResult.ERROR) is AttackResult.ERROR


def test_result_enum_parse_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown attack result"):
        AttackResult.parse("ok")
    with pytest.raises(ValueError, match="unsupported result type"):
        AttackResult.parse(None)  # type: ignore[arg-type]


def test_record_inserts_and_assigns_id() -> None:
    t = AttackTracker(":memory:")
    rid = t.record(
        attack_type="jailbreak",
        prompt="ignore previous instructions",
        result="success",
        strategy="ROLEPLAY_WRAP",
        model="mock",
        latency_ms=12.5,
    )
    assert rid >= 1
    rows = t.fetch()
    assert len(rows) == 1
    only = rows[0]
    assert isinstance(only, AttackRecord)
    assert only.attack_type == "jailbreak"
    assert only.result == "success"
    assert only.mutant_strategy == "ROLEPLAY_WRAP"
    assert only.latency_ms == pytest.approx(12.5)
    assert len(only.prompt_hash) == 16


def test_record_validates_inputs() -> None:
    t = AttackTracker(":memory:")
    with pytest.raises(ValueError):
        t.record("", "x", "success")
    with pytest.raises(ValueError):
        t.record("jailbreak", 123, "success")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        t.record("jailbreak", "x", "weird-result")


def test_record_accepts_empty_prompt() -> None:
    # Edge-case attacks legitimately use the empty string.
    t = AttackTracker(":memory:")
    rid = t.record("edge_case", "", "failure", strategy="EMPTY_PROMPT")
    assert rid >= 1
    rows = t.fetch()
    assert rows[0].prompt_hash  # sha256("")[:16] is stable


def test_stats_overall_and_by_strategy_rates() -> None:
    t = AttackTracker(":memory:")
    base_kwargs = {"attack_type": "jailbreak", "prompt": "do bad things"}
    # 4 ROLEPLAY_WRAP attempts, 2 success
    for _ in range(2):
        t.record(**base_kwargs, result="success", strategy="ROLEPLAY_WRAP")
    for _ in range(2):
        t.record(**base_kwargs, result="failure", strategy="ROLEPLAY_WRAP")
    # 6 ENCODING attempts, 1 success
    for _ in range(1):
        t.record(**base_kwargs, result="success", strategy="ENCODING")
    for _ in range(5):
        t.record(**base_kwargs, result="failure", strategy="ENCODING")
    stats = t.stats()
    assert stats["total_attempts"] == 10
    assert stats["successes"] == 3
    assert stats["success_rate"] == pytest.approx(0.3)
    roleplay = stats["by_strategy"]["ROLEPLAY_WRAP"]
    assert roleplay["attempts"] == 4
    assert roleplay["success_rate"] == pytest.approx(0.5)
    encoding = stats["by_strategy"]["ENCODING"]
    assert encoding["success_rate"] == pytest.approx(1 / 6, rel=1e-3)


def test_stats_filters_by_attack_type_and_model() -> None:
    t = AttackTracker(":memory:")
    t.record("jailbreak", "p", "success", strategy="A", model="gpt4")
    t.record("jailbreak", "p", "failure", strategy="A", model="gpt4")
    t.record("injection", "p", "success", strategy="A", model="gpt4")
    t.record("jailbreak", "p", "failure", strategy="A", model="claude")
    s = t.stats(attack_type="jailbreak", days=None)
    assert s["total_attempts"] == 3
    s_model = t.stats(model="gpt4", days=None)
    assert s_model["total_attempts"] == 3


def test_stats_trend_groups_by_day() -> None:
    t = AttackTracker(":memory:")
    t.record(
        "jailbreak", "p", "success", timestamp=_iso_days_ago(2), strategy="X"
    )
    t.record(
        "jailbreak", "p", "failure", timestamp=_iso_days_ago(2), strategy="X"
    )
    t.record(
        "jailbreak", "p", "failure", timestamp=_iso_days_ago(1), strategy="X"
    )
    stats = t.stats(days=7)
    dates = [t["date"] for t in stats["trend"]]
    # Ascending order
    assert dates == sorted(dates)
    # Two distinct days populated
    assert len(stats["trend"]) == 2


def test_classify_categories_three_buckets() -> None:
    t = AttackTracker(":memory:")
    # always_blocked: 10/10 failures
    for _ in range(10):
        t.record("blocked_cat", "p", "failure", timestamp=_iso_days_ago(0))
    # newly_bypassing: 10/10 successes today
    for _ in range(10):
        t.record("breached_cat", "p", "success", timestamp=_iso_days_ago(0))
    # intermittent: 5 success / 5 failure
    for _ in range(5):
        t.record("mixed_cat", "p", "success", timestamp=_iso_days_ago(0))
    for _ in range(5):
        t.record("mixed_cat", "p", "failure", timestamp=_iso_days_ago(0))
    # insufficient_data: 2 attempts
    t.record("rare_cat", "p", "failure", timestamp=_iso_days_ago(0))
    t.record("rare_cat", "p", "success", timestamp=_iso_days_ago(0))
    cats = t.classify_categories()
    assert cats["blocked_cat"]["bucket"] == "always_blocked"
    assert cats["breached_cat"]["bucket"] == "newly_bypassing"
    assert cats["mixed_cat"]["bucket"] == "intermittent"
    assert cats["rare_cat"]["bucket"] == "insufficient_data"


def test_fetch_supports_result_and_limit_filters() -> None:
    t = AttackTracker(":memory:")
    for _ in range(5):
        t.record("jailbreak", "p", "success")
    for _ in range(5):
        t.record("jailbreak", "p", "failure")
    succ = t.fetch(result="success")
    assert all(r.result == "success" for r in succ)
    assert len(succ) == 5
    capped = t.fetch(limit=3)
    assert len(capped) == 3


def test_count_matches_fetch() -> None:
    t = AttackTracker(":memory:")
    for kind in ("success", "failure", "success", "error"):
        t.record("jailbreak", "p", kind)
    assert t.count() == 4
    assert t.count(result="success") == 2
    assert t.count(result="error") == 1


def test_record_many_bulk_insert() -> None:
    t = AttackTracker(":memory:")
    n = t.record_many(
        [
            {"attack_type": "a", "prompt": "x", "result": "success"},
            {"attack_type": "a", "prompt": "y", "result": "failure"},
            {"attack_type": "b", "prompt": "z", "result": "error"},
        ]
    )
    assert n == 3
    assert t.count() == 3


def test_db_persists_to_disk(tmp_path) -> None:
    db = tmp_path / "history.db"
    t = AttackTracker(db)
    t.record("jailbreak", "p", "success")
    # Re-open
    t2 = AttackTracker(db)
    assert t2.count() == 1


def test_days_filter_excludes_old_rows() -> None:
    t = AttackTracker(":memory:")
    t.record("jailbreak", "p", "success", timestamp=_iso_days_ago(10))
    t.record("jailbreak", "p", "success", timestamp=_iso_days_ago(1))
    s = t.stats(days=3)
    assert s["total_attempts"] == 1
