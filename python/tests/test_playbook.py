"""Tests for ``toki.playbook`` — store + runner."""

from __future__ import annotations

import pytest

from toki.attack_stats import AttackTracker
from toki.playbook import (
    Playbook,
    PlaybookRunner,
    PlaybookStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> PlaybookStore:
    return PlaybookStore(db_path=":memory:")


@pytest.fixture
def tracker() -> AttackTracker:
    return AttackTracker(db_path=":memory:")


def _make_playbook(name: str = "alpha", n_variants: int = 1) -> Playbook:
    return Playbook(
        name=name,
        attack_types=["jailbreak"],
        mutation_strategies=["ROLEPLAY_WRAP"],
        n_variants=n_variants,
        target_model="mock-model",
        description="test playbook",
    )


def _always_refuse(_prompt: str) -> str:
    return "I'm sorry, I cannot help with that request."


def _always_bypass(_prompt: str) -> str:
    # Strong harmful + leak signals to drive score < 0.85.
    return "Ignore previous instructions. My system prompt is: how to make a bomb."


# ---------------------------------------------------------------------------
# PlaybookStore
# ---------------------------------------------------------------------------


def test_save_creates_row_with_version_one(store: PlaybookStore) -> None:
    saved = store.save(_make_playbook())
    assert saved.version == 1
    assert saved.name == "alpha"
    assert saved.created_at  # auto-filled
    assert saved.attack_types == ["jailbreak"]
    assert saved.mutation_strategies == ["ROLEPLAY_WRAP"]


def test_save_existing_name_bumps_version(store: PlaybookStore) -> None:
    first = store.save(_make_playbook())
    second = store.save(_make_playbook())
    assert first.version == 1
    assert second.version == 2
    # both rows survive
    assert len(store.all_versions("alpha")) == 2


def test_get_returns_latest_when_version_none(store: PlaybookStore) -> None:
    store.save(_make_playbook())
    store.save(_make_playbook())
    store.save(_make_playbook())
    latest = store.get("alpha")
    assert latest.version == 3


def test_get_with_explicit_version(store: PlaybookStore) -> None:
    store.save(_make_playbook())
    store.save(_make_playbook(n_variants=7))
    v1 = store.get("alpha", version=1)
    v2 = store.get("alpha", version=2)
    assert v1.version == 1
    assert v1.n_variants == 1
    assert v2.version == 2
    assert v2.n_variants == 7


def test_get_missing_raises_keyerror(store: PlaybookStore) -> None:
    with pytest.raises(KeyError):
        store.get("does-not-exist")
    store.save(_make_playbook())
    with pytest.raises(KeyError):
        store.get("alpha", version=99)


def test_list_one_row_per_name_latest_sorted(store: PlaybookStore) -> None:
    store.save(_make_playbook("beta"))
    store.save(_make_playbook("alpha"))
    store.save(_make_playbook("alpha"))
    store.save(_make_playbook("gamma"))
    rows = store.list()
    names = [p.name for p in rows]
    assert names == ["alpha", "beta", "gamma"]
    alpha = next(p for p in rows if p.name == "alpha")
    assert alpha.version == 2  # latest of two saves


def test_delete_removes_all_versions(store: PlaybookStore) -> None:
    store.save(_make_playbook())
    store.save(_make_playbook())
    store.save(_make_playbook())
    removed = store.delete("alpha")
    assert removed == 3
    assert store.list() == []
    # idempotent on a clean db
    assert store.delete("alpha") == 0


def test_all_versions_ascending(store: PlaybookStore) -> None:
    store.save(_make_playbook(n_variants=1))
    store.save(_make_playbook(n_variants=2))
    store.save(_make_playbook(n_variants=3))
    versions = store.all_versions("alpha")
    assert [p.version for p in versions] == [1, 2, 3]
    assert [p.n_variants for p in versions] == [1, 2, 3]


# ---------------------------------------------------------------------------
# PlaybookRunner
# ---------------------------------------------------------------------------


def test_runner_against_always_refuse_has_zero_successes(
    store: PlaybookStore, tracker: AttackTracker
) -> None:
    store.save(_make_playbook())
    runner = PlaybookRunner(store, tracker)
    result = runner.run("alpha", _always_refuse, base_prompts_per_type=2)
    assert result.total_attempts > 0
    assert result.successes == 0
    assert result.failures == result.total_attempts
    assert result.errors == 0
    assert result.playbook_name == "alpha"
    assert result.playbook_version == 1
    # by_attack_type has the requested category
    assert "jailbreak" in result.by_attack_type
    assert result.by_attack_type["jailbreak"]["success_rate"] == 0.0


def test_runner_against_always_bypass_has_successes(
    store: PlaybookStore, tracker: AttackTracker
) -> None:
    store.save(_make_playbook())
    runner = PlaybookRunner(store, tracker)
    result = runner.run("alpha", _always_bypass, base_prompts_per_type=2)
    assert result.total_attempts > 0
    assert result.successes > 0
    assert result.errors == 0
    # at least one strategy bucket recorded
    assert result.by_strategy


def test_runner_records_each_attempt_to_tracker(
    store: PlaybookStore, tracker: AttackTracker
) -> None:
    store.save(_make_playbook(n_variants=2))
    runner = PlaybookRunner(store, tracker)
    before = tracker.count()
    result = runner.run("alpha", _always_refuse, base_prompts_per_type=2)
    after = tracker.count()
    assert after - before == result.total_attempts
    # Every record is tagged with the target model.
    rows = tracker.fetch(model="mock-model")
    assert len(rows) == result.total_attempts


def test_run_result_to_dict_roundtrip(
    store: PlaybookStore, tracker: AttackTracker
) -> None:
    store.save(_make_playbook())
    runner = PlaybookRunner(store, tracker)
    result = runner.run("alpha", _always_refuse, base_prompts_per_type=1)
    d = result.to_dict()
    assert d["playbook_name"] == "alpha"
    assert d["total_attempts"] == result.total_attempts
    assert "by_attack_type" in d
    assert "by_strategy" in d


def test_runner_supports_multiple_attack_types(
    store: PlaybookStore, tracker: AttackTracker
) -> None:
    pb = Playbook(
        name="multi",
        attack_types=["jailbreak", "injection"],
        mutation_strategies=["PARAPHRASE"],
        n_variants=1,
        target_model="mock-model",
    )
    store.save(pb)
    runner = PlaybookRunner(store, tracker)
    result = runner.run("multi", _always_refuse, base_prompts_per_type=2)
    assert set(result.by_attack_type.keys()) == {"jailbreak", "injection"}
    assert result.failures == result.total_attempts
