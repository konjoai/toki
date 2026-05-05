"""Tests for toki.leaderboard — multi-model adversarial leaderboard."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from toki.compare import BASELINES, ModelSpec, baseline_mixed, baseline_safe, baseline_unsafe
from toki.leaderboard import (
    Leaderboard,
    LeaderboardConfig,
    LeaderboardEntry,
    LeaderboardResult,
    PairResult,
    _bonferroni_alpha,
    _compare_pair,
    _rank_entries,
    _all_baseline_specs,
)
from toki.compare import ModelScores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_cfg(**kw) -> LeaderboardConfig:
    """Return a tiny LeaderboardConfig for fast tests."""
    defaults = dict(
        name="test_lb",
        seed=7,
        jailbreak_count=3,
        injection_count=3,
        boundary_count=2,
    )
    defaults.update(kw)
    return LeaderboardConfig(**defaults)


def _make_scores(name: str, mean: float, scores: list | None = None) -> ModelScores:
    if scores is None:
        scores = [mean] * 10
    return ModelScores(
        name=name,
        mean_score=mean,
        refusal_rate=0.0,
        harmful_rate=0.0,
        leak_rate=0.0,
        by_category={},
        scores=scores,
        total_prompts=len(scores),
    )


SAFE   = ModelSpec("safe",   baseline_safe)
UNSAFE = ModelSpec("unsafe", baseline_unsafe)
MIXED  = ModelSpec("mixed",  baseline_mixed)


# ---------------------------------------------------------------------------
# Test 1: Bonferroni correction formula
# ---------------------------------------------------------------------------

def test_bonferroni_alpha_k3():
    """For k=3 models: n_pairs = 3, α_bonf = α/3."""
    k = 3
    n_pairs = k * (k - 1) // 2   # = 3
    result = _bonferroni_alpha(0.05, n_pairs)
    assert result == pytest.approx(0.05 / 3)


def test_bonferroni_alpha_k4():
    """For k=4 models: n_pairs = 6, α_bonf = α/6."""
    k = 4
    n_pairs = k * (k - 1) // 2   # = 6
    result = _bonferroni_alpha(0.05, n_pairs)
    assert result == pytest.approx(0.05 / 6)


def test_bonferroni_alpha_zero_pairs():
    """If n_pairs == 0 (degenerate), returns nominal alpha unchanged."""
    assert _bonferroni_alpha(0.05, 0) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Test 4: LeaderboardEntry dataclass construction
# ---------------------------------------------------------------------------

def test_leaderboard_entry_fields():
    entry = LeaderboardEntry(
        name="safe",
        mean_score=0.95,
        n_comparisons=2,
        wins=2,
        losses=0,
        ties=0,
        rank=1,
        significant=True,
    )
    assert entry.rank == 1
    assert entry.name == "safe"
    assert entry.significant is True


# ---------------------------------------------------------------------------
# Test 5: Leaderboard requires ≥ 2 models
# ---------------------------------------------------------------------------

def test_leaderboard_requires_two_models():
    with pytest.raises(ValueError, match="at least 2 models"):
        Leaderboard([SAFE])


# ---------------------------------------------------------------------------
# Test 6: Leaderboard requires unique model names
# ---------------------------------------------------------------------------

def test_leaderboard_requires_unique_names():
    dup = ModelSpec("safe", baseline_unsafe)   # same name, different fn
    with pytest.raises(ValueError, match="unique"):
        Leaderboard([SAFE, dup])


# ---------------------------------------------------------------------------
# Test 7: Three-model leaderboard — structure
# ---------------------------------------------------------------------------

def test_three_model_leaderboard_structure():
    """k=3 → 3 pairs, 3 entries, all ranks ∈ {1,2,3}."""
    cfg = _small_cfg()
    lb = Leaderboard([SAFE, UNSAFE, MIXED], cfg)
    result = lb.run()

    assert result.n_models == 3
    assert result.n_pairs  == 3                 # 3*(3-1)//2
    assert len(result.entries) == 3
    assert len(result.pairs)   == 3
    ranks = {e.rank for e in result.entries}
    assert ranks.issubset({1, 2, 3})


# ---------------------------------------------------------------------------
# Test 8: Safe model ranks #1 above unsafe
# ---------------------------------------------------------------------------

def test_safe_outranks_unsafe():
    """safe model (high refusal → high score) must rank above unsafe (harmful → low score)."""
    cfg = _small_cfg()
    lb = Leaderboard([SAFE, UNSAFE, MIXED], cfg)
    result = lb.run()

    by_name = {e.name: e for e in result.entries}
    assert by_name["safe"].rank < by_name["unsafe"].rank, (
        f"safe rank={by_name['safe'].rank} should be < unsafe rank={by_name['unsafe'].rank}"
    )


# ---------------------------------------------------------------------------
# Test 9: Bonferroni alpha stored in result
# ---------------------------------------------------------------------------

def test_bonferroni_alpha_in_result():
    """Result must carry the corrected alpha = nominal / n_pairs."""
    cfg = _small_cfg(alpha=0.05)
    lb = Leaderboard([SAFE, UNSAFE, MIXED], cfg)
    result = lb.run()

    expected = 0.05 / 3   # 3 pairs for k=3
    assert result.alpha_bonferroni == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 10: PairResult winner semantics
# ---------------------------------------------------------------------------

def test_pair_result_winner_semantics():
    """safe vs unsafe → safe wins; alpha_bonferroni stored in pair."""
    cfg = _small_cfg()
    lb = Leaderboard([SAFE, UNSAFE], cfg)
    result = lb.run()

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.winner in {"safe", "unsafe", "tie"}
    assert pair.alpha_bonferroni == pytest.approx(result.alpha_bonferroni)
    # safe has higher mean than unsafe, winner should be safe (or tie if not sig)
    if pair.significant:
        assert pair.winner == "safe"


# ---------------------------------------------------------------------------
# Test 11: _rank_entries with synthetic scores
# ---------------------------------------------------------------------------

def test_rank_entries_ordering():
    """Higher mean score → lower rank number (rank 1 = safest)."""
    a = _make_scores("alpha",  0.90)
    b = _make_scores("beta",   0.50)
    c = _make_scores("gamma",  0.70)
    all_scores = {"alpha": a, "beta": b, "gamma": c}

    # Build fake pairs where all are ties (significant=False)
    pairs = [
        PairResult("alpha", "beta",  0.90, 0.50, "tie", False, 0, 1, 0, 1, 0.05),
        PairResult("alpha", "gamma", 0.90, 0.70, "tie", False, 0, 1, 0, 1, 0.05),
        PairResult("beta",  "gamma", 0.50, 0.70, "tie", False, 0, 1, 0, 1, 0.05),
    ]
    entries = _rank_entries(all_scores, pairs)
    by_name = {e.name: e for e in entries}
    assert by_name["alpha"].rank == 1
    assert by_name["gamma"].rank == 2
    assert by_name["beta"].rank  == 3


# ---------------------------------------------------------------------------
# Test 12: _rank_entries tie in mean score shares rank
# ---------------------------------------------------------------------------

def test_rank_entries_shared_rank_on_equal_mean():
    """Two models with identical mean scores share the same rank."""
    a = _make_scores("a", 0.80, [0.8] * 10)
    b = _make_scores("b", 0.80, [0.8] * 10)
    c = _make_scores("c", 0.50, [0.5] * 10)
    all_scores = {"a": a, "b": b, "c": c}
    pairs = [
        PairResult("a", "b", 0.80, 0.80, "tie", False, 0, 1, 0, 1, 0.05),
        PairResult("a", "c", 0.80, 0.50, "a",   True,  5, 0, 5, 0, 0.05),
        PairResult("b", "c", 0.80, 0.50, "b",   True,  5, 0, 5, 0, 0.05),
    ]
    entries = _rank_entries(all_scores, pairs)
    by_name = {e.name: e for e in entries}
    # a and b share rank 1 (same mean), c gets rank 3
    assert by_name["a"].rank == by_name["b"].rank == 1
    assert by_name["c"].rank == 3


# ---------------------------------------------------------------------------
# Test 13: Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    cfg = _small_cfg(name="persistence_test", output_dir=str(tmp_path))
    lb = Leaderboard([SAFE, UNSAFE, MIXED], cfg)
    result = lb.run(save=True)

    out = tmp_path / f"{result.timestamp}_persistence_test" / "leaderboard.json"
    assert out.exists()

    loaded = LeaderboardResult.load(out)
    assert loaded.name == result.name
    assert loaded.n_models == result.n_models
    assert loaded.n_pairs  == result.n_pairs
    assert loaded.alpha_bonferroni == pytest.approx(result.alpha_bonferroni)
    assert len(loaded.entries) == len(result.entries)
    assert len(loaded.pairs)   == len(result.pairs)
    assert isinstance(loaded.entries[0], LeaderboardEntry)
    assert isinstance(loaded.pairs[0],   PairResult)
    # Spot-check winner stability
    assert loaded.entries[0].name == result.entries[0].name
    assert loaded.entries[0].rank == result.entries[0].rank


# ---------------------------------------------------------------------------
# Test 14: save() raises FileExistsError on second call (no overwrite)
# ---------------------------------------------------------------------------

def test_save_no_overwrite(tmp_path):
    cfg = _small_cfg(name="no_overwrite", output_dir=str(tmp_path))
    lb = Leaderboard([SAFE, UNSAFE], cfg)
    result = lb.run(save=True)

    with pytest.raises(FileExistsError):
        result.save(str(tmp_path))


# ---------------------------------------------------------------------------
# Test 15: format_table output contains expected fields
# ---------------------------------------------------------------------------

def test_format_table_contains_all_models():
    cfg = _small_cfg()
    result = Leaderboard([SAFE, UNSAFE, MIXED], cfg).run()
    table = result.format_table()
    for model_name in ("safe", "unsafe", "mixed"):
        assert model_name in table, f"{model_name} missing from leaderboard table"
    # Check header-like markers
    assert "Rank" in table or "rank" in table.lower()
    assert "Bonferroni" in table or "bonf" in table.lower()


# ---------------------------------------------------------------------------
# Test 16: All-baseline specs helper
# ---------------------------------------------------------------------------

def test_all_baseline_specs():
    specs = _all_baseline_specs()
    assert len(specs) == len(BASELINES)
    names = {s.name for s in specs}
    assert names == set(BASELINES.keys())
    for spec in specs:
        # Each spec's fn is callable and returns a string
        out = spec.model_fn("some prompt")
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# Test 17: _compare_pair uses Bonferroni-corrected alpha
# ---------------------------------------------------------------------------

def test_compare_pair_uses_corrected_alpha():
    """The alpha stored in pair.alpha_bonferroni must be the corrected value."""
    a = _make_scores("safe",   0.95, [1.0] * 20)
    b = _make_scores("unsafe", 0.05, [0.0] * 20)
    alpha_bonf = 0.05 / 6  # simulating k=4 (6 pairs)
    pair = _compare_pair(a, b, alpha_bonf)
    assert pair.alpha_bonferroni == pytest.approx(alpha_bonf)
    # perfect separation → significant
    assert pair.significant is True
    assert pair.winner == "safe"


# ---------------------------------------------------------------------------
# Test 18: wins/losses tally is consistent across all entries
# ---------------------------------------------------------------------------

def test_wins_losses_consistent(tmp_path):
    """Total wins across all entries must equal total losses (zero-sum)."""
    cfg = _small_cfg()
    result = Leaderboard([SAFE, UNSAFE, MIXED], cfg).run()
    total_wins   = sum(e.wins   for e in result.entries)
    total_losses = sum(e.losses for e in result.entries)
    assert total_wins == total_losses


# ---------------------------------------------------------------------------
# Test 19: Four-model leaderboard — n_pairs = 6
# ---------------------------------------------------------------------------

def test_four_model_n_pairs():
    """k=4 → 4*(4-1)//2 = 6 pairs."""
    def _fn(suffix: str):
        return lambda _p: f"response_{suffix}"

    models = [
        ModelSpec("m1", _fn("1")),
        ModelSpec("m2", _fn("2")),
        ModelSpec("m3", _fn("3")),
        ModelSpec("safe", baseline_safe),
    ]
    cfg = _small_cfg(name="four_models")
    lb = Leaderboard(models, cfg)
    result = lb.run()
    assert result.n_pairs  == 6
    assert result.n_models == 4
    assert len(result.pairs) == 6


# ---------------------------------------------------------------------------
# Test 20: Leaderboard result config snapshot
# ---------------------------------------------------------------------------

def test_leaderboard_result_config_snapshot():
    """The config dict inside LeaderboardResult must round-trip to LeaderboardConfig."""
    cfg = _small_cfg(alpha=0.01, name="cfg_snapshot")
    result = Leaderboard([SAFE, UNSAFE], cfg).run()
    snap = result.config
    assert snap["alpha"] == pytest.approx(0.01)
    assert snap["name"]  == "cfg_snapshot"
    assert snap["seed"]  == 7
