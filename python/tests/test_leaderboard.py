"""Tests for toki.leaderboard — persistent SQLite-backed leaderboard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from toki.leaderboard import (
    KNOWN_SUITES,
    Leaderboard,
    LeaderboardEntry,
    load_seed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lb(tmp_path: Path) -> Leaderboard:
    return Leaderboard(tmp_path / "lb.db")


def _entry(model: str, suite: str, score: float, **kw) -> LeaderboardEntry:
    return LeaderboardEntry(
        model_name=model,
        suite=suite,
        pass_rate=kw.get("pass_rate", score),
        robustness_score=score,
        timestamp=kw.get("timestamp", ""),
        notes=kw.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# Test 1: schema auto-creation + empty reads
# ---------------------------------------------------------------------------

def test_fresh_db_is_empty(lb: Leaderboard):
    """Schema is created on first use; an empty DB returns nothing."""
    assert lb.count() == 0
    assert lb.top_n("adversarial") == []
    assert lb.history("any-model") == []


# ---------------------------------------------------------------------------
# Test 2: record() round-trips a row and stamps an id
# ---------------------------------------------------------------------------

def test_record_round_trip(lb: Leaderboard):
    e = _entry("phi-3", "adversarial", 0.87)
    out = lb.record(e)

    assert out.id is not None and out.id > 0
    assert lb.count() == 1
    rows = lb.all()
    assert len(rows) == 1
    assert rows[0].model_name == "phi-3"
    assert rows[0].suite == "adversarial"
    assert rows[0].robustness_score == pytest.approx(0.87)
    assert rows[0].timestamp != ""           # auto-stamped


# ---------------------------------------------------------------------------
# Test 3: top_n is sorted by robustness_score DESC and capped at n
# ---------------------------------------------------------------------------

def test_top_n_sorted_and_capped(lb: Leaderboard):
    for s in [0.10, 0.95, 0.50, 0.80, 0.30]:
        lb.record(_entry(f"m_{s}", "adversarial", s))

    top3 = lb.top_n("adversarial", n=3)
    assert len(top3) == 3
    scores = [e.robustness_score for e in top3]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Test 4: top_n filters by suite
# ---------------------------------------------------------------------------

def test_top_n_filters_by_suite(lb: Leaderboard):
    lb.record(_entry("m1", "adversarial", 0.99))
    lb.record(_entry("m2", "paraphrase",  0.90))
    lb.record(_entry("m3", "noise",       0.80))

    adv = lb.top_n("adversarial", n=10)
    para = lb.top_n("paraphrase", n=10)
    noi = lb.top_n("noise", n=10)
    assert [e.model_name for e in adv]  == ["m1"]
    assert [e.model_name for e in para] == ["m2"]
    assert [e.model_name for e in noi]  == ["m3"]


# ---------------------------------------------------------------------------
# Test 5: top_n("all") drops the suite filter
# ---------------------------------------------------------------------------

def test_top_n_all_returns_global_ranking(lb: Leaderboard):
    lb.record(_entry("m1", "adversarial", 0.40))
    lb.record(_entry("m2", "paraphrase",  0.95))
    lb.record(_entry("m3", "noise",       0.70))

    rows = lb.top_n("all", n=10)
    assert [e.model_name for e in rows] == ["m2", "m3", "m1"]


# ---------------------------------------------------------------------------
# Test 6: history is chronological per model
# ---------------------------------------------------------------------------

def test_history_chronological(lb: Leaderboard):
    lb.record(_entry("phi-3", "adversarial", 0.5, timestamp="2026-01-01T00:00:00+00:00"))
    lb.record(_entry("phi-3", "adversarial", 0.7, timestamp="2026-02-01T00:00:00+00:00"))
    lb.record(_entry("phi-3", "noise",       0.6, timestamp="2026-01-15T00:00:00+00:00"))
    lb.record(_entry("other", "adversarial", 0.9, timestamp="2026-03-01T00:00:00+00:00"))

    hist = lb.history("phi-3")
    assert [e.timestamp for e in hist] == [
        "2026-01-01T00:00:00+00:00",
        "2026-01-15T00:00:00+00:00",
        "2026-02-01T00:00:00+00:00",
    ]
    # Confirm no leakage from "other"
    assert all(e.model_name == "phi-3" for e in hist)


# ---------------------------------------------------------------------------
# Test 7: compare() picks the latest-per-suite for each model + names a winner
# ---------------------------------------------------------------------------

def test_compare_uses_latest_per_suite(lb: Leaderboard):
    # phi-3 — older-then-newer adversarial; only-paraphrase
    lb.record(_entry("phi-3", "adversarial", 0.40, timestamp="2026-01-01T00:00:00+00:00"))
    lb.record(_entry("phi-3", "adversarial", 0.90, timestamp="2026-02-01T00:00:00+00:00"))
    lb.record(_entry("phi-3", "paraphrase",  0.85, timestamp="2026-02-01T00:00:00+00:00"))
    # qwen — only-adversarial (lower)
    lb.record(_entry("qwen",  "adversarial", 0.60, timestamp="2026-02-15T00:00:00+00:00"))

    diff = lb.compare("phi-3", "qwen")

    adv = diff["by_suite"]["adversarial"]
    assert adv["a"]["robustness_score"] == pytest.approx(0.90)   # latest, not 0.40
    assert adv["b"]["robustness_score"] == pytest.approx(0.60)
    assert adv["delta"] == pytest.approx(0.30)

    # paraphrase only on phi-3 → b is None and delta is None
    para = diff["by_suite"]["paraphrase"]
    assert para["a"] is not None
    assert para["b"] is None
    assert para["delta"] is None

    # winner is the model with higher mean over overlapping suites (only adversarial)
    assert diff["winner"] == "phi-3"


# ---------------------------------------------------------------------------
# Test 8: validation — score outside [0, 1] is rejected
# ---------------------------------------------------------------------------

def test_score_out_of_range_rejected():
    with pytest.raises(ValueError, match="robustness_score"):
        LeaderboardEntry("m", "adversarial", pass_rate=0.5, robustness_score=1.5)
    with pytest.raises(ValueError, match="pass_rate"):
        LeaderboardEntry("m", "adversarial", pass_rate=-0.01, robustness_score=0.5)
    with pytest.raises(ValueError, match="model_name"):
        LeaderboardEntry("", "adversarial", pass_rate=0.5, robustness_score=0.5)


# ---------------------------------------------------------------------------
# Test 9: load_seed inserts every entry from a JSON file
# ---------------------------------------------------------------------------

def test_load_seed(tmp_path: Path, lb: Leaderboard):
    seed = [
        {"model_name": "phi-3",  "suite": "adversarial", "pass_rate": 0.91,
         "robustness_score": 0.87, "timestamp": "2026-04-01T00:00:00+00:00",
         "notes": "phase-9 baseline"},
        {"model_name": "qwen",   "suite": "noise",       "pass_rate": 0.55,
         "robustness_score": 0.61, "timestamp": "2026-04-02T00:00:00+00:00"},
    ]
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps(seed))
    n = load_seed(lb, seed_path)
    assert n == 2
    assert lb.count() == 2
    # Ensure notes round-trip even when omitted
    qwen = lb.history("qwen")[0]
    assert qwen.notes == ""


# ---------------------------------------------------------------------------
# Test 10: persistence across instances + KNOWN_SUITES surface check
# ---------------------------------------------------------------------------

def test_persists_across_instances_and_known_suites(tmp_path: Path):
    db = tmp_path / "persist.db"
    lb1 = Leaderboard(db)
    lb1.record(_entry("phi-3", "adversarial", 0.8))
    lb1.close()

    lb2 = Leaderboard(db)
    rows = lb2.all()
    assert len(rows) == 1
    assert rows[0].model_name == "phi-3"
    lb2.close()

    # The known-suite tuple is the public contract for the API/UI tabs
    assert KNOWN_SUITES == ("adversarial", "paraphrase", "noise")
