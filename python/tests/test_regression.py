"""Tests for toki.regression — safety regression CI gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from toki.regression import (
    Baseline,
    CategoryDelta,
    RegressionReport,
    compare,
)


SUMMARY_A = {
    "mean_score": 0.85,
    "by_category": {"jailbreak": 0.90, "injection": 0.80, "boundary": 0.85},
}
SUMMARY_B_SAME = {
    "mean_score": 0.85,
    "by_category": {"jailbreak": 0.90, "injection": 0.80, "boundary": 0.85},
}
SUMMARY_B_REGRESSED = {
    "mean_score": 0.75,
    "by_category": {"jailbreak": 0.70, "injection": 0.80, "boundary": 0.85},
}
SUMMARY_B_IMPROVED = {
    "mean_score": 0.92,
    "by_category": {"jailbreak": 0.95, "injection": 0.90, "boundary": 0.91},
}


def test_baseline_from_summary_populates_per_category_and_overall():
    b = Baseline.from_summary(SUMMARY_A)
    assert b.overall == pytest.approx(0.85)
    assert b.per_category == {"jailbreak": 0.90, "injection": 0.80, "boundary": 0.85}
    assert b.created.endswith("Z")


def test_baseline_save_load_round_trip(tmp_path: Path):
    b = Baseline.from_summary(SUMMARY_A, meta={"seed": 42, "model": "demo"})
    p = b.save(tmp_path / "baseline.json")
    assert p.exists()
    loaded = Baseline.load(p)
    assert loaded.overall == pytest.approx(b.overall)
    assert loaded.per_category == b.per_category
    assert loaded.meta == b.meta


def test_compare_identical_summaries_passes():
    rep = compare(Baseline.from_summary(SUMMARY_A), SUMMARY_B_SAME, tolerance=0.02)
    assert rep.failed is False
    assert rep.exit_code() == 0
    assert rep.regressed == []
    assert rep.overall_delta == pytest.approx(0.0)


def test_compare_detects_category_regression():
    rep = compare(Baseline.from_summary(SUMMARY_A), SUMMARY_B_REGRESSED, tolerance=0.02)
    assert rep.failed is True
    assert rep.exit_code() == 1
    assert len(rep.regressed) == 1
    assert rep.regressed[0].category == "jailbreak"
    assert rep.regressed[0].delta == pytest.approx(-0.20)
    assert rep.worst_delta.category == "jailbreak"


def test_compare_tolerance_absorbs_small_drops():
    summary = {"mean_score": 0.84, "by_category": {"jailbreak": 0.89, "injection": 0.79}}
    rep = compare(Baseline.from_summary(SUMMARY_A), summary, tolerance=0.02)
    # both deltas are -0.01 which is within tolerance
    assert rep.regressed == []
    assert rep.failed is False


def test_compare_classifies_improvements_separately():
    rep = compare(Baseline.from_summary(SUMMARY_A), SUMMARY_B_IMPROVED, tolerance=0.02)
    assert rep.failed is False
    cats = [d.category for d in rep.improved]
    assert "jailbreak" in cats
    assert "injection" in cats
    assert "boundary" in cats


def test_compare_reports_missing_and_new_categories():
    cur = {"mean_score": 0.8, "by_category": {"jailbreak": 0.90, "newcat": 0.7}}
    rep = compare(Baseline.from_summary(SUMMARY_A), cur, tolerance=0.02)
    assert "injection" in rep.missing_from_current
    assert "boundary"  in rep.missing_from_current
    assert "newcat"    in rep.new_in_current


def test_compare_invalid_tolerance_raises():
    with pytest.raises(ValueError):
        compare(Baseline.from_summary(SUMMARY_A), SUMMARY_A, tolerance=-0.1)
    with pytest.raises(ValueError):
        compare(Baseline.from_summary(SUMMARY_A), SUMMARY_A, tolerance=1.5)


def test_classmethod_form_available():
    """PLAN.md documents `RegressionReport.compare(...)` as the fluent form."""
    rep = RegressionReport.compare(
        Baseline.from_summary(SUMMARY_A), SUMMARY_B_REGRESSED, tolerance=0.02
    )
    assert isinstance(rep, RegressionReport)
    assert rep.failed is True


def test_report_as_dict_is_serializable():
    rep = compare(Baseline.from_summary(SUMMARY_A), SUMMARY_B_REGRESSED, tolerance=0.02)
    payload = rep.as_dict()
    s = json.dumps(payload)
    assert "regressed" in s
    assert "jailbreak" in s


def test_report_to_markdown_includes_table_for_regressions():
    rep = compare(Baseline.from_summary(SUMMARY_A), SUMMARY_B_REGRESSED, tolerance=0.02)
    md = rep.to_markdown()
    assert "FAILED" in md
    assert "Regressions" in md
    assert "jailbreak" in md
    assert "-0.2000" in md or "-0.20" in md
