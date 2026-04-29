"""Tests for toki.benchmark — statistical analysis module."""
from __future__ import annotations

import math
import pytest

from toki.benchmark import (
    BenchmarkReport,
    BenchmarkStats,
    StatTestResult,
    compute_stats,
    generate_report,
    paired_t_test,
    wilcoxon_test,
)
from toki.results import ExperimentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(pre: float = 0.6, post: float | None = 0.85) -> ExperimentResult:
    return ExperimentResult(
        name="test_exp",
        timestamp="20260428T120000Z",
        model_name="mock",
        seed=42,
        pre_score=pre,
        post_score=post,
        total_prompts=20,
        category_scores={"jailbreak": pre, "injection": pre},
        config={},
    )


# ---------------------------------------------------------------------------
# Test 1 — compute_stats on known list
# ---------------------------------------------------------------------------

def test_compute_stats_ordering():
    scores = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0]
    stats = compute_stats(scores)
    assert isinstance(stats, BenchmarkStats)
    assert stats.p50 <= stats.p95 <= stats.p99
    assert stats.min <= stats.mean <= stats.max
    assert stats.n == 10
    assert stats.min == pytest.approx(0.1)
    assert stats.max == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 2 — paired_t_test: identical before/after → not significant
# ---------------------------------------------------------------------------

def test_paired_t_test_identical():
    scores = [0.5] * 20
    result = paired_t_test(scores, scores[:])
    assert isinstance(result, StatTestResult)
    assert result.test_name == "paired_t_test"
    assert result.p_value >= 0.9
    assert result.significant is False


# ---------------------------------------------------------------------------
# Test 3 — paired_t_test: clearly different → significant
# ---------------------------------------------------------------------------

def test_paired_t_test_significant():
    before = [0.3] * 20
    after = [0.9] * 20
    result = paired_t_test(before, after)
    assert result.significant is True
    assert result.p_value < 0.05


# ---------------------------------------------------------------------------
# Test 4 — wilcoxon_test: identical lists → not significant
# ---------------------------------------------------------------------------

def test_wilcoxon_identical():
    scores = [0.5] * 20
    result = wilcoxon_test(scores, scores[:])
    assert result.test_name == "wilcoxon"
    assert result.significant is False


# ---------------------------------------------------------------------------
# Test 5 — wilcoxon_test: clearly different lists → significant
# ---------------------------------------------------------------------------

def test_wilcoxon_significant():
    before = [0.2, 0.25, 0.3, 0.2, 0.25, 0.3, 0.22, 0.28, 0.24, 0.26,
              0.2, 0.25, 0.3, 0.2, 0.25, 0.3, 0.22, 0.28, 0.24, 0.26]
    after  = [0.8, 0.85, 0.9, 0.8, 0.85, 0.9, 0.82, 0.88, 0.84, 0.86,
              0.8, 0.85, 0.9, 0.8, 0.85, 0.9, 0.82, 0.88, 0.84, 0.86]
    result = wilcoxon_test(before, after)
    assert result.significant is True
    assert result.p_value < 0.05


# ---------------------------------------------------------------------------
# Test 6 — generate_report with only pre_scores → post/tests are None
# ---------------------------------------------------------------------------

def test_generate_report_pre_only():
    result = _make_result(pre=0.6, post=None)
    pre_scores = [0.55, 0.6, 0.65, 0.58, 0.62] * 4
    report = generate_report(result, pre_scores)
    assert isinstance(report, BenchmarkReport)
    assert report.post_stats is None
    assert report.t_test is None
    assert report.wilcoxon is None
    assert report.score_delta is None
    assert report.experiment_name == "test_exp"


# ---------------------------------------------------------------------------
# Test 7 — generate_report with both pre and post → all fields populated
# ---------------------------------------------------------------------------

def test_generate_report_full():
    result = _make_result(pre=0.4, post=0.8)
    pre_scores = [0.38 + i * 0.01 for i in range(20)]
    post_scores = [0.78 + i * 0.01 for i in range(20)]
    report = generate_report(result, pre_scores, post_scores)
    assert report.post_stats is not None
    assert report.t_test is not None
    assert report.wilcoxon is not None
    assert report.score_delta is not None
    assert report.score_delta > 0


# ---------------------------------------------------------------------------
# Test 8 — to_json round-trip has required keys
# ---------------------------------------------------------------------------

def test_to_json_roundtrip():
    import json
    from toki.report import to_json

    result = _make_result(pre=0.5, post=0.75)
    pre_scores = [0.5] * 15
    post_scores = [0.75] * 15
    report = generate_report(result, pre_scores, post_scores)

    raw = to_json(report)
    data = json.loads(raw)
    assert "experiment_name" in data
    assert data["experiment_name"] == "test_exp"
    assert "pre_stats" in data
    assert "post_stats" in data
    assert "t_test" in data
    assert "wilcoxon" in data
    assert "score_delta" in data
