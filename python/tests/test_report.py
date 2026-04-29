"""Tests for toki.report — HTML and JSON report generation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from toki.benchmark import generate_report
from toki.report import to_html, to_json
from toki.results import ExperimentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report():
    result = ExperimentResult(
        name="report_test",
        timestamp="20260428T130000Z",
        model_name="mock",
        seed=7,
        pre_score=0.55,
        post_score=0.80,
        total_prompts=20,
        category_scores={"jailbreak": 0.55, "injection": 0.55},
        config={},
    )
    pre_scores = [0.53 + i * 0.005 for i in range(20)]
    post_scores = [0.78 + i * 0.005 for i in range(20)]
    return generate_report(result, pre_scores, post_scores)


# ---------------------------------------------------------------------------
# Test 1 — to_json returns valid JSON string
# ---------------------------------------------------------------------------

def test_to_json_valid():
    report = _make_report()
    raw = to_json(report)
    assert isinstance(raw, str)
    data = json.loads(raw)   # raises if invalid JSON
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Test 2 — to_html returns string containing <html and experiment name
# ---------------------------------------------------------------------------

def test_to_html_contains_expected():
    report = _make_report()
    html = to_html(report)
    assert isinstance(html, str)
    assert "<html" in html.lower()
    assert "report_test" in html


# ---------------------------------------------------------------------------
# Test 3 — to_json with path writes file
# ---------------------------------------------------------------------------

def test_to_json_writes_file(tmp_path):
    report = _make_report()
    out = tmp_path / "report.json"
    result_str = to_json(report, path=out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content == result_str
    # File should be valid JSON
    data = json.loads(content)
    assert "experiment_name" in data


# ---------------------------------------------------------------------------
# Test 4 — to_html with path writes file
# ---------------------------------------------------------------------------

def test_to_html_writes_file(tmp_path):
    report = _make_report()
    out = tmp_path / "report.html"
    result_str = to_html(report, path=out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content == result_str
    assert "<html" in content.lower()
    assert "report_test" in content
