"""Tests for toki.compliance — compliance certification report."""

from __future__ import annotations

import json

import pytest

from toki.compliance import (
    COVERED,
    GAP,
    PARTIAL,
    ComplianceReport,
    Control,
    ControlStatus,
    Framework,
    assess_compliance,
    compliance_from_dataset,
    count_categories,
    get_catalog,
)
from toki.generate import AdversarialPrompt


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------


def test_all_frameworks_have_catalogs():
    for fw in Framework:
        catalog = get_catalog(fw)
        assert len(catalog) >= 3
        assert all(isinstance(c, Control) for c in catalog)


def test_get_catalog_by_string():
    assert get_catalog("nist_ai_rmf") == get_catalog(Framework.NIST_AI_RMF)


def test_get_catalog_unknown_raises():
    with pytest.raises(ValueError):
        get_catalog("sox")


def test_catalog_categories_are_known_toki_categories():
    known = {
        "jailbreak",
        "injection",
        "edge_case",
        "boundary",
        "encoding",
        "indirect",
        "agentic",
        "multiturn",
    }
    for fw in Framework:
        for control in get_catalog(fw):
            assert set(control.categories) <= known, control.control_id


# ---------------------------------------------------------------------------
# Per-control assessment
# ---------------------------------------------------------------------------


def test_full_coverage_certifies():
    counts = {
        "jailbreak": 5,
        "injection": 5,
        "edge_case": 5,
        "boundary": 5,
        "encoding": 5,
        "indirect": 5,
        "agentic": 5,
        "multiturn": 5,
    }
    report = assess_compliance(Framework.NIST_AI_RMF, counts)
    assert report.certified is True
    assert report.n_gap == 0
    assert report.coverage_score == 1.0
    assert all(c.status == COVERED for c in report.controls)


def test_empty_coverage_is_all_gaps():
    report = assess_compliance(Framework.NIST_AI_RMF, {})
    assert report.certified is False
    assert report.n_gap == report.n_controls
    assert report.coverage_score == 0.0
    assert all(c.status == GAP for c in report.controls)


def test_partial_control_detected():
    # MEASURE-2.5 maps to jailbreak+injection+encoding; give only two of three
    counts = {"jailbreak": 3, "injection": 3}
    report = assess_compliance(Framework.NIST_AI_RMF, counts)
    m25 = next(c for c in report.controls if c.control_id == "MEASURE-2.5")
    assert m25.status == PARTIAL
    assert "encoding" in m25.missing_categories
    assert set(m25.evidence_categories) == {"jailbreak", "injection"}


def test_min_tests_threshold_enforced():
    counts = {"boundary": 1, "edge_case": 1}
    # MEASURE-2.2 maps to boundary+edge_case
    lax = assess_compliance(Framework.NIST_AI_RMF, counts, min_tests=1)
    strict = assess_compliance(Framework.NIST_AI_RMF, counts, min_tests=2)
    m22_lax = next(c for c in lax.controls if c.control_id == "MEASURE-2.2")
    m22_strict = next(c for c in strict.controls if c.control_id == "MEASURE-2.2")
    assert m22_lax.status == COVERED
    assert m22_strict.status == GAP


def test_min_tests_below_one_raises():
    with pytest.raises(ValueError):
        assess_compliance(Framework.NIST_AI_RMF, {}, min_tests=0)


def test_coverage_score_counts_partial_as_half():
    # exactly one control fully covered, rest gap, on OWASP (8 controls)
    counts = {"agentic": 5}  # ASI01 + ASI04 fully covered (agentic-only controls)
    report = assess_compliance(Framework.OWASP_AGENTIC, counts)
    # ASI01 and ASI04 are agentic-only -> covered; ASI05 is multiturn+indirect -> gap
    covered_ids = {c.control_id for c in report.controls if c.status == COVERED}
    assert {"ASI01", "ASI04"} <= covered_ids


def test_test_count_aggregated_across_categories():
    counts = {"jailbreak": 2, "injection": 3, "encoding": 4}
    report = assess_compliance(Framework.NIST_AI_RMF, counts)
    m25 = next(c for c in report.controls if c.control_id == "MEASURE-2.5")
    assert m25.test_count == 9


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_is_deterministic():
    counts = {"jailbreak": 1, "injection": 1}
    a = assess_compliance(Framework.ISO_42001, counts)
    b = assess_compliance(Framework.ISO_42001, counts)
    assert a.manifest_sha256 == b.manifest_sha256
    assert len(a.manifest_sha256) == 64


def test_manifest_changes_with_evidence():
    a = assess_compliance(Framework.ISO_42001, {"jailbreak": 1})
    b = assess_compliance(Framework.ISO_42001, {"jailbreak": 9})
    assert a.manifest_sha256 != b.manifest_sha256


# ---------------------------------------------------------------------------
# count_categories / dataset
# ---------------------------------------------------------------------------


def _prompt(cat):
    return AdversarialPrompt(text="x", category=cat, strategy="t", seed=1)


def test_count_categories_tallies():
    prompts = [_prompt("jailbreak"), _prompt("jailbreak"), _prompt("injection")]
    assert count_categories(prompts) == {"jailbreak": 2, "injection": 1}


def test_count_categories_missing_category_bucketed_unknown():
    class Bare:
        category = None

    assert count_categories([Bare()]) == {"unknown": 1}


def test_compliance_from_dataset():
    prompts = [_prompt("jailbreak"), _prompt("injection"), _prompt("encoding")]
    report = compliance_from_dataset(Framework.NIST_AI_RMF, prompts)
    m25 = next(c for c in report.controls if c.control_id == "MEASURE-2.5")
    assert m25.status == COVERED


# ---------------------------------------------------------------------------
# Serialization + persistence
# ---------------------------------------------------------------------------


def test_to_json_valid():
    report = assess_compliance(Framework.EU_AI_ACT, {"jailbreak": 1})
    data = json.loads(report.to_json())
    assert data["framework"] == "eu_ai_act"
    assert isinstance(data["controls"], list)


def test_to_markdown_contains_badge_and_manifest():
    report = assess_compliance(Framework.NIST_AI_RMF, {})
    md = report.to_markdown()
    assert "GAPS PRESENT" in md
    assert report.manifest_sha256 in md


def test_to_html_is_self_contained():
    report = assess_compliance(Framework.NIST_AI_RMF, {"jailbreak": 1})
    html = report.to_html()
    assert html.startswith("<!doctype html>")
    assert "http://" not in html  # no external CDN


def test_save_and_load_roundtrip(tmp_path):
    report = assess_compliance(Framework.OWASP_AGENTIC, {"agentic": 2})
    json_path, html_path = report.save(str(tmp_path))
    assert json_path.exists() and html_path.exists()

    loaded = ComplianceReport.load(json_path)
    assert loaded.framework == report.framework
    assert loaded.manifest_sha256 == report.manifest_sha256
    assert all(isinstance(c, ControlStatus) for c in loaded.controls)
    assert loaded.controls == report.controls


def test_report_carries_toki_version():
    report = assess_compliance(Framework.NIST_AI_RMF, {"jailbreak": 1})
    assert report.toki_version and report.toki_version != "unknown"
