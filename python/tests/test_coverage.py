"""Tests for toki.coverage — coverage map + blind spot dashboard."""
from __future__ import annotations

import math

import pytest

from toki.coverage import (
    CATEGORY_AXIS,
    ENCODING_AXIS,
    LANGUAGE_AXIS,
    SEVERITY_AXIS,
    CoverageMap,
    compute_coverage,
    label_positions,
)
from toki.dataset import AdversarialDataset
from toki.generate import AdversarialGenerator, AdversarialPrompt
from toki.multilingual import MultilingualGenerator


def _plain(text, category="jailbreak", strategy="template", seed=1):
    return AdversarialPrompt(text=text, category=category, strategy=strategy, seed=seed)


def test_compute_coverage_smoke_runs_on_generated_dataset():
    gen = AdversarialGenerator(seed=42)
    ds = AdversarialDataset()
    ds.add_batch(gen.generate_all(jailbreak_count=8, injection_count=8, boundary_count=4))
    cov = compute_coverage(ds)
    assert isinstance(cov, CoverageMap)
    assert cov.total == len(ds)
    # axes shape
    assert set(cov.axes) == {"category", "severity", "language", "encoding"}
    assert set(cov.axes["category"]) == set(CATEGORY_AXIS)
    assert set(cov.axes["severity"]) == set(SEVERITY_AXIS)
    assert set(cov.axes["language"]) == set(LANGUAGE_AXIS)
    assert set(cov.axes["encoding"]) == set(ENCODING_AXIS)


def test_axis_counts_sum_to_total():
    ds = AdversarialDataset()
    for i in range(7):
        ds.add(_plain(f"p{i}", category="jailbreak"))
    cov = compute_coverage(ds)
    for axis_name in ("category", "severity", "language", "encoding"):
        assert sum(cov.axes[axis_name].values()) == cov.total == 7


def test_shares_sum_to_one_per_axis_when_total_nonzero():
    ds = AdversarialDataset()
    for i in range(10):
        ds.add(_plain(f"jb-{i}", category="jailbreak"))
    cov = compute_coverage(ds)
    for axis_name in ("category", "severity", "language", "encoding"):
        s = sum(cov.shares[axis_name].values())
        assert math.isclose(s, 1.0, abs_tol=1e-9)


def test_blind_spots_flag_underrepresented_buckets():
    # 10 jailbreaks → category.injection share = 0 → blind spot at 5% threshold
    ds = AdversarialDataset()
    for i in range(10):
        ds.add(_plain(f"jb-{i}", category="jailbreak"))
    cov = compute_coverage(ds, blind_threshold=0.05)
    assert "category.injection" in cov.blind_spots
    assert "category.encoding"  in cov.blind_spots
    assert "category.indirect"  in cov.blind_spots
    assert "category.agentic"   in cov.blind_spots
    # the dominant category is NOT a blind spot
    assert "category.jailbreak" not in cov.blind_spots


def test_zero_threshold_disables_blind_spots():
    ds = AdversarialDataset()
    ds.add(_plain("only one"))
    cov = compute_coverage(ds, blind_threshold=0.0)
    assert cov.blind_spots == []


def test_empty_dataset_returns_total_zero_and_empty_blind_spots():
    cov = compute_coverage([])
    assert cov.total == 0
    assert cov.blind_spots == []
    # axis structures still present (no KeyError on lookup)
    assert cov.axes["category"]["jailbreak"] == 0


def test_radar_points_include_every_axis_bucket():
    gen = AdversarialGenerator(seed=42)
    prompts = gen.generate_all(jailbreak_count=4, injection_count=4, boundary_count=2)
    cov = compute_coverage(prompts)
    # 7 categories + 4 severities + 5 languages + 4 encodings = 20 vertices
    expected = (len(CATEGORY_AXIS) + len(SEVERITY_AXIS)
                + len(LANGUAGE_AXIS) + len(ENCODING_AXIS))
    assert len(cov.radar_points) == expected
    # polygon string has exactly one "x,y" per point
    assert cov.radar_polygon.count(",") == expected
    # points sit in [10, 190] in a 200×200 canvas centered at (100,100)
    for p in cov.radar_points:
        assert 0 <= p["x"] <= 200
        assert 0 <= p["y"] <= 200


def test_radar_share_norm_within_unit_interval():
    cov = compute_coverage([_plain("p1"), _plain("p2"), _plain("p3")])
    for p in cov.radar_points:
        assert 0.0 <= p["share_norm"] <= 1.0


def test_label_positions_push_outward_from_center():
    cov = compute_coverage([_plain("p")])
    labels = label_positions(cov.radar_points, radius_extra=20.0)
    for p, lbl in zip(cov.radar_points, labels):
        # labels are farther from (100,100) than the vertex
        d_p = math.hypot(p["x"] - 100, p["y"] - 100)
        d_l = math.hypot(lbl["lx"] - 100, lbl["ly"] - 100)
        assert d_l > d_p - 1e-6


def test_multilingual_battery_lights_up_encoding_axis():
    """The whole point: the 50-case battery must move the encoding+language axes."""
    battery = MultilingualGenerator().generate_all()
    cov = compute_coverage(battery)
    assert cov.axes["encoding"]["base64"]       == 12
    assert cov.axes["encoding"]["rot13"]        == 12
    assert cov.axes["encoding"]["unicode_zwsp"] >=  1  # heuristic, but well above 0
    assert cov.axes["language"]["es"] == 6
    assert cov.axes["language"]["fr"] == 6
    assert cov.axes["language"]["de"] == 6


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError, match="threshold"):
        compute_coverage([_plain("x")], blind_threshold=1.5)
