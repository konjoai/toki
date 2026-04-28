"""Tests for toki.results — ExperimentResult and list_experiments."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from toki.results import ExperimentResult, list_experiments


def _make_result(**kwargs) -> ExperimentResult:
    defaults = dict(
        name="test_exp",
        timestamp="20240101T000000Z",
        model_name="mock",
        seed=42,
        pre_score=0.9,
        post_score=None,
        total_prompts=25,
        category_scores={"jailbreak": 0.9, "injection": 0.9},
        config={"seed": 42},
    )
    defaults.update(kwargs)
    return ExperimentResult(**defaults)


def test_experiment_result_creates():
    r = _make_result()
    assert r.name == "test_exp"
    assert r.model_name == "mock"
    assert r.pre_score == 0.9
    assert r.post_score is None


def test_improvement_with_post_score():
    r = _make_result(pre_score=0.8, post_score=0.95)
    assert r.improvement is not None
    assert abs(r.improvement - 0.15) < 1e-9


def test_improvement_without_post_score_is_none():
    r = _make_result(post_score=None)
    assert r.improvement is None


def test_save_creates_file(tmp_path):
    r = _make_result(name="save_test", timestamp="20240101T120000Z")
    out = r.save(base_dir=str(tmp_path))
    assert out.exists()
    assert out.name == "result.json"
    data = json.loads(out.read_text())
    assert data["name"] == "save_test"


def test_load_roundtrip(tmp_path):
    original = _make_result(
        name="roundtrip",
        timestamp="20240202T030405Z",
        pre_score=0.87,
        post_score=0.93,
        total_prompts=30,
        category_scores={"jailbreak": 0.85, "boundary": 0.9},
        config={"seed": 7, "run_finetune": True},
    )
    out = original.save(base_dir=str(tmp_path))
    loaded = ExperimentResult.load(out)
    assert loaded.name == original.name
    assert loaded.timestamp == original.timestamp
    assert loaded.pre_score == original.pre_score
    assert loaded.post_score == original.post_score
    assert loaded.total_prompts == original.total_prompts
    assert loaded.category_scores == original.category_scores
    assert loaded.config == original.config


def test_make_timestamp_format():
    ts = ExperimentResult.make_timestamp()
    # Must match YYYYMMDDTHHMMSSZ
    assert re.fullmatch(r"\d{8}T\d{6}Z", ts), f"Unexpected timestamp format: {ts!r}"


def test_list_experiments_empty_dir(tmp_path):
    results = list_experiments(str(tmp_path))
    assert results == []


def test_list_experiments_missing_dir(tmp_path):
    nonexistent = str(tmp_path / "no_such_dir")
    results = list_experiments(nonexistent)
    assert results == []


def test_list_experiments_finds_files(tmp_path):
    r1 = _make_result(name="exp_a", timestamp="20240101T000000Z")
    r2 = _make_result(name="exp_b", timestamp="20240102T000000Z")
    r1.save(base_dir=str(tmp_path))
    r2.save(base_dir=str(tmp_path))
    found = list_experiments(str(tmp_path))
    assert len(found) == 2
    names_found = {json.loads(p.read_text())["name"] for p in found}
    assert names_found == {"exp_a", "exp_b"}
