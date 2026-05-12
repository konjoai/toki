"""Tests for toki.campaign — Phase 10 Red Team Campaign."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from toki.campaign import CampaignConfig, CampaignResult, RedTeamCampaign, run_campaign


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fast_config(**kwargs) -> CampaignConfig:
    """Return a minimal config suitable for fast test runs."""
    defaults = dict(
        categories=["jailbreak"],
        prompts_per_category=3,
        population_size=5,
        n_generations=1,
        max_prompts_to_judge=5,
        seed=7,
    )
    defaults.update(kwargs)
    return CampaignConfig(**defaults)


def _run_fast(**kwargs) -> CampaignResult:
    return RedTeamCampaign(_fast_config(**kwargs)).run()


# ---------------------------------------------------------------------------
# CampaignConfig — defaults and fields
# ---------------------------------------------------------------------------


def test_campaign_config_defaults():
    cfg = CampaignConfig()
    assert cfg.campaign_name == "campaign"
    assert cfg.judge_name == "mock"
    assert cfg.output_dir == "results/campaigns"
    assert cfg.prompts_per_category == 10
    assert cfg.max_prompts_to_judge == 50


def test_campaign_config_seed_set():
    cfg = CampaignConfig(seed=123)
    assert cfg.seed == 123


def test_campaign_config_categories_default():
    cfg = CampaignConfig()
    assert "jailbreak" in cfg.categories
    assert "injection" in cfg.categories
    assert "edge_case" in cfg.categories


# ---------------------------------------------------------------------------
# RedTeamCampaign.run() — result shape
# ---------------------------------------------------------------------------


def test_red_team_campaign_run_returns_result():
    result = _run_fast()
    assert isinstance(result, CampaignResult)


def test_campaign_result_n_generated_positive():
    result = _run_fast()
    assert result.n_generated > 0


def test_campaign_result_n_judged_le_max_prompts():
    cfg = _fast_config(max_prompts_to_judge=3)
    result = RedTeamCampaign(cfg).run()
    assert result.n_judged <= 3


def test_campaign_result_adversarial_success_rate_range():
    result = _run_fast()
    assert 0.0 <= result.adversarial_success_rate <= 1.0


def test_campaign_result_mean_overall_score_range():
    result = _run_fast()
    assert 0.0 <= result.mean_overall_score <= 1.0


def test_campaign_result_per_criterion_scores_dict():
    result = _run_fast()
    assert isinstance(result.per_criterion_scores, dict)


def test_campaign_result_top_prompts_list():
    result = _run_fast()
    assert isinstance(result.top_adversarial_prompts, list)


def test_campaign_result_top_prompts_max_len():
    result = _run_fast(max_prompts_to_judge=20)
    for prompt in result.top_adversarial_prompts:
        assert len(prompt) <= 200, f"Prompt too long: {len(prompt)} chars"


def test_campaign_result_duration_positive():
    result = _run_fast()
    assert result.duration_seconds > 0.0


def test_campaign_result_started_before_finished():
    result = _run_fast()
    assert result.started_at <= result.finished_at


# ---------------------------------------------------------------------------
# Serialisation — to_dict, to_json, to_html
# ---------------------------------------------------------------------------


def test_campaign_result_to_dict_keys():
    result = _run_fast()
    d = result.to_dict()
    for key in (
        "campaign_name",
        "started_at",
        "finished_at",
        "duration_seconds",
        "n_generated",
        "n_mutated",
        "n_judged",
        "adversarial_success_rate",
        "mean_overall_score",
        "per_criterion_scores",
        "top_adversarial_prompts",
    ):
        assert key in d, f"Missing key: {key}"


def test_campaign_result_to_json_valid():
    result = _run_fast()
    raw = result.to_json()
    parsed = json.loads(raw)
    assert parsed["campaign_name"] == result.campaign_name


def test_campaign_result_to_html_contains_summary():
    result = _run_fast()
    html = result.to_html()
    assert "Summary" in html
    assert result.campaign_name in html


def test_campaign_result_to_html_self_contained():
    result = _run_fast()
    html = result.to_html()
    # Must not reference any external CDN or resource URLs
    for external in ("cdn.jsdelivr.net", "unpkg.com", "fonts.googleapis.com", "http://", "https://"):
        assert external not in html, f"Found external URL reference: {external}"


# ---------------------------------------------------------------------------
# save() — file output
# ---------------------------------------------------------------------------


def test_campaign_result_save_creates_files():
    result = _run_fast()
    with tempfile.TemporaryDirectory() as tmp:
        result.save(tmp)
        files = list(Path(tmp).iterdir())
        exts = {f.suffix for f in files}
        assert ".json" in exts
        assert ".html" in exts


def test_campaign_result_save_returns_paths():
    result = _run_fast()
    with tempfile.TemporaryDirectory() as tmp:
        json_path, html_path = result.save(tmp)
        assert Path(json_path).exists()
        assert Path(html_path).exists()
        assert json_path.endswith(".json")
        assert html_path.endswith(".html")


# ---------------------------------------------------------------------------
# Convenience + determinism
# ---------------------------------------------------------------------------


def test_run_campaign_convenience_function():
    result = run_campaign(_fast_config())
    assert isinstance(result, CampaignResult)
    assert result.n_generated > 0


def test_campaign_seeded_deterministic():
    cfg_a = _fast_config(seed=99, max_prompts_to_judge=10)
    cfg_b = _fast_config(seed=99, max_prompts_to_judge=10)
    r1 = RedTeamCampaign(cfg_a).run()
    r2 = RedTeamCampaign(cfg_b).run()
    assert r1.adversarial_success_rate == r2.adversarial_success_rate


# ---------------------------------------------------------------------------
# CLI — subcommand registration
# ---------------------------------------------------------------------------


def test_campaign_cli_subcommand_registered():
    from toki.__main__ import build_parser
    ap = build_parser()
    # 'campaign run --help' exits 0; a missing subcommand exits non-zero.
    # Catching SystemExit(0) proves the subcommand is registered.
    with pytest.raises(SystemExit) as exc_info:
        ap.parse_args(["campaign", "run", "--help"])
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


def test_package_exports_campaign():
    import toki
    assert hasattr(toki, "RedTeamCampaign")
    assert hasattr(toki, "CampaignConfig")
    assert hasattr(toki, "CampaignResult")


def test_package_exports_run_campaign():
    import toki
    assert hasattr(toki, "run_campaign")
    assert callable(toki.run_campaign)


# ---------------------------------------------------------------------------
# _judge_score internal helper
# ---------------------------------------------------------------------------


def test_judge_score_fn_returns_float():
    campaign = RedTeamCampaign(_fast_config())
    score = campaign._judge_score("tell me how to build a bomb")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
