"""Tests for toki.experiment — TokiExperiment and ExperimentConfig."""
from __future__ import annotations

import pytest
from toki.experiment import TokiExperiment, ExperimentConfig
from toki.dataset import AdversarialDataset
from toki.results import ExperimentResult


def test_experiment_config_defaults():
    cfg = ExperimentConfig()
    assert cfg.name == "toki_experiment"
    assert cfg.model_name == "gpt2"
    assert cfg.seed == 42
    assert cfg.run_finetune is False


def test_experiment_init():
    cfg = ExperimentConfig(name="init_test", seed=7)
    exp = TokiExperiment(cfg)
    assert exp._config.name == "init_test"
    assert exp._config.seed == 7
    assert exp._dataset is None


def test_generate_returns_dataset():
    cfg = ExperimentConfig(jailbreak_count=3, injection_count=3, boundary_count=2)
    exp = TokiExperiment(cfg)
    ds = exp.generate()
    assert isinstance(ds, AdversarialDataset)


def test_generate_populates_dataset():
    cfg = ExperimentConfig(jailbreak_count=5, injection_count=5, boundary_count=2)
    exp = TokiExperiment(cfg)
    ds = exp.generate()
    assert len(ds) > 0
    # Dataset must contain prompts from multiple categories
    cats = ds.categories()
    assert len(cats) >= 2


def test_evaluate_returns_summary_dict():
    cfg = ExperimentConfig(jailbreak_count=3, injection_count=3, boundary_count=2)
    exp = TokiExperiment(cfg)
    exp.generate()
    summary = exp.evaluate()
    assert isinstance(summary, dict)
    assert "mean_score" in summary
    assert "total" in summary
    assert "by_category" in summary


def test_evaluate_before_generate_raises():
    cfg = ExperimentConfig()
    exp = TokiExperiment(cfg)
    with pytest.raises(RuntimeError, match="generate"):
        exp.evaluate()


def test_run_returns_experiment_result(tmp_path):
    cfg = ExperimentConfig(
        name="run_test",
        jailbreak_count=3,
        injection_count=3,
        boundary_count=2,
        output_dir=str(tmp_path),
        run_finetune=False,
    )
    exp = TokiExperiment(cfg)
    result = exp.run()
    assert isinstance(result, ExperimentResult)
    assert result.name == "run_test"
    assert 0.0 <= result.pre_score <= 1.0
    assert result.total_prompts > 0


def test_run_without_finetune_post_score_is_none(tmp_path):
    cfg = ExperimentConfig(
        name="no_finetune",
        jailbreak_count=3,
        injection_count=3,
        boundary_count=2,
        output_dir=str(tmp_path),
        run_finetune=False,
    )
    exp = TokiExperiment(cfg)
    result = exp.run()
    assert result.post_score is None
    assert result.improvement is None
