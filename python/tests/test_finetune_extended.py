"""Tests for extended LoRAConfig, LoRATrainResult, and LoRAFinetuner (Sprint 17)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from toki.finetune import LoRAConfig, LoRAFinetuner, TrainingConfig
from toki.safety_lora import LoRATrainResult, SploraAuditResult


# ---------------------------------------------------------------------------
# LoRAConfig — new safety-subspace fields (Sprint 17)
# ---------------------------------------------------------------------------


def test_lora_config_safety_lora_rank_default():
    cfg = LoRAConfig()
    assert cfg.safety_lora_rank == 0


def test_lora_config_safety_subspace_path_default():
    cfg = LoRAConfig()
    assert cfg.safety_subspace_path is None


def test_lora_config_enable_splora_audit_default():
    cfg = LoRAConfig()
    assert cfg.enable_splora_audit is False


def test_lora_config_splora_threshold_default():
    cfg = LoRAConfig()
    assert cfg.splora_threshold == pytest.approx(0.15)


def test_lora_config_safety_fields_set():
    cfg = LoRAConfig(
        safety_lora_rank=1,
        safety_subspace_path="/tmp/safety.pt",
        enable_splora_audit=True,
        splora_threshold=0.25,
    )
    assert cfg.safety_lora_rank == 1
    assert cfg.safety_subspace_path == "/tmp/safety.pt"
    assert cfg.enable_splora_audit is True
    assert cfg.splora_threshold == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# LoRAConfig — base fields unchanged
# ---------------------------------------------------------------------------


def test_lora_config_base_fields_unchanged():
    cfg = LoRAConfig()
    assert cfg.r == 8
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == pytest.approx(0.1)
    assert cfg.target_modules == ["q_proj", "v_proj"]
    assert cfg.bias == "none"


def test_lora_config_independent_instances():
    a = LoRAConfig(r=4)
    b = LoRAConfig(r=16)
    assert a.r != b.r
    assert a.safety_lora_rank == b.safety_lora_rank


# ---------------------------------------------------------------------------
# LoRAFinetuner — construction and properties
# ---------------------------------------------------------------------------


def test_finetuner_default_construction():
    ft = LoRAFinetuner()
    assert ft.lora_config.r == 8
    assert ft.training_config.num_epochs == 3


def test_finetuner_safety_fields_accessible():
    cfg = LoRAConfig(safety_lora_rank=1, enable_splora_audit=True)
    ft = LoRAFinetuner(lora_config=cfg)
    assert ft.lora_config.safety_lora_rank == 1
    assert ft.lora_config.enable_splora_audit is True


def test_finetuner_config_summary_includes_safety_fields():
    cfg = LoRAConfig(safety_lora_rank=1, enable_splora_audit=True)
    ft = LoRAFinetuner(lora_config=cfg)
    summary = ft.config_summary()
    assert "safety_lora_rank" in summary["lora"]
    assert summary["lora"]["safety_lora_rank"] == 1
    assert "enable_splora_audit" in summary["lora"]
    assert summary["lora"]["enable_splora_audit"] is True


def test_finetuner_config_summary_safety_subspace_path():
    cfg = LoRAConfig(safety_subspace_path="/tmp/s.pt")
    ft = LoRAFinetuner(lora_config=cfg)
    summary = ft.config_summary()
    assert summary["lora"]["safety_subspace_path"] == "/tmp/s.pt"


# ---------------------------------------------------------------------------
# LoRAFinetuner.train() — import guard (no torch available)
# ---------------------------------------------------------------------------


def test_train_raises_import_error_without_torch():
    with patch.dict(sys.modules, {"torch": None, "transformers": None, "datasets": None}):
        ft = LoRAFinetuner()
        with pytest.raises(ImportError, match="toki\\[hf\\]"):
            ft.train(MagicMock(), MagicMock(), prompts=["test"])


# ---------------------------------------------------------------------------
# LoRAFinetuner.train() — safety hook integration (mocked torch)
# ---------------------------------------------------------------------------


def _make_mock_env():
    """Build a minimal mock environment for LoRAFinetuner.train()."""
    mock_torch = MagicMock()
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    mock_trainer_output = MagicMock()
    mock_trainer_output.training_loss = 0.25
    mock_trainer_output.global_step = 10

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = mock_trainer_output

    mock_transformers = MagicMock()
    mock_transformers.Trainer.return_value = mock_trainer
    mock_transformers.TrainingArguments = MagicMock()
    mock_transformers.DataCollatorForLanguageModeling = MagicMock()

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}

    mock_hf_dataset = MagicMock()
    mock_hf_dataset.map.return_value = mock_hf_dataset

    mock_datasets = MagicMock()
    mock_datasets.Dataset.from_dict.return_value = mock_hf_dataset

    return mock_torch, mock_transformers, mock_datasets, mock_tokenizer


def test_train_returns_lora_train_result():
    mock_torch, mock_transformers, mock_datasets, _ = _make_mock_env()
    modules = {
        "torch": mock_torch,
        "transformers": mock_transformers,
        "datasets": mock_datasets,
    }
    with patch.dict(sys.modules, modules):
        ft = LoRAFinetuner()
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []
        result = ft.train(mock_model, MagicMock(), prompts=["p1", "p2"])

    assert isinstance(result, LoRATrainResult)
    assert result.training_loss == pytest.approx(0.25)
    assert result.num_steps == 10
    assert result.splora_audit is None


def test_train_no_safety_fields_no_audit():
    mock_torch, mock_transformers, mock_datasets, _ = _make_mock_env()
    modules = {
        "torch": mock_torch,
        "transformers": mock_transformers,
        "datasets": mock_datasets,
    }
    with patch.dict(sys.modules, modules):
        cfg = LoRAConfig()  # all safety defaults (disabled)
        ft = LoRAFinetuner(lora_config=cfg)
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []
        result = ft.train(mock_model, MagicMock(), prompts=["p"])

    assert result.splora_audit is None


def test_train_raises_without_prompts_or_dataset():
    mock_torch, mock_transformers, mock_datasets, _ = _make_mock_env()
    modules = {
        "torch": mock_torch,
        "transformers": mock_transformers,
        "datasets": mock_datasets,
    }
    with patch.dict(sys.modules, modules):
        ft = LoRAFinetuner()
        with pytest.raises(ValueError, match="dataset or prompts"):
            ft.train(MagicMock(), MagicMock())
