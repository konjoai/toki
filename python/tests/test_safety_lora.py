"""Tests for toki.safety_lora — SafetyLoRAConfig, SploraAuditResult,
LoRATrainResult, load_safety_subspace, freeze_safety_adapter, splora_audit."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from toki.safety_lora import (
    LoRATrainResult,
    SafetyLoRAConfig,
    SploraAuditResult,
    freeze_safety_adapter,
    load_safety_subspace,
    splora_audit,
)


# ---------------------------------------------------------------------------
# SafetyLoRAConfig
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = SafetyLoRAConfig()
    assert cfg.safety_lora_rank == 0
    assert cfg.safety_subspace_path is None
    assert cfg.enable_splora_audit is False
    assert cfg.splora_threshold == pytest.approx(0.15)


def test_config_custom_values():
    cfg = SafetyLoRAConfig(
        safety_lora_rank=1,
        safety_subspace_path="/tmp/safety.pt",
        enable_splora_audit=True,
        splora_threshold=0.2,
    )
    assert cfg.safety_lora_rank == 1
    assert cfg.safety_subspace_path == "/tmp/safety.pt"
    assert cfg.enable_splora_audit is True
    assert cfg.splora_threshold == pytest.approx(0.2)


def test_config_rank_zero_disabled():
    cfg = SafetyLoRAConfig(safety_lora_rank=0)
    assert cfg.safety_lora_rank == 0


def test_config_fields_independent():
    a = SafetyLoRAConfig(safety_lora_rank=1)
    b = SafetyLoRAConfig(safety_lora_rank=0)
    assert a.safety_lora_rank != b.safety_lora_rank
    assert a.enable_splora_audit == b.enable_splora_audit


# ---------------------------------------------------------------------------
# SploraAuditResult
# ---------------------------------------------------------------------------


def test_audit_result_passed_no_flagged():
    r = SploraAuditResult(flagged_layers=[], max_ediem=0.05, passed=True, threshold=0.15)
    assert r.passed is True
    assert r.flagged_layers == []


def test_audit_result_failed_with_flagged():
    r = SploraAuditResult(
        flagged_layers=["model.layer.0.weight"],
        max_ediem=0.3,
        passed=False,
        threshold=0.15,
    )
    assert r.passed is False
    assert len(r.flagged_layers) == 1


def test_audit_result_to_dict():
    r = SploraAuditResult(
        flagged_layers=["a", "b"], max_ediem=0.2, passed=False, threshold=0.15
    )
    d = r.to_dict()
    assert d["passed"] is False
    assert d["flagged_layers"] == ["a", "b"]
    assert d["max_ediem"] == pytest.approx(0.2)
    assert d["threshold"] == pytest.approx(0.15)


def test_audit_result_frozen():
    r = SploraAuditResult(flagged_layers=[], max_ediem=0.0, passed=True, threshold=0.15)
    with pytest.raises(Exception):
        r.passed = False  # type: ignore[misc]


def test_audit_result_max_ediem_propagated():
    r = SploraAuditResult(flagged_layers=[], max_ediem=0.08, passed=True, threshold=0.15)
    assert r.max_ediem == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# LoRATrainResult
# ---------------------------------------------------------------------------


def test_lora_train_result_fields():
    r = LoRATrainResult(training_loss=0.42, num_steps=100)
    assert r.training_loss == pytest.approx(0.42)
    assert r.num_steps == 100
    assert r.splora_audit is None


def test_lora_train_result_with_audit():
    audit = SploraAuditResult(flagged_layers=[], max_ediem=0.01, passed=True, threshold=0.15)
    r = LoRATrainResult(training_loss=0.1, num_steps=50, splora_audit=audit)
    assert r.splora_audit is audit
    assert r.splora_audit.passed is True


def test_lora_train_result_splora_none_by_default():
    r = LoRATrainResult(training_loss=0.5, num_steps=10)
    assert r.splora_audit is None


# ---------------------------------------------------------------------------
# load_safety_subspace — import guard
# ---------------------------------------------------------------------------


def test_load_safety_subspace_import_error():
    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match=r"toki\[hf\]"):
            load_safety_subspace("any_path.pt")


def test_load_safety_subspace_import_error_message():
    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match="pip install toki"):
            load_safety_subspace("any_path.pt")


def test_load_safety_subspace_file_not_found(tmp_path):
    mock_torch = MagicMock()
    with patch.dict(sys.modules, {"torch": mock_torch}):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_safety_subspace(str(tmp_path / "nonexistent.pt"))


def test_load_safety_subspace_returns_state_dict(tmp_path):
    fake_state = {"weight": MagicMock()}
    mock_torch = MagicMock()
    mock_torch.load.return_value = fake_state
    checkpoint = tmp_path / "safety.pt"
    checkpoint.write_bytes(b"fake")
    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = load_safety_subspace(str(checkpoint))
    assert result is fake_state


# ---------------------------------------------------------------------------
# freeze_safety_adapter — import guard and no-op
# ---------------------------------------------------------------------------


def test_freeze_safety_adapter_noop_when_none():
    mock_model = MagicMock()
    freeze_safety_adapter(mock_model, None)
    mock_model.named_parameters.assert_not_called()


def test_freeze_safety_adapter_import_error():
    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match=r"toki\[hf\]"):
            freeze_safety_adapter(MagicMock(), {"layer": MagicMock()})


def test_freeze_safety_adapter_applies_matching_tensors():
    # Build minimal mock model and torch
    mock_param = MagicMock()
    mock_param.device = "cpu"
    mock_param.shape = (4, 4)
    mock_param.data = MagicMock()

    mock_model = MagicMock()
    mock_model.named_parameters.return_value = [("base_model.model.weight", mock_param)]

    mock_delta = MagicMock()
    mock_delta.shape = (4, 4)
    mock_delta.to.return_value = mock_delta

    mock_torch = MagicMock()
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict(sys.modules, {"torch": mock_torch}):
        freeze_safety_adapter(mock_model, {"weight": mock_delta})

    mock_param.requires_grad_.assert_called_once_with(False)


def test_freeze_safety_adapter_skips_non_matching_keys():
    mock_model = MagicMock()
    mock_model.named_parameters.return_value = [("model.weight", MagicMock())]

    mock_torch = MagicMock()
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict(sys.modules, {"torch": mock_torch}):
        # safety_delta has different key — no match → no freeze
        freeze_safety_adapter(mock_model, {"completely_different_key": MagicMock()})

    for _, param in mock_model.named_parameters():
        param.requires_grad_.assert_not_called()


# ---------------------------------------------------------------------------
# splora_audit — import guard and basic behaviour
# ---------------------------------------------------------------------------


def test_splora_audit_import_error():
    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match=r"toki\[hf\]"):
            splora_audit(MagicMock(), {})


def test_splora_audit_returns_audit_result():
    mock_torch = MagicMock()
    mock_model = MagicMock()
    mock_model.named_parameters.return_value = []

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = splora_audit(mock_model, {})

    assert isinstance(result, SploraAuditResult)
    assert result.passed is True
    assert result.flagged_layers == []


def test_splora_audit_empty_base_state_passes():
    mock_torch = MagicMock()
    mock_model = MagicMock()
    mock_model.named_parameters.return_value = []

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = splora_audit(mock_model, {}, threshold=0.15)

    assert result.passed is True
    assert result.max_ediem == pytest.approx(0.0)


def test_splora_audit_threshold_propagated():
    mock_torch = MagicMock()
    mock_model = MagicMock()
    mock_model.named_parameters.return_value = []

    with patch.dict(sys.modules, {"torch": mock_torch}):
        result = splora_audit(mock_model, {}, threshold=0.3)

    assert result.threshold == pytest.approx(0.3)
