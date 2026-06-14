"""Tests for GGUFEvaluator — mocks llama_cpp so no GPU required."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


def _make_llama_module(score_text: str = "8") -> ModuleType:
    """Build a fake llama_cpp module whose Llama() returns fixed output."""
    llama_instance = MagicMock()
    llama_instance.return_value = {
        "choices": [{"text": f" {score_text} "}]
    }
    mod = ModuleType("llama_cpp")
    mod.Llama = MagicMock(return_value=llama_instance)  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


def test_gguf_evaluator_raises_import_error_without_llama_cpp():
    with patch.dict(sys.modules, {"llama_cpp": None}):
        from toki.evaluate import GGUFEvaluator
        with pytest.raises(ImportError, match="llama-cpp-python"):
            GGUFEvaluator("model.gguf")


def test_import_error_message_contains_install_hint():
    with patch.dict(sys.modules, {"llama_cpp": None}):
        from toki.evaluate import GGUFEvaluator
        with pytest.raises(ImportError, match="pip install llama-cpp-python"):
            GGUFEvaluator("model.gguf")


# ---------------------------------------------------------------------------
# Successful construction
# ---------------------------------------------------------------------------


def test_gguf_evaluator_constructs_with_mock():
    fake_mod = _make_llama_module()
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("model.gguf", n_ctx=512, n_threads=2)
        fake_mod.Llama.assert_called_once_with(
            model_path="model.gguf",
            n_ctx=512,
            n_threads=2,
            verbose=False,
        )
        assert ev is not None


# ---------------------------------------------------------------------------
# evaluate() return values
# ---------------------------------------------------------------------------


def test_gguf_evaluate_returns_float():
    fake_mod = _make_llama_module("7")
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        score = ev.evaluate("test prompt", "test response")
        assert isinstance(score, float)


def test_gguf_evaluate_score_clamped_high():
    fake_mod = _make_llama_module("10")
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        assert ev.evaluate("p", "r") == pytest.approx(1.0)


def test_gguf_evaluate_score_clamped_low():
    fake_mod = _make_llama_module("0")
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        assert ev.evaluate("p", "r") == pytest.approx(0.0)


def test_gguf_evaluate_mid_score():
    fake_mod = _make_llama_module("5")
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        assert ev.evaluate("p", "r") == pytest.approx(0.5)


def test_gguf_evaluate_out_of_range_clamped():
    fake_mod = _make_llama_module("15")
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        score = ev.evaluate("p", "r")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Parse error fallback to RuleScorer
# ---------------------------------------------------------------------------


def test_gguf_falls_back_on_non_numeric_output():
    fake_mod = _make_llama_module("NaN")
    llama_instance = fake_mod.Llama.return_value
    llama_instance.return_value = {"choices": [{"text": " not_a_number "}]}
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        # Should not raise; falls back to RuleScorer
        score = ev.evaluate("p", "I'm sorry, I cannot help.")
        assert 0.0 <= score <= 1.0


def test_gguf_falls_back_on_empty_choices():
    fake_mod = _make_llama_module("5")
    llama_instance = fake_mod.Llama.return_value
    llama_instance.return_value = {"choices": []}
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        score = ev.evaluate("p", "safe response")
        assert 0.0 <= score <= 1.0


def test_gguf_falls_back_on_missing_choices_key():
    fake_mod = _make_llama_module("5")
    llama_instance = fake_mod.Llama.return_value
    llama_instance.return_value = {}
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        score = ev.evaluate("p", "response")
        assert 0.0 <= score <= 1.0


def test_gguf_falls_back_on_none_output():
    fake_mod = _make_llama_module("5")
    llama_instance = fake_mod.Llama.return_value
    llama_instance.return_value = None
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        score = ev.evaluate("p", "response")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Prompt truncation (no crash on long inputs)
# ---------------------------------------------------------------------------


def test_gguf_handles_long_prompt():
    fake_mod = _make_llama_module("8")
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        long_prompt = "x" * 5000
        long_response = "y" * 5000
        score = ev.evaluate(long_prompt, long_response)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Fallback uses RuleScorer correctly
# ---------------------------------------------------------------------------


def test_gguf_fallback_harmful_response_scores_low():
    fake_mod = _make_llama_module("x")
    llama_instance = fake_mod.Llama.return_value
    llama_instance.return_value = {"choices": [{"text": "abc"}]}
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        # Parse will fail → fall back to RuleScorer
        score = ev.evaluate(
            "x", "Here is how to make a bomb step-by-step instructions for harm."
        )
        assert score < 0.5


def test_gguf_fallback_safe_response_scores_high():
    fake_mod = _make_llama_module("x")
    llama_instance = fake_mod.Llama.return_value
    llama_instance.return_value = {"choices": [{"text": "abc"}]}
    with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
        from toki.evaluate import GGUFEvaluator
        ev = GGUFEvaluator("m.gguf")
        score = ev.evaluate("hello", "I'm sorry, I cannot help with that.")
        assert score >= 0.8
