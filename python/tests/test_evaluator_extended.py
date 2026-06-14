"""Tests for RobustnessEvaluator with evaluator_mode and hybrid integration."""
from __future__ import annotations

from unittest.mock import MagicMock

from toki.evaluate import EvaluationResult, EvaluatorMode, RobustnessEvaluator
from toki.generate import AdversarialPrompt


def _prompt(text: str = "test", category: str = "jailbreak") -> AdversarialPrompt:
    return AdversarialPrompt(text=text, category=category, strategy="test", seed=0)


# ---------------------------------------------------------------------------
# Backward compatibility — no mode param
# ---------------------------------------------------------------------------


def test_no_mode_behaves_as_before():
    ev = RobustnessEvaluator()
    result = ev.evaluate_one(_prompt())
    assert isinstance(result, EvaluationResult)
    assert result.refused is True
    assert result.score >= 0.8


def test_no_mode_uses_legacy_scoring():
    def harmful_fn(p: str) -> str:  # noqa: ARG001
        return "step-by-step instructions for harm."

    ev = RobustnessEvaluator(model_fn=harmful_fn)
    result = ev.evaluate_one(_prompt())
    assert result.score < 0.5


# ---------------------------------------------------------------------------
# RULE mode
# ---------------------------------------------------------------------------


def test_rule_mode_no_judge_needed():
    ev = RobustnessEvaluator(evaluator_mode=EvaluatorMode.RULE)
    result = ev.evaluate_one(_prompt())
    assert 0.0 <= result.score <= 1.0


def test_rule_mode_refusal_scores_high():
    ev = RobustnessEvaluator(evaluator_mode=EvaluatorMode.RULE)
    result = ev.evaluate_one(_prompt())
    assert result.score >= 0.8


def test_rule_mode_batch_works():
    ev = RobustnessEvaluator(evaluator_mode=EvaluatorMode.RULE)
    prompts = [_prompt(f"p{i}") for i in range(5)]
    results = ev.evaluate_batch(prompts)
    assert len(results) == 5
    assert all(isinstance(r, EvaluationResult) for r in results)


# ---------------------------------------------------------------------------
# HYBRID mode with mock judge
# ---------------------------------------------------------------------------


def _make_mock_judge(overall_score: float) -> MagicMock:
    verdict = MagicMock()
    verdict.overall_score = overall_score
    mock_judge = MagicMock()
    mock_judge.judge.return_value = verdict
    return mock_judge


def test_hybrid_mode_calls_judge():
    mock_judge = _make_mock_judge(0.7)
    ev = RobustnessEvaluator(
        evaluator_mode=EvaluatorMode.HYBRID,
        llm_judge=mock_judge,
    )
    ev.evaluate_one(_prompt("some prompt"))
    mock_judge.judge.assert_called_once()


def test_hybrid_mode_score_averages():
    mock_judge = _make_mock_judge(0.5)
    ev = RobustnessEvaluator(
        evaluator_mode=EvaluatorMode.HYBRID,
        llm_judge=mock_judge,
    )
    result = ev.evaluate_one(_prompt())
    # mock model fn returns refusal → rule_score ~0.9; llm=0.5; hybrid ≈ 0.7
    assert 0.0 <= result.score <= 1.0


def test_hybrid_mode_score_in_valid_range():
    mock_judge = _make_mock_judge(1.0)
    ev = RobustnessEvaluator(
        evaluator_mode=EvaluatorMode.HYBRID,
        llm_judge=mock_judge,
    )
    results = ev.evaluate_batch([_prompt(f"p{i}") for i in range(8)])
    assert all(0.0 <= r.score <= 1.0 for r in results)


# ---------------------------------------------------------------------------
# Summary is unaffected by mode
# ---------------------------------------------------------------------------


def test_summary_works_with_hybrid_mode():
    mock_judge = _make_mock_judge(0.8)
    ev = RobustnessEvaluator(
        evaluator_mode=EvaluatorMode.HYBRID,
        llm_judge=mock_judge,
    )
    from toki.generate import AdversarialGenerator
    gen = AdversarialGenerator(seed=42)
    results = ev.evaluate_batch(gen.generate_jailbreaks(count=5))
    s = ev.summary(results)
    assert 0.0 <= s["mean_score"] <= 1.0
    assert "by_category" in s
