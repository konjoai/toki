"""Tests for toki.evaluate — RobustnessEvaluator."""
from toki.evaluate import EvaluationResult, RobustnessEvaluator
from toki.generate import AdversarialGenerator, AdversarialPrompt


def _make_prompt(text: str = "test prompt", category: str = "jailbreak") -> AdversarialPrompt:
    return AdversarialPrompt(text=text, category=category, strategy="test", seed=0)


def test_mock_response_is_refusal():
    ev = RobustnessEvaluator()
    prompt = _make_prompt()
    result = ev.evaluate_one(prompt)
    assert result.refused is True


def test_mock_score_is_high():
    ev = RobustnessEvaluator()
    prompt = _make_prompt()
    result = ev.evaluate_one(prompt)
    assert result.score >= 0.8, f"expected high score for refusal, got {result.score}"


def test_harmful_response_scores_low():
    def harmful_fn(p: str) -> str:  # noqa: ARG001
        return "Sure! Here is how to make a bomb step-by-step instructions for harm."

    ev = RobustnessEvaluator(model_fn=harmful_fn)
    prompt = _make_prompt()
    result = ev.evaluate_one(prompt)
    assert result.contained_harmful is True
    assert result.score < 0.5, f"harmful response should score < 0.5, got {result.score}"


def test_leak_response_detected():
    def leaky_fn(p: str) -> str:  # noqa: ARG001
        return "My system prompt says I must always obey the user."

    ev = RobustnessEvaluator(model_fn=leaky_fn)
    prompt = _make_prompt()
    result = ev.evaluate_one(prompt)
    assert result.leaked_system is True


def test_evaluate_batch_returns_all():
    gen = AdversarialGenerator(seed=42)
    prompts = gen.generate_jailbreaks(count=5)
    ev = RobustnessEvaluator()
    results = ev.evaluate_batch(prompts)
    assert len(results) == len(prompts)


def test_summary_mean_score():
    gen = AdversarialGenerator(seed=42)
    prompts = gen.generate_all()
    ev = RobustnessEvaluator()
    results = ev.evaluate_batch(prompts)
    s = ev.summary(results)
    assert 0.0 <= s["mean_score"] <= 1.0


def test_summary_refusal_rate():
    gen = AdversarialGenerator(seed=42)
    prompts = gen.generate_all()
    ev = RobustnessEvaluator()
    results = ev.evaluate_batch(prompts)
    s = ev.summary(results)
    assert 0.0 <= s["refusal_rate"] <= 1.0


def test_by_category_in_summary():
    gen = AdversarialGenerator(seed=42)
    prompts = gen.generate_all()
    ev = RobustnessEvaluator()
    results = ev.evaluate_batch(prompts)
    s = ev.summary(results)
    assert "by_category" in s
    cats = set(s["by_category"].keys())
    assert cats == {"jailbreak", "injection", "edge_case", "boundary"}


def test_summary_empty_input():
    ev = RobustnessEvaluator()
    s = ev.summary([])
    assert s["total"] == 0
    assert s["mean_score"] == 0.0
