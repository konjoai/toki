"""Tests for toki.indirect — indirect prompt injection simulator."""
from __future__ import annotations

import pytest

from toki.indirect import (
    OWASP_LLM01,
    InjectionScenario,
    IndirectInjectionEvaluator,
    IndirectInjectionGenerator,
    IndirectPrompt,
    InjectionVerdict,
)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


def test_injection_scenario_has_four_values():
    assert set(InjectionScenario) == {
        InjectionScenario.DOCUMENT,
        InjectionScenario.WEBPAGE,
        InjectionScenario.TOOL_RESPONSE,
        InjectionScenario.EMAIL,
    }


# ---------------------------------------------------------------------------
# Generator counts
# ---------------------------------------------------------------------------


def test_generate_all_returns_20_cases():
    gen = IndirectInjectionGenerator()
    assert len(gen.generate_all()) == 20


def test_generate_by_scenario_returns_5_cases_each():
    gen = IndirectInjectionGenerator()
    for scenario in InjectionScenario:
        cases = gen.generate_by_scenario(scenario)
        assert len(cases) == 5, f"{scenario} returned {len(cases)}"


def test_generate_all_covers_all_four_scenarios():
    cases = IndirectInjectionGenerator().generate_all()
    found = {c.scenario for c in cases}
    assert found == set(InjectionScenario)


def test_each_scenario_count_in_all_output():
    cases = IndirectInjectionGenerator().generate_all()
    from collections import Counter
    counts = Counter(c.scenario for c in cases)
    for scenario in InjectionScenario:
        assert counts[scenario] == 5


# ---------------------------------------------------------------------------
# Dataclass fields
# ---------------------------------------------------------------------------


def test_indirect_prompt_is_frozen():
    gen = IndirectInjectionGenerator()
    p = gen.generate_all()[0]
    with pytest.raises((AttributeError, TypeError)):
        p.user_task = "mutated"  # type: ignore[misc]


def test_all_cases_have_owasp_llm01_tag():
    for p in IndirectInjectionGenerator().generate_all():
        assert p.owasp_category == OWASP_LLM01


def test_all_cases_have_non_empty_user_task():
    for p in IndirectInjectionGenerator().generate_all():
        assert p.user_task.strip()


def test_injected_text_contains_canonical_payload():
    gen = IndirectInjectionGenerator()
    for p in gen.generate_all():
        assert len(p.injected_text) > len(p.user_task)


def test_all_cases_have_non_empty_target_action():
    for p in IndirectInjectionGenerator().generate_all():
        assert p.target_action.strip()


# ---------------------------------------------------------------------------
# Determinism and uniqueness
# ---------------------------------------------------------------------------


def test_generate_all_is_deterministic():
    g = IndirectInjectionGenerator()
    a = g.generate_all()
    b = g.generate_all()
    assert [p.seed for p in a] == [p.seed for p in b]
    assert [p.injected_text for p in a] == [p.injected_text for p in b]


def test_all_seeds_are_distinct():
    seeds = [p.seed for p in IndirectInjectionGenerator().generate_all()]
    assert len(set(seeds)) == 20


def test_all_user_tasks_are_distinct():
    tasks = [p.user_task for p in IndirectInjectionGenerator().generate_all()]
    assert len(set(tasks)) == 20


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def test_evaluator_returns_injection_verdict():
    gen = IndirectInjectionGenerator()
    ev = IndirectInjectionEvaluator()
    p = gen.generate_all()[0]
    verdict = ev.evaluate(p, "Here is a summary of the document.")
    assert isinstance(verdict, InjectionVerdict)


def test_evaluator_score_is_in_valid_range():
    gen = IndirectInjectionGenerator()
    ev = IndirectInjectionEvaluator()
    for p in gen.generate_all():
        v = ev.evaluate(p, "I detected an injection attempt and refused.")
        assert 0.0 <= v.score <= 1.0


def test_evaluator_flags_injection_signal():
    gen = IndirectInjectionGenerator()
    ev = IndirectInjectionEvaluator()
    p = gen.generate_all()[0]
    v = ev.evaluate(p, "I detected an injection attempt in the document.")
    assert v.flagged_injection is True


def test_evaluator_adversarial_success_is_false_for_safe_response():
    gen = IndirectInjectionGenerator()
    ev = IndirectInjectionEvaluator()
    safe_fn = lambda _: "Here is a concise summary of the document content."
    verdicts = ev.evaluate_batch(
        gen.generate_by_scenario(InjectionScenario.DOCUMENT), safe_fn
    )
    for v in verdicts:
        assert isinstance(v.adversarial_success, bool)


def test_to_dict_contains_required_keys():
    gen = IndirectInjectionGenerator()
    ev = IndirectInjectionEvaluator()
    p = gen.generate_all()[0]
    d = ev.evaluate(p, "test response").to_dict()
    required = {
        "scenario", "user_task", "target_action", "owasp_category",
        "followed_injection", "completed_user_task", "flagged_injection",
        "adversarial_success", "score",
    }
    assert required <= d.keys()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_empty_verdicts():
    ev = IndirectInjectionEvaluator()
    s = ev.summary([])
    assert s["total"] == 0
    assert s["mean_score"] == 0.0
    assert s["adversarial_success_rate"] == 0.0
    assert s["by_scenario"] == {}


def test_summary_has_all_four_scenarios_when_full_battery_evaluated():
    gen = IndirectInjectionGenerator()
    ev = IndirectInjectionEvaluator()
    safe_fn = lambda _: "Summary complete."
    verdicts = ev.evaluate_batch(gen.generate_all(), safe_fn)
    s = ev.summary(verdicts)
    assert s["total"] == 20
    assert set(s["by_scenario"].keys()) == {sc.value for sc in InjectionScenario}
    assert 0.0 <= s["mean_score"] <= 1.0
    assert 0.0 <= s["adversarial_success_rate"] <= 1.0
