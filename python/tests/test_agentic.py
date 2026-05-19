"""Tests for toki.agentic — agentic + MCP attack battery."""
from __future__ import annotations

import pytest

from toki.agentic import (
    OWASP_MAPPING,
    AgentAttackBattery,
    AgentAttackEvaluator,
    AgentAttackScenario,
    AgentAttackType,
    AgentVerdict,
)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


def test_attack_type_has_nine_values():
    assert len(AgentAttackType) == 9


def test_agentlab_five_classes_present():
    names = {t.value for t in AgentAttackType}
    assert "intent_hijacking" in names
    assert "tool_chaining" in names
    assert "task_injection" in names
    assert "objective_drifting" in names
    assert "memory_poisoning" in names


def test_mcp_three_primitives_present():
    names = {t.value for t in AgentAttackType}
    assert "mcp_rug_pull" in names
    assert "mcp_tool_shadowing" in names
    assert "mcp_registry_poisoning" in names


def test_tool_metadata_poisoning_present():
    assert AgentAttackType.TOOL_METADATA_POISONING in AgentAttackType


# ---------------------------------------------------------------------------
# OWASP mapping
# ---------------------------------------------------------------------------


def test_every_attack_type_has_owasp_mapping():
    for attack_type in AgentAttackType:
        assert attack_type in OWASP_MAPPING
        assert OWASP_MAPPING[attack_type].startswith("OWASP-LLM")


def test_mcp_types_map_to_supply_chain():
    mcp_types = {
        AgentAttackType.MCP_RUG_PULL,
        AgentAttackType.MCP_TOOL_SHADOWING,
        AgentAttackType.MCP_REGISTRY_POISONING,
    }
    for t in mcp_types:
        assert OWASP_MAPPING[t] == "OWASP-LLM03:2025"


def test_injection_types_map_to_lm01():
    injection_types = {
        AgentAttackType.INTENT_HIJACKING,
        AgentAttackType.TASK_INJECTION,
        AgentAttackType.MEMORY_POISONING,
    }
    for t in injection_types:
        assert OWASP_MAPPING[t] == "OWASP-LLM01:2025"


# ---------------------------------------------------------------------------
# Battery counts
# ---------------------------------------------------------------------------


def test_generate_all_returns_36_cases():
    battery = AgentAttackBattery()
    assert len(battery.generate_all()) == 36


def test_generate_by_type_returns_4_cases_each():
    battery = AgentAttackBattery()
    for attack_type in AgentAttackType:
        cases = battery.generate_by_type(attack_type)
        assert len(cases) == 4, f"{attack_type} returned {len(cases)}"


def test_generate_all_covers_all_nine_types():
    cases = AgentAttackBattery().generate_all()
    found = {c.attack_type for c in cases}
    assert found == set(AgentAttackType)


# ---------------------------------------------------------------------------
# Dataclass fields
# ---------------------------------------------------------------------------


def test_scenario_is_frozen():
    scenario = AgentAttackBattery().generate_all()[0]
    with pytest.raises((AttributeError, TypeError)):
        scenario.description = "mutated"  # type: ignore[misc]


def test_all_scenarios_have_non_empty_fields():
    for s in AgentAttackBattery().generate_all():
        assert s.system_prompt.strip()
        assert s.user_goal.strip()
        assert s.attack_payload.strip()
        assert s.description.strip()
        assert s.owasp_category.strip()


def test_all_scenarios_carry_matching_owasp_tag():
    for s in AgentAttackBattery().generate_all():
        assert s.owasp_category == OWASP_MAPPING[s.attack_type]


# ---------------------------------------------------------------------------
# Determinism and uniqueness
# ---------------------------------------------------------------------------


def test_generate_all_is_deterministic():
    b = AgentAttackBattery()
    a = b.generate_all()
    c = b.generate_all()
    assert [s.seed for s in a] == [s.seed for s in c]
    assert [s.description for s in a] == [s.description for s in c]


def test_all_seeds_are_distinct():
    seeds = [s.seed for s in AgentAttackBattery().generate_all()]
    assert len(set(seeds)) == 36


def test_mcp_primitives_have_distinct_descriptions():
    battery = AgentAttackBattery()
    rug = battery.generate_by_type(AgentAttackType.MCP_RUG_PULL)
    shadow = battery.generate_by_type(AgentAttackType.MCP_TOOL_SHADOWING)
    registry = battery.generate_by_type(AgentAttackType.MCP_REGISTRY_POISONING)
    rug_descs = {s.description for s in rug}
    shadow_descs = {s.description for s in shadow}
    registry_descs = {s.description for s in registry}
    assert not (rug_descs & shadow_descs)
    assert not (rug_descs & registry_descs)
    assert not (shadow_descs & registry_descs)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def test_evaluator_returns_agent_verdict():
    battery = AgentAttackBattery()
    ev = AgentAttackEvaluator()
    s = battery.generate_all()[0]
    verdict = ev.evaluate(s, "I cannot follow that instruction — it is unsafe.")
    assert isinstance(verdict, AgentVerdict)


def test_evaluator_score_is_in_valid_range():
    battery = AgentAttackBattery()
    ev = AgentAttackEvaluator()
    for s in battery.generate_all():
        v = ev.evaluate(s, "I refuse this request as it is malicious.")
        assert 0.0 <= v.score <= 1.0


def test_evaluator_refusal_response_gives_safe_refusal_true():
    battery = AgentAttackBattery()
    ev = AgentAttackEvaluator()
    s = battery.generate_all()[0]
    v = ev.evaluate(s, "I cannot comply — this request is unsafe and I refuse.")
    assert v.safe_refusal is True


def test_to_dict_contains_required_keys():
    battery = AgentAttackBattery()
    ev = AgentAttackEvaluator()
    s = battery.generate_all()[0]
    d = ev.evaluate(s, "response").to_dict()
    required = {
        "attack_type", "description", "owasp_category",
        "attack_succeeded", "safe_refusal", "score",
    }
    assert required <= d.keys()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_empty_verdicts():
    ev = AgentAttackEvaluator()
    s = ev.summary([])
    assert s["total"] == 0
    assert s["mean_score"] == 0.0
    assert s["attack_success_rate"] == 0.0
    assert s["by_type"] == {}


def test_summary_covers_all_nine_types_when_full_battery():
    battery = AgentAttackBattery()
    ev = AgentAttackEvaluator()
    safe_fn = lambda _: "I cannot follow that instruction."
    verdicts = ev.evaluate_batch(battery.generate_all(), safe_fn)
    s = ev.summary(verdicts)
    assert s["total"] == 36
    assert set(s["by_type"].keys()) == {t.value for t in AgentAttackType}
    assert 0.0 <= s["mean_score"] <= 1.0
    assert 0.0 <= s["attack_success_rate"] <= 1.0
