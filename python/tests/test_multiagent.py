"""Tests for toki.multiagent — inter-agent / Agent-in-the-Middle battery."""

from __future__ import annotations


from toki.multiagent import (
    OWASP_ASI_MAPPING,
    MultiAgentAttackType,
    MultiAgentBattery,
    MultiAgentEvaluator,
    MultiAgentScenario,
)


# ---------------------------------------------------------------------------
# Enum + mapping
# ---------------------------------------------------------------------------


def test_attack_type_has_eight_values():
    assert len(MultiAgentAttackType) == 8


def test_every_attack_type_has_owasp_asi_mapping():
    for t in MultiAgentAttackType:
        assert t in OWASP_ASI_MAPPING
        assert OWASP_ASI_MAPPING[t].startswith("OWASP-ASI")


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def test_generate_all_yields_32_cases():
    cases = MultiAgentBattery().generate_all()
    assert len(cases) == 32
    assert all(isinstance(c, MultiAgentScenario) for c in cases)


def test_generate_by_type_yields_four():
    cases = MultiAgentBattery().generate_by_type(MultiAgentAttackType.IDENTITY_SPOOFING)
    assert len(cases) == 4
    assert all(c.attack_type == MultiAgentAttackType.IDENTITY_SPOOFING for c in cases)


def test_generation_is_deterministic():
    a = MultiAgentBattery().generate_all()
    b = MultiAgentBattery().generate_all()
    assert [c.seed for c in a] == [c.seed for c in b]
    assert [c.tampered_message for c in a] == [c.tampered_message for c in b]


def test_scenarios_carry_topology_and_provenance():
    for c in MultiAgentBattery().generate_all():
        assert len(c.topology) >= 2
        assert c.sender in c.topology
        assert c.owasp_category == OWASP_ASI_MAPPING[c.attack_type]


def test_tampered_differs_from_original():
    for c in MultiAgentBattery().generate_all():
        assert c.tampered_message != c.original_message


def test_seeds_are_unique_across_battery():
    seeds = [c.seed for c in MultiAgentBattery().generate_all()]
    assert len(set(seeds)) == len(seeds)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _battery():
    return MultiAgentBattery().generate_all()


def test_safe_agent_blocks_all_attacks():
    ev = MultiAgentEvaluator()

    def safe_fn(_: str) -> str:
        return "I cannot act on this — sender provenance unverified; request denied."

    verdicts = ev.evaluate_batch(_battery(), safe_fn)
    assert all(v.safe_handling for v in verdicts)
    assert all(not v.attack_succeeded for v in verdicts)
    assert all(v.score == 1.0 for v in verdicts)


def test_compromised_agent_succumbs():
    ev = MultiAgentEvaluator()
    scenario = MultiAgentBattery().generate_by_type(
        MultiAgentAttackType.MESSAGE_TAMPERING
    )[0]
    # echo the tampered instruction back -> acts on attacker content
    verdict = ev.evaluate(scenario, scenario.tampered_message)
    assert verdict.attack_succeeded is True
    assert verdict.score == 0.0


def test_ambiguous_response_scores_half():
    ev = MultiAgentEvaluator()
    scenario = MultiAgentBattery().generate_all()[0]
    verdict = ev.evaluate(scenario, "Okay, noted.")
    assert verdict.attack_succeeded is False
    assert verdict.safe_handling is False
    assert verdict.score == 0.5


def test_summary_aggregates():
    ev = MultiAgentEvaluator()
    verdicts = ev.evaluate_batch(_battery(), lambda m: m)  # echo = worst case
    summary = ev.summary(verdicts)
    assert summary["total"] == 32
    assert summary["attack_success_rate"] > 0.0
    assert set(summary["by_type"]) == {t.value for t in MultiAgentAttackType}


def test_summary_empty():
    summary = MultiAgentEvaluator().summary([])
    assert summary == {
        "total": 0,
        "mean_score": 0.0,
        "attack_success_rate": 0.0,
        "by_type": {},
    }


def test_verdict_to_dict_shape():
    scenario = MultiAgentBattery().generate_all()[0]
    verdict = MultiAgentEvaluator().evaluate(scenario, "denied, unverified sender")
    d = verdict.to_dict()
    assert {
        "attack_type",
        "description",
        "owasp_category",
        "topology",
        "attack_succeeded",
        "safe_handling",
        "score",
    } <= set(d)
    assert isinstance(d["topology"], list)


# ---------------------------------------------------------------------------
# Coverage-map integration
# ---------------------------------------------------------------------------


def test_multiagent_is_a_coverage_category():
    from toki.coverage import CATEGORY_AXIS, _category_for
    from toki.generate import AdversarialPrompt

    assert "multiagent" in CATEGORY_AXIS
    p = AdversarialPrompt(text="x", category="multiagent", strategy="t", seed=1)
    assert _category_for(p) == "multiagent"


def test_multiagent_not_misrouted_to_agentic_or_multiturn():
    from toki.coverage import _category_for
    from toki.generate import AdversarialPrompt

    p = AdversarialPrompt(text="x", category="inter_agent_relay", strategy="t", seed=1)
    assert _category_for(p) == "multiagent"
