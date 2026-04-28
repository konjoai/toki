"""Tests for toki.generate — AdversarialGenerator."""
import pytest
from toki.generate import AdversarialGenerator, AdversarialPrompt


def test_generator_creates_jailbreaks():
    gen = AdversarialGenerator(seed=42)
    results = gen.generate_jailbreaks(count=5)
    assert len(results) > 0


def test_jailbreak_category():
    gen = AdversarialGenerator(seed=42)
    results = gen.generate_jailbreaks(count=8)
    assert all(p.category == "jailbreak" for p in results)


def test_generate_injections():
    gen = AdversarialGenerator(seed=42)
    results = gen.generate_injections(count=6)
    assert len(results) > 0
    assert all(p.category == "injection" for p in results)


def test_generate_edge_cases():
    gen = AdversarialGenerator(seed=42)
    results = gen.generate_edge_cases()
    assert len(results) > 0
    texts = [p.text for p in results]
    # Must include empty string, emoji flood, and control chars
    assert "" in texts
    assert any("\U0001f525" in t for t in texts), "emoji flood missing"
    assert any("\x00" in t for t in texts), "control char missing"


def test_generate_boundary_cases():
    gen = AdversarialGenerator(seed=42)
    results = gen.generate_boundary_cases(count=4)
    assert len(results) == 4
    lengths = [len(p.text) for p in results]
    # Each successive boundary prompt must be strictly longer
    for i in range(1, len(lengths)):
        assert lengths[i] > lengths[i - 1], "boundary lengths must scale monotonically"


def test_generate_all_returns_all_categories():
    gen = AdversarialGenerator(seed=42)
    results = gen.generate_all()
    cats = {p.category for p in results}
    assert cats == {"jailbreak", "injection", "edge_case", "boundary"}


def test_prompts_are_unique_within_category():
    gen = AdversarialGenerator(seed=42)
    jailbreaks = gen.generate_jailbreaks(count=10)
    texts = [p.text for p in jailbreaks]
    assert len(texts) == len(set(texts)), "duplicate jailbreak texts found"


def test_different_seeds_differ():
    gen1 = AdversarialGenerator(seed=1)
    gen2 = AdversarialGenerator(seed=2)
    prompts1 = gen1.generate_jailbreaks(count=5)
    prompts2 = gen2.generate_jailbreaks(count=5)
    # Seeds affect the stored seed field even if text templates are shared;
    # verify the seed fields differ
    seeds1 = {p.seed for p in prompts1}
    seeds2 = {p.seed for p in prompts2}
    assert seeds1 != seeds2, "different generator seeds should yield different prompt seeds"


def test_prompt_dataclass_frozen():
    gen = AdversarialGenerator(seed=42)
    p = gen.generate_jailbreaks(count=1)[0]
    with pytest.raises((AttributeError, TypeError)):
        p.text = "mutated"  # type: ignore[misc]


def test_iter_prompts():
    gen = AdversarialGenerator(seed=42)
    all_prompts = gen.generate_all()
    iterated = list(gen.iter_prompts())
    assert len(iterated) == len(all_prompts)
    assert iterated[0].category == all_prompts[0].category


def test_generate_all_count_sane():
    """Smoke test: generate_all with default params returns at least 20 prompts."""
    gen = AdversarialGenerator(seed=42)
    results = gen.generate_all()
    assert len(results) >= 20
