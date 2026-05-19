"""Tests for ChatInjectGenerator — chat template role-boundary injection cases."""
from __future__ import annotations

from toki.multilingual import (
    ChatInjectGenerator,
    MultilingualPrompt,
    generate_chat_inject_battery,
)


def test_generate_all_returns_20_cases():
    assert len(ChatInjectGenerator().generate_all()) == 20


def test_module_convenience_returns_20_cases():
    assert len(generate_chat_inject_battery()) == 20


def test_all_cases_are_multilingual_prompt_instances():
    for p in ChatInjectGenerator().generate_all():
        assert isinstance(p, MultilingualPrompt)


def test_all_cases_have_chat_template_encoding():
    for p in ChatInjectGenerator().generate_all():
        assert p.encoding == "chat_template"


def test_all_cases_have_chat_template_strategy():
    for p in ChatInjectGenerator().generate_all():
        assert p.strategy == "chat_template"


def test_all_cases_have_encoding_category():
    for p in ChatInjectGenerator().generate_all():
        assert p.category == "encoding"


def test_all_cases_are_english():
    for p in ChatInjectGenerator().generate_all():
        assert p.language == "en"


def test_all_cases_have_high_severity():
    for p in ChatInjectGenerator().generate_all():
        assert p.severity == "high"


def test_all_seeds_are_distinct():
    seeds = [p.seed for p in ChatInjectGenerator().generate_all()]
    assert len(set(seeds)) == 20


def test_all_texts_are_distinct():
    texts = [p.text for p in ChatInjectGenerator().generate_all()]
    assert len(set(texts)) == 20


def test_generate_all_is_deterministic():
    g = ChatInjectGenerator()
    a = g.generate_all()
    b = g.generate_all()
    assert [p.text for p in a] == [p.text for p in b]
    assert [p.seed for p in a] == [p.seed for p in b]


def test_cases_contain_role_boundary_markers():
    texts = [p.text for p in ChatInjectGenerator().generate_all()]
    role_markers = {
        "[SYSTEM]", "[INST]", "<|im_start|>", "<|system|>",
        "<<SYS>>", "<|begin_of_text|>", "### System",
    }
    matched = [t for t in texts if any(m in t for m in role_markers)]
    assert len(matched) >= 10, "Expected ≥10 cases to contain role boundary markers"
