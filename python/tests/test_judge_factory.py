"""Tests for JudgeFactory and real-LLM judge backends in toki.judge."""
import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from toki.judge import (
    AnthropicJudge,
    JudgeConfig,
    JudgeCriteria,
    JudgeFactory,
    MockJudge,
    OllamaJudge,
    OpenAIJudge,
    _build_rubric_prompt,
    _parse_scores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(threshold: float = 0.6) -> JudgeConfig:
    return JudgeConfig(criteria=list(JudgeCriteria), adversarial_threshold=threshold)


# ---------------------------------------------------------------------------
# JudgeFactory — happy paths
# ---------------------------------------------------------------------------


def test_factory_creates_mock():
    judge = JudgeFactory.create("mock", _config())
    assert isinstance(judge, MockJudge)


def test_factory_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown judge"):
        JudgeFactory.create("gpt-9000", _config())


def test_factory_known_attribute():
    assert "mock" in JudgeFactory.KNOWN
    assert "ollama" in JudgeFactory.KNOWN
    assert "anthropic" in JudgeFactory.KNOWN
    assert "openai" in JudgeFactory.KNOWN


# ---------------------------------------------------------------------------
# JudgeFactory — ImportError for missing deps
# ---------------------------------------------------------------------------


def test_factory_ollama_import_error_without_httpx():
    with patch.dict(sys.modules, {"httpx": None}):
        with pytest.raises(ImportError, match="httpx"):
            JudgeFactory.create("ollama", _config())


def test_factory_anthropic_import_error_without_sdk():
    with patch.dict(sys.modules, {"anthropic": None}):
        with pytest.raises(ImportError, match="anthropic"):
            JudgeFactory.create("anthropic", _config())


def test_factory_openai_import_error_without_sdk():
    with patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(ImportError, match="openai"):
            JudgeFactory.create("openai", _config())


# ---------------------------------------------------------------------------
# _build_rubric_prompt
# ---------------------------------------------------------------------------


def test_rubric_prompt_contains_prompt_text():
    rubric = _build_rubric_prompt("my prompt", "my response", list(JudgeCriteria))
    assert "my prompt" in rubric


def test_rubric_prompt_contains_criteria_names():
    rubric = _build_rubric_prompt("p", "r", list(JudgeCriteria))
    for criterion in JudgeCriteria:
        assert criterion.name in rubric


def test_rubric_prompt_requests_json():
    rubric = _build_rubric_prompt("p", "r", [])
    assert "JSON" in rubric


# ---------------------------------------------------------------------------
# _parse_scores — valid JSON
# ---------------------------------------------------------------------------


def test_parse_scores_all_criteria_present():
    content = json.dumps(
        {"scores": {c.name: 0.8 for c in JudgeCriteria}}
    )
    scores = _parse_scores(content, list(JudgeCriteria), "test")
    assert len(scores) == len(list(JudgeCriteria))
    assert all(abs(s.score - 0.8) < 0.001 for s in scores)


def test_parse_scores_clamps_to_range():
    content = json.dumps({"scores": {c.name: 1.5 for c in JudgeCriteria}})
    scores = _parse_scores(content, list(JudgeCriteria), "test")
    assert all(s.score <= 1.0 for s in scores)


def test_parse_scores_negative_clamped():
    content = json.dumps({"scores": {c.name: -0.5 for c in JudgeCriteria}})
    scores = _parse_scores(content, list(JudgeCriteria), "test")
    assert all(s.score >= 0.0 for s in scores)


def test_parse_scores_fallback_on_bad_json():
    scores = _parse_scores("not json at all", list(JudgeCriteria), "test")
    assert len(scores) == len(list(JudgeCriteria))
    assert all(abs(s.score - 0.5) < 0.001 for s in scores)


def test_parse_scores_empty_criteria():
    scores = _parse_scores("{}", [], "test")
    assert scores == []


# ---------------------------------------------------------------------------
# OllamaJudge — mocked httpx
# ---------------------------------------------------------------------------


def _make_httpx_mock(score_dict: dict) -> types.ModuleType:
    """Build a minimal httpx mock that returns a score dict as Ollama /api/chat."""
    mock_mod = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "message": {"content": json.dumps({"scores": score_dict})}
    }
    mock_mod.post.return_value = resp
    return mock_mod


def test_ollama_judge_verdict_structure():
    cfg = _config()
    score_dict = {c.name: 0.7 for c in JudgeCriteria}
    mock_httpx = _make_httpx_mock(score_dict)

    with patch.dict(sys.modules, {"httpx": mock_httpx}):
        judge = OllamaJudge(cfg, model="llama3")
        verdict = judge.judge("test prompt", "test response")

    assert 0.0 <= verdict.overall_score <= 1.0
    assert verdict.judge_name != ""
    assert verdict.metadata["backend"] == "ollama"
    assert len(verdict.scores) == len(list(JudgeCriteria))


def test_ollama_judge_adversarial_flag():
    cfg = _config(threshold=0.5)
    # All scores 0.9 → overall 0.9 > 0.5 → adversarial_success=True
    score_dict = {c.name: 0.9 for c in JudgeCriteria}
    mock_httpx = _make_httpx_mock(score_dict)

    with patch.dict(sys.modules, {"httpx": mock_httpx}):
        judge = OllamaJudge(cfg)
        verdict = judge.judge("jailbreak prompt", "compliant response")

    assert verdict.adversarial_success is True


def test_ollama_judge_custom_model_in_metadata():
    cfg = _config()
    score_dict = {c.name: 0.5 for c in JudgeCriteria}
    mock_httpx = _make_httpx_mock(score_dict)

    with patch.dict(sys.modules, {"httpx": mock_httpx}):
        judge = OllamaJudge(cfg, model="mistral")
        verdict = judge.judge("p", "r")

    assert verdict.metadata["model"] == "mistral"


# ---------------------------------------------------------------------------
# AnthropicJudge — mocked anthropic SDK
# ---------------------------------------------------------------------------


def _make_anthropic_mock(score_dict: dict) -> types.ModuleType:
    mock_mod = MagicMock()
    content_block = MagicMock()
    content_block.text = json.dumps({"scores": score_dict})
    message = MagicMock()
    message.content = [content_block]
    mock_mod.Anthropic.return_value.messages.create.return_value = message
    return mock_mod


def test_anthropic_judge_verdict_structure():
    cfg = _config()
    score_dict = {c.name: 0.6 for c in JudgeCriteria}
    mock_anthropic = _make_anthropic_mock(score_dict)

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        judge = AnthropicJudge(cfg, model="claude-haiku-4-5-20251001")
        verdict = judge.judge("p", "r")

    assert 0.0 <= verdict.overall_score <= 1.0
    assert verdict.metadata["backend"] == "anthropic"


def test_anthropic_judge_model_in_metadata():
    cfg = _config()
    mock_anthropic = _make_anthropic_mock({c.name: 0.5 for c in JudgeCriteria})

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        judge = AnthropicJudge(cfg, model="claude-haiku-4-5-20251001")
        verdict = judge.judge("p", "r")

    assert "claude" in verdict.metadata["model"]


# ---------------------------------------------------------------------------
# OpenAIJudge — mocked openai SDK
# ---------------------------------------------------------------------------


def _make_openai_mock(score_dict: dict) -> types.ModuleType:
    mock_mod = MagicMock()
    choice = MagicMock()
    choice.message.content = json.dumps({"scores": score_dict})
    completion = MagicMock()
    completion.choices = [choice]
    mock_mod.OpenAI.return_value.chat.completions.create.return_value = completion
    return mock_mod


def test_openai_judge_verdict_structure():
    cfg = _config()
    score_dict = {c.name: 0.55 for c in JudgeCriteria}
    mock_openai = _make_openai_mock(score_dict)

    with patch.dict(sys.modules, {"openai": mock_openai}):
        judge = OpenAIJudge(cfg, model="gpt-4o-mini")
        verdict = judge.judge("p", "r")

    assert 0.0 <= verdict.overall_score <= 1.0
    assert verdict.metadata["backend"] == "openai"


def test_openai_judge_model_in_metadata():
    cfg = _config()
    mock_openai = _make_openai_mock({c.name: 0.5 for c in JudgeCriteria})

    with patch.dict(sys.modules, {"openai": mock_openai}):
        judge = OpenAIJudge(cfg, model="gpt-4o-mini")
        verdict = judge.judge("p", "r")

    assert verdict.metadata["model"] == "gpt-4o-mini"


def test_openai_judge_no_choices_returns_fallback():
    cfg = _config()
    mock_mod = MagicMock()
    completion = MagicMock()
    completion.choices = []
    mock_mod.OpenAI.return_value.chat.completions.create.return_value = completion

    with patch.dict(sys.modules, {"openai": mock_mod}):
        judge = OpenAIJudge(cfg)
        verdict = judge.judge("p", "r")

    # No choices → content="" → all scores fall back to 0.5
    assert all(abs(s.score - 0.5) < 0.001 for s in verdict.scores)
