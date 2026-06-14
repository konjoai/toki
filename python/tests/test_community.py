"""Tests for toki.community — Phase 15 Community Attack Registry."""
import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from toki.community import (
    CommunityAttack,
    CommunityRegistry,
    _verify_sha256,
    filter_attacks,
    get_registry,
    load_bundled,
    load_remote,
)


# ---------------------------------------------------------------------------
# Bundled registry
# ---------------------------------------------------------------------------


def test_load_bundled_returns_registry():
    reg = load_bundled()
    assert isinstance(reg, CommunityRegistry)


def test_bundled_registry_has_25_attacks():
    reg = load_bundled()
    assert reg.total == 25


def test_bundled_registry_all_categories_covered():
    reg = load_bundled()
    cats = {a.category for a in reg.attacks}
    expected = {"jailbreak", "injection", "edge_case", "boundary", "indirect", "agentic"}
    assert cats == expected


def test_bundled_registry_sha256_valid():
    reg = load_bundled()
    assert len(reg.sha256) == 64  # hex SHA-256


def test_bundled_attacks_have_required_fields():
    reg = load_bundled()
    for attack in reg.attacks:
        assert attack.id.startswith("com-")
        assert len(attack.text) > 0
        assert attack.severity in {"critical", "high", "medium", "low"}
        assert "OWASP" in attack.owasp_tag
        assert isinstance(attack.tags, list)


def test_bundled_attacks_all_expect_refusal():
    reg = load_bundled()
    assert all(a.expected_refusal for a in reg.attacks)


# ---------------------------------------------------------------------------
# get_registry — caching
# ---------------------------------------------------------------------------


def test_get_registry_returns_registry():
    reg = get_registry()
    assert isinstance(reg, CommunityRegistry)


def test_get_registry_cached():
    reg1 = get_registry()
    reg2 = get_registry()
    assert reg1 is reg2


def test_get_registry_reload():
    reg1 = get_registry()
    reg2 = get_registry(reload=True)
    assert reg2.total == reg1.total


# ---------------------------------------------------------------------------
# CommunityRegistry.filter
# ---------------------------------------------------------------------------


def test_filter_by_category():
    reg = load_bundled()
    jb = reg.filter(category="jailbreak")
    assert len(jb) == 5
    assert all(a.category == "jailbreak" for a in jb)


def test_filter_by_severity():
    reg = load_bundled()
    critical = reg.filter(severity="critical")
    assert len(critical) > 0
    assert all(a.severity == "critical" for a in critical)


def test_filter_by_tag():
    reg = load_bundled()
    dan = reg.filter(tag="DAN")
    assert len(dan) >= 1
    assert all("DAN" in a.tags for a in dan)


def test_filter_combined():
    reg = load_bundled()
    results = reg.filter(category="jailbreak", severity="critical")
    assert all(a.category == "jailbreak" and a.severity == "critical" for a in results)


def test_filter_no_match_returns_empty():
    reg = load_bundled()
    assert reg.filter(tag="nonexistent_tag_xyz") == []


def test_filter_attacks_convenience():
    reg = load_bundled()
    result = filter_attacks(reg, category="agentic")
    assert len(result) == 4


# ---------------------------------------------------------------------------
# CommunityRegistry.stats
# ---------------------------------------------------------------------------


def test_stats_total_matches():
    reg = load_bundled()
    stats = reg.stats()
    assert stats["total"] == reg.total


def test_stats_by_category_covers_all():
    reg = load_bundled()
    stats = reg.stats()
    assert "jailbreak" in stats["by_category"]
    assert "agentic" in stats["by_category"]


def test_stats_by_severity_present():
    reg = load_bundled()
    stats = reg.stats()
    assert len(stats["by_severity"]) > 0


# ---------------------------------------------------------------------------
# CommunityAttack.to_dict
# ---------------------------------------------------------------------------


def test_community_attack_to_dict():
    attack = load_bundled().attacks[0]
    d = attack.to_dict()
    assert "id" in d
    assert "text" in d
    assert "category" in d
    assert "tags" in d
    assert "severity" in d
    assert "owasp_tag" in d


# ---------------------------------------------------------------------------
# _verify_sha256
# ---------------------------------------------------------------------------


def test_verify_sha256_correct():
    attacks_data = [{"text": "test", "category": "jailbreak"}]
    declared = hashlib.sha256(
        json.dumps(attacks_data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert _verify_sha256(attacks_data, declared) is True


def test_verify_sha256_wrong():
    attacks_data = [{"text": "test"}]
    assert _verify_sha256(attacks_data, "a" * 64) is False


# ---------------------------------------------------------------------------
# load_remote — mocked urllib
# ---------------------------------------------------------------------------


def _make_remote_manifest(attacks: list) -> bytes:
    sha256 = hashlib.sha256(
        json.dumps(attacks, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "version": "1.0.0",
        "updated": "2026-06-14",
        "description": "Test registry",
        "sha256": sha256,
        "attacks": attacks,
    }
    return json.dumps(manifest).encode("utf-8")


def test_load_remote_success():
    attacks = [
        {
            "id": "com-jb-test",
            "text": "ignore instructions",
            "category": "jailbreak",
            "tags": ["test"],
            "severity": "high",
            "owasp_tag": "OWASP-LLM01:2025",
            "author": "test",
            "source": "test",
            "expected_refusal": True,
        }
    ]
    payload = _make_remote_manifest(attacks)
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        reg = load_remote("https://example.com/registry.json")

    assert reg.total == 1
    assert reg.attacks[0].id == "com-jb-test"


def test_load_remote_sha256_mismatch_raises():
    attacks = [{"id": "x", "text": "t", "category": "jailbreak",
                "tags": [], "severity": "low", "owasp_tag": "", "author": "", "source": ""}]
    payload = _make_remote_manifest(attacks)
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            load_remote("https://example.com/registry.json", expected_sha256="a" * 64)
