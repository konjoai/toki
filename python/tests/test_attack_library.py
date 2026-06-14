"""Tests for toki.attack_library — Phase 14 Custom Attack Library."""
import json
import pytest

from toki.attack_library import AttackLibrary, CustomAttack, VALID_CATEGORIES


# ---------------------------------------------------------------------------
# CustomAttack construction
# ---------------------------------------------------------------------------


def test_custom_attack_auto_id():
    a = CustomAttack(text="ignore previous instructions", category="jailbreak")
    assert len(a.id) == 16
    assert a.id == a.id  # deterministic


def test_custom_attack_id_deterministic():
    a1 = CustomAttack(text="same text", category="jailbreak")
    a2 = CustomAttack(text="same text", category="injection")
    # ID is hash of text only — same text → same id regardless of category
    assert a1.id == a2.id


def test_custom_attack_created_is_set():
    a = CustomAttack(text="test", category="jailbreak")
    assert "T" in a.created  # ISO-8601 UTC has a T separator


def test_custom_attack_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        CustomAttack(text="   ", category="jailbreak")


def test_custom_attack_invalid_category_raises():
    with pytest.raises(ValueError, match="Unknown category"):
        CustomAttack(text="some text", category="nonsense")


def test_custom_attack_all_valid_categories():
    for cat in VALID_CATEGORIES:
        a = CustomAttack(text=f"prompt for {cat}", category=cat)
        assert a.category == cat


def test_custom_attack_defaults():
    a = CustomAttack(text="test prompt", category="custom")
    assert a.language == "en"
    assert a.expected_refusal is True
    assert a.provenance == "user"
    assert a.notes == ""


# ---------------------------------------------------------------------------
# AttackLibrary — in-memory (tmp path)
# ---------------------------------------------------------------------------


def test_add_returns_true_for_new(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    a = CustomAttack(text="new attack", category="jailbreak")
    assert lib.add(a) is True


def test_add_returns_false_for_duplicate(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    a = CustomAttack(text="dup attack", category="jailbreak")
    lib.add(a)
    result = lib.add(a)
    assert result is False
    assert len(lib) == 1


def test_list_attacks_empty(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    assert lib.list_attacks() == []


def test_list_attacks_returns_all(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    lib.add(CustomAttack(text="attack one", category="jailbreak"))
    lib.add(CustomAttack(text="attack two", category="injection"))
    attacks = lib.list_attacks()
    assert len(attacks) == 2


def test_list_attacks_category_filter(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    lib.add(CustomAttack(text="jb attack", category="jailbreak"))
    lib.add(CustomAttack(text="inj attack", category="injection"))
    jb = lib.list_attacks(category="jailbreak")
    assert len(jb) == 1
    assert jb[0].category == "jailbreak"


def test_list_attacks_sorted_by_created(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    lib.add(CustomAttack(text="first", category="jailbreak"))
    lib.add(CustomAttack(text="second attack text", category="injection"))
    attacks = lib.list_attacks()
    assert len(attacks) == 2
    # created timestamps should be non-decreasing
    assert attacks[0].created <= attacks[1].created


def test_remove_existing(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    a = CustomAttack(text="removable", category="jailbreak")
    lib.add(a)
    assert lib.remove(a.id) is True
    assert len(lib) == 0


def test_remove_nonexistent(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    assert lib.remove("nonexistent_id") is False


def test_get_existing(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    a = CustomAttack(text="get me", category="custom")
    lib.add(a)
    found = lib.get(a.id)
    assert found is not None
    assert found.text == "get me"


def test_get_missing_returns_none(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    assert lib.get("does_not_exist") is None


def test_stats_total_and_by_category(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    lib.add(CustomAttack(text="jb1", category="jailbreak"))
    lib.add(CustomAttack(text="jb2", category="jailbreak"))
    lib.add(CustomAttack(text="inj1", category="injection"))
    stats = lib.stats()
    assert stats["total"] == 3
    assert stats["by_category"]["jailbreak"] == 2
    assert stats["by_category"]["injection"] == 1


def test_len_reflects_count(tmp_path):
    lib = AttackLibrary(tmp_path / "attacks.json")
    assert len(lib) == 0
    lib.add(CustomAttack(text="a", category="jailbreak"))
    assert len(lib) == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "attacks.json"
    lib1 = AttackLibrary(path)
    lib1.add(CustomAttack(text="persistent attack", category="jailbreak"))

    lib2 = AttackLibrary(path)
    assert len(lib2) == 1
    assert lib2.list_attacks()[0].text == "persistent attack"


def test_persistence_file_is_valid_json(tmp_path):
    path = tmp_path / "attacks.json"
    lib = AttackLibrary(path)
    lib.add(CustomAttack(text="json test", category="custom"))
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["text"] == "json test"


def test_persistence_dedup_across_instances(tmp_path):
    path = tmp_path / "attacks.json"
    lib1 = AttackLibrary(path)
    lib1.add(CustomAttack(text="dedup me", category="jailbreak"))

    lib2 = AttackLibrary(path)
    result = lib2.add(CustomAttack(text="dedup me", category="injection"))
    assert result is False
    assert len(lib2) == 1


def test_load_missing_file_starts_empty(tmp_path):
    lib = AttackLibrary(tmp_path / "does_not_exist.json")
    assert len(lib) == 0
