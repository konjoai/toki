"""Tests for toki.dataset — AdversarialDataset."""
import tempfile
from pathlib import Path

import pytest
from toki.dataset import AdversarialDataset
from toki.generate import AdversarialGenerator, AdversarialPrompt


def _make_prompt(text: str = "hello", category: str = "jailbreak") -> AdversarialPrompt:
    return AdversarialPrompt(text=text, category=category, strategy="test", seed=0)


def test_add_returns_true_for_new():
    ds = AdversarialDataset()
    result = ds.add(_make_prompt("unique text 1"))
    assert result is True


def test_add_returns_false_for_duplicate():
    ds = AdversarialDataset()
    p = _make_prompt("duplicate text")
    ds.add(p)
    duplicate = AdversarialPrompt(text=p.text, category="injection", strategy="test", seed=99)
    result = ds.add(duplicate)
    assert result is False
    assert len(ds) == 1


def test_add_batch_deduplicates():
    ds = AdversarialDataset()
    gen = AdversarialGenerator(seed=42)
    prompts = gen.generate_jailbreaks(count=5)
    added_first = ds.add_batch(prompts)
    assert added_first == len(prompts)
    # Adding same batch again should add 0 new prompts
    added_second = ds.add_batch(prompts)
    assert added_second == 0
    assert len(ds) == len(prompts)


def test_by_category():
    ds = AdversarialDataset()
    gen = AdversarialGenerator(seed=42)
    ds.add_batch(gen.generate_all())
    jailbreaks = ds.by_category("jailbreak")
    assert len(jailbreaks) > 0
    assert all(p.category == "jailbreak" for p in jailbreaks)


def test_categories_sorted():
    ds = AdversarialDataset()
    gen = AdversarialGenerator(seed=42)
    ds.add_batch(gen.generate_all())
    cats = ds.categories()
    assert cats == sorted(cats), "categories() must return a sorted list"
    assert set(cats) == {"jailbreak", "injection", "edge_case", "boundary"}


def test_save_and_load_roundtrip():
    ds = AdversarialDataset()
    gen = AdversarialGenerator(seed=42)
    ds.add_batch(gen.generate_all())
    original_len = len(ds)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dataset.json"
        ds.save(path)
        loaded = AdversarialDataset.load(path)

    assert len(loaded) == original_len
    assert loaded.categories() == ds.categories()
    # Spot-check first prompt
    assert loaded[0].text == ds[0].text
    assert loaded[0].category == ds[0].category
    assert loaded[0].seed == ds[0].seed


def test_stats_totals():
    ds = AdversarialDataset()
    gen = AdversarialGenerator(seed=42)
    ds.add_batch(gen.generate_all())
    stats = ds.stats()
    assert stats["total"] == len(ds)
    category_sum = sum(stats["categories"].values())
    assert category_sum == stats["total"]


def test_iteration():
    ds = AdversarialDataset()
    gen = AdversarialGenerator(seed=42)
    prompts = gen.generate_jailbreaks(count=5)
    ds.add_batch(prompts)
    iterated = list(ds)
    assert len(iterated) == 5
