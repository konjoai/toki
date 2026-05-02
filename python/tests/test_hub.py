"""Tests for toki.hub — dataset metadata, card rendering, upload orchestration."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from toki import __version__
from toki.dataset import AdversarialDataset
from toki.generate import AdversarialPrompt
from toki.hub import (
    DatasetMetadata,
    HubUploader,
    build_dataset_card,
    to_hf_dataset,
    write_card,
)


def _sample_dataset() -> AdversarialDataset:
    ds = AdversarialDataset()
    ds.add(AdversarialPrompt(text="jail-1", category="jailbreak", strategy="t1", seed=1))
    ds.add(AdversarialPrompt(text="jail-2", category="jailbreak", strategy="t1", seed=2))
    ds.add(AdversarialPrompt(text="inj-1", category="injection", strategy="t2", seed=3))
    return ds


def test_metadata_defaults_filled():
    md = DatasetMetadata(name="toki-adv")
    assert md.name == "toki-adv"
    assert md.version == "0.1.0"
    assert md.toki_version == __version__
    assert md.license == "BUSL-1.1"
    assert "adversarial" in md.tags
    # created auto-populated as ISO-8601 UTC string
    assert md.created.endswith("Z")
    assert "T" in md.created


def test_metadata_explicit_created_preserved():
    md = DatasetMetadata(name="x", created="2025-01-01T00:00:00Z")
    assert md.created == "2025-01-01T00:00:00Z"


def test_build_card_contains_stats_and_frontmatter():
    ds = _sample_dataset()
    md = DatasetMetadata(name="toki-adv-v1", version="1.2.3")
    card = build_dataset_card(ds.stats(), md)

    # YAML frontmatter
    assert card.startswith("---\n")
    assert "license: busl-1.1" in card
    assert "dataset_version: 1.2.3" in card
    assert f"toki_version: {__version__}" in card

    # Title + description
    assert "# toki-adv-v1" in card

    # Category table — sorted
    assert "| jailbreak | 2 |" in card
    assert "| injection | 1 |" in card
    # Total
    assert "**Total prompts:** 3" in card


def test_build_card_handles_empty_dataset():
    ds = AdversarialDataset()
    md = DatasetMetadata(name="empty")
    card = build_dataset_card(ds.stats(), md)
    assert "**Total prompts:** 0" in card
    assert "| (none) | 0 |" in card


def test_write_card_round_trip(tmp_path: Path):
    ds = _sample_dataset()
    md = DatasetMetadata(name="toki-adv", version="0.4.0")
    out = tmp_path / "sub" / "README.md"
    written = write_card(ds, md, out)
    assert written == out
    assert out.exists()
    assert "# toki-adv" in out.read_text()


def test_to_hf_dataset_raises_without_datasets(monkeypatch):
    """If `datasets` is not importable, to_hf_dataset must raise ImportError clearly."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    ds = _sample_dataset()
    with pytest.raises(ImportError, match=r"toki\[hf\]"):
        to_hf_dataset(ds)


def test_uploader_orchestrates_create_push_card(monkeypatch):
    """HubUploader.upload must: create_repo, push_to_hub, upload README.md card."""
    calls: list[tuple[str, dict]] = []

    class FakeHfApi:
        def __init__(self, token=None):
            calls.append(("HfApi.__init__", {"token": token}))

        def create_repo(self, **kw):
            calls.append(("create_repo", kw))

        def upload_file(self, **kw):
            # capture content separately so the assertion is readable
            content = kw.pop("path_or_fileobj")
            kw["_content_len"] = len(content)
            kw["_content_starts_with_frontmatter"] = content.startswith(b"---\n")
            calls.append(("upload_file", kw))

    class FakeHFDataset:
        def __init__(self, records):
            self.records = records

        @classmethod
        def from_dict(cls, records):
            calls.append(("HFDataset.from_dict", {"keys": sorted(records.keys()),
                                                   "n": len(records["text"])}))
            return cls(records)

        def push_to_hub(self, repo_id, token=None, commit_message=None):
            calls.append(("push_to_hub", {
                "repo_id": repo_id,
                "token": token,
                "commit_message": commit_message,
            }))

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeHfApi
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = FakeHFDataset

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    ds = _sample_dataset()
    md = DatasetMetadata(name="toki-adv", version="0.4.0")
    uploader = HubUploader(repo_id="user/toki-adv", token="hf_xxx", private=True)
    summary = uploader.upload(ds, md)

    # Order: HfApi → create_repo → from_dict → push_to_hub → upload_file
    names = [c[0] for c in calls]
    assert names == [
        "HfApi.__init__",
        "create_repo",
        "HFDataset.from_dict",
        "push_to_hub",
        "upload_file",
    ]

    assert calls[0][1] == {"token": "hf_xxx"}
    assert calls[1][1] == {
        "repo_id": "user/toki-adv",
        "repo_type": "dataset",
        "private": True,
        "exist_ok": True,
    }
    assert calls[2][1] == {"keys": ["category", "seed", "strategy", "text"], "n": 3}
    assert calls[3][1]["repo_id"] == "user/toki-adv"
    assert calls[3][1]["token"] == "hf_xxx"
    assert "0.4.0" in calls[3][1]["commit_message"]

    upload_kw = calls[4][1]
    assert upload_kw["repo_id"] == "user/toki-adv"
    assert upload_kw["repo_type"] == "dataset"
    assert upload_kw["path_in_repo"] == "README.md"
    assert upload_kw["_content_starts_with_frontmatter"] is True
    assert upload_kw["_content_len"] > 100

    # Summary dict
    assert summary == {
        "repo_id": "user/toki-adv",
        "dataset_version": "0.4.0",
        "toki_version": __version__,
        "total_prompts": 3,
        "categories": {"jailbreak": 2, "injection": 1},
    }


def test_uploader_custom_commit_message(monkeypatch):
    pushed: dict = {}

    class FakeHfApi:
        def __init__(self, token=None): pass
        def create_repo(self, **kw): pass
        def upload_file(self, **kw): pass

    class FakeHFDataset:
        @classmethod
        def from_dict(cls, records):
            return cls()

        def push_to_hub(self, repo_id, token=None, commit_message=None):
            pushed["msg"] = commit_message

    fake_hub = types.ModuleType("huggingface_hub"); fake_hub.HfApi = FakeHfApi
    fake_datasets = types.ModuleType("datasets"); fake_datasets.Dataset = FakeHFDataset
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    HubUploader("u/r").upload(_sample_dataset(), DatasetMetadata(name="x"), commit_message="custom!")
    assert pushed["msg"] == "custom!"


def test_uploader_raises_without_huggingface_hub(monkeypatch):
    """Without huggingface_hub installed, upload must raise a guiding ImportError."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)

    with pytest.raises(ImportError, match=r"toki\[hf\]"):
        HubUploader("u/r").upload(_sample_dataset(), DatasetMetadata(name="x"))
