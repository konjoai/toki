"""Dataset management: load, save, and deduplicate adversarial prompts."""
from __future__ import annotations

import json
from pathlib import Path

from toki.generate import AdversarialPrompt


class AdversarialDataset:
    """In-memory collection of AdversarialPrompt objects with deduplication.

    Deduplication key: stripped prompt text.
    """

    def __init__(self) -> None:
        self._prompts: list[AdversarialPrompt] = []
        self._seen: set[str] = set()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, prompt: AdversarialPrompt) -> bool:
        """Add a prompt.  Returns True if added, False if duplicate."""
        key = prompt.text.strip()
        if key in self._seen:
            return False
        self._seen.add(key)
        self._prompts.append(prompt)
        return True

    def add_batch(self, prompts: list[AdversarialPrompt]) -> int:
        """Add a batch.  Returns count of newly added (non-duplicate) prompts."""
        return sum(1 for p in prompts if self.add(p))

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._prompts)

    def __iter__(self):
        return iter(self._prompts)

    def __getitem__(self, idx: int) -> AdversarialPrompt:
        return self._prompts[idx]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def by_category(self, category: str) -> list[AdversarialPrompt]:
        """Return all prompts in a given category."""
        return [p for p in self._prompts if p.category == category]

    def categories(self) -> list[str]:
        """Return sorted list of distinct categories present in the dataset."""
        return sorted({p.category for p in self._prompts})

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the dataset to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "text": p.text,
                "category": p.category,
                "strategy": p.strategy,
                "seed": p.seed,
            }
            for p in self._prompts
        ]
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AdversarialDataset":
        """Deserialize a dataset from a JSON file."""
        path = Path(path)
        records: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        ds = cls()
        for r in records:
            ds.add(
                AdversarialPrompt(
                    text=r["text"],
                    category=r["category"],
                    strategy=r["strategy"],
                    seed=r["seed"],
                )
            )
        return ds

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary dict with total count and per-category counts."""
        cats = self.categories()
        return {
            "total": len(self),
            "categories": {cat: len(self.by_category(cat)) for cat in cats},
        }
