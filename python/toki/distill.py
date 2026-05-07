"""Attack Corpus Distiller — Phase 9.

Takes evolved Individual populations (from PromptMutator) or raw
AdversarialDataset objects and distills them into a compact, diverse,
high-fitness attack corpus.

Key design choices:
- Fitness filtering: only keep individuals above a configurable threshold.
- Category-aware bucketing: per-operator-origin buckets within each
  category; top-k selected by fitness from each bucket.
- Diversity guarantee: min_prompts floor prevents empty distillations;
  random survivors are promoted when the filter is too aggressive.
- No-overwrite: DistillResult.save() always creates a new timestamped
  directory and raises FileExistsError if it would clobber an existing one.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from toki.generate import AdversarialPrompt
from toki.dataset import AdversarialDataset
from toki.mutator import Individual

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DistillConfig:
    """Configuration for corpus distillation."""
    fitness_threshold: float = 0.4
    """Minimum fitness score to consider an individual (inclusive)."""
    max_per_category: int = 50
    """Maximum prompts retained per category after distillation."""
    min_prompts: int = 5
    """Minimum total prompts to retain (promotes survivors if threshold is too strict)."""
    max_total: int = 500
    """Hard cap on total prompts in the distilled corpus."""
    seed: Optional[int] = 42
    """RNG seed for reproducible shuffle / tie-breaking."""
    output_dir: str = "experiments/distilled"
    """Root directory where DistillResult artefacts are written."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.fitness_threshold <= 1.0:
            raise ValueError("fitness_threshold must be in [0.0, 1.0]")
        if self.max_per_category < 1:
            raise ValueError("max_per_category must be >= 1")
        if self.min_prompts < 0:
            raise ValueError("min_prompts must be >= 0")
        if self.max_total < 1:
            raise ValueError("max_total must be >= 1")
        if self.max_total < self.min_prompts:
            raise ValueError("max_total must be >= min_prompts")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class DistillStats:
    """Summary statistics for a distillation run."""
    total_input: int
    total_retained: int
    mean_fitness: float
    max_fitness: float
    retention_rate: float          # total_retained / total_input
    categories_covered: List[str]  # sorted list
    per_category: Dict[str, int]   # category -> count retained


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class DistillResult:
    """Output of a distillation run."""
    dataset: AdversarialDataset
    stats: DistillStats
    timestamp: str
    config: DistillConfig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, base_dir: Optional[str | Path] = None) -> Path:
        """Persist the distilled corpus + metadata under a new timestamped dir.

        Raises FileExistsError if the directory already exists (no overwrite).
        Returns the directory path.
        """
        root = Path(base_dir or self.config.output_dir)
        run_dir = root / f"{self.timestamp}_distilled"
        if run_dir.exists():
            raise FileExistsError(f"Distill result directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)

        # Corpus
        self.dataset.save(run_dir / "corpus.json")

        # Metadata
        meta = {
            "timestamp": self.timestamp,
            "config": asdict(self.config),
            "stats": {
                "total_input": self.stats.total_input,
                "total_retained": self.stats.total_retained,
                "mean_fitness": self.stats.mean_fitness,
                "max_fitness": self.stats.max_fitness,
                "retention_rate": self.stats.retention_rate,
                "categories_covered": self.stats.categories_covered,
                "per_category": self.stats.per_category,
            },
        }
        (run_dir / "distill_result.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("DistillResult saved to %s", run_dir)
        return run_dir

    @classmethod
    def load(cls, run_dir: str | Path) -> "DistillResult":
        """Load a persisted DistillResult from its directory."""
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "distill_result.json").read_text(encoding="utf-8"))
        dataset = AdversarialDataset.load(run_dir / "corpus.json")
        raw_cfg = meta["config"]
        config = DistillConfig(
            fitness_threshold=raw_cfg["fitness_threshold"],
            max_per_category=raw_cfg["max_per_category"],
            min_prompts=raw_cfg["min_prompts"],
            max_total=raw_cfg["max_total"],
            seed=raw_cfg.get("seed"),
            output_dir=raw_cfg["output_dir"],
        )
        raw_stats = meta["stats"]
        stats = DistillStats(
            total_input=raw_stats["total_input"],
            total_retained=raw_stats["total_retained"],
            mean_fitness=raw_stats["mean_fitness"],
            max_fitness=raw_stats["max_fitness"],
            retention_rate=raw_stats["retention_rate"],
            categories_covered=raw_stats["categories_covered"],
            per_category=raw_stats["per_category"],
        )
        return cls(
            dataset=dataset,
            stats=stats,
            timestamp=meta["timestamp"],
            config=config,
        )


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------

class CorpusDistiller:
    """Distil a high-quality, diverse attack corpus from evolved populations.

    Usage (from mutation output)::

        from toki.mutator import PromptMutator, MutationConfig
        from toki.distill import CorpusDistiller, DistillConfig

        result = PromptMutator(MutationConfig(seed=0)).evolve(seeds, "jailbreak")
        distiller = CorpusDistiller(DistillConfig(fitness_threshold=0.5))
        dr = distiller.distill_from_individuals(result.final_population, "jailbreak")

    Usage (from raw dataset + optional score map)::

        distiller = CorpusDistiller()
        dr = distiller.distill_from_dataset(my_dataset)
    """

    def __init__(self, config: Optional[DistillConfig] = None) -> None:
        self._config = config or DistillConfig()
        self._rng = random.Random(self._config.seed)

    @property
    def config(self) -> DistillConfig:
        return self._config

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def distill_from_individuals(
        self,
        individuals: Sequence[Individual],
        category: str,
    ) -> DistillResult:
        """Distil from a list of Individual objects (mutation run output).

        Parameters
        ----------
        individuals:
            Population returned by PromptMutator.evolve().
        category:
            Attack category label (e.g. ``"jailbreak"``).
        """
        items: List[tuple[str, str, float]] = [
            (ind.prompt, category, ind.fitness) for ind in individuals
        ]
        return self._run(items)

    def distill_from_dataset(
        self,
        dataset: AdversarialDataset,
        scores_by_text: Optional[Dict[str, float]] = None,
    ) -> DistillResult:
        """Distil from an AdversarialDataset.

        Parameters
        ----------
        dataset:
            Source dataset.
        scores_by_text:
            Optional mapping of ``prompt.text -> fitness score``.  When
            absent every prompt is assigned fitness 1.0 so that all
            prompts pass the threshold and the distillation only applies
            the diversity caps.
        """
        scores = scores_by_text or {}
        items: List[tuple[str, str, float]] = [
            (p.text, p.category, scores.get(p.text, 1.0))
            for p in dataset
        ]
        return self._run(items)

    def distill_multi(
        self,
        runs: List[tuple[Sequence[Individual], str]],
    ) -> DistillResult:
        """Merge multiple mutation runs (each a (population, category) pair) then distil.

        Useful when you ran the mutator across several categories and want a
        single unified corpus.
        """
        if not runs:
            raise ValueError("runs must be non-empty")
        items: List[tuple[str, str, float]] = []
        for individuals, category in runs:
            for ind in individuals:
                items.append((ind.prompt, category, ind.fitness))
        return self._run(items)

    # ------------------------------------------------------------------
    # Core distillation logic
    # ------------------------------------------------------------------

    def _run(self, items: List[tuple[str, str, float]]) -> DistillResult:
        """Core pipeline: filter → bucket → select → cap → build dataset.

        Each item is a (prompt_text, category, fitness) triple.
        """
        cfg = self._config
        total_input = len(items)

        if total_input == 0:
            return self._empty_result(total_input)

        # --- Step 1: deduplicate on text ---
        seen_texts: set[str] = set()
        deduped: List[tuple[str, str, float]] = []
        for text, cat, fit in items:
            key = text.strip()
            if key not in seen_texts:
                seen_texts.add(key)
                deduped.append((text, cat, fit))

        # --- Step 2: fitness filter ---
        passing = [
            (text, cat, fit)
            for text, cat, fit in deduped
            if fit >= cfg.fitness_threshold
        ]

        # --- Step 3: min_prompts floor (promote survivors when too few pass) ---
        if len(passing) < cfg.min_prompts and deduped:
            # Sort remaining (not already in passing) by fitness descending
            passing_texts = {text for text, _, _ in passing}
            survivors = sorted(
                [(t, c, f) for t, c, f in deduped if t not in passing_texts],
                key=lambda x: x[2],
                reverse=True,
            )
            needed = cfg.min_prompts - len(passing)
            passing = passing + survivors[:needed]

        # --- Step 4: group by category then select top-k per bucket ---
        by_category: Dict[str, List[tuple[str, float]]] = {}
        for text, cat, fit in passing:
            by_category.setdefault(cat, []).append((text, fit))

        selected: List[tuple[str, str, float]] = []
        for cat, entries in sorted(by_category.items()):
            # Sort by fitness descending, take top max_per_category
            top = sorted(entries, key=lambda x: x[1], reverse=True)[: cfg.max_per_category]
            selected.extend((text, cat, fit) for text, fit in top)

        # --- Step 5: shuffle for variety, apply max_total cap ---
        self._rng.shuffle(selected)
        selected = selected[: cfg.max_total]

        # --- Step 6: build AdversarialDataset ---
        dataset = AdversarialDataset()
        for text, cat, _fit in selected:
            dataset.add(
                AdversarialPrompt(
                    text=text,
                    category=cat,
                    strategy="distilled",
                    seed=self._stable_seed(text),
                )
            )

        # --- Step 7: compute stats ---
        fitnesses = [fit for _, _, fit in selected]
        per_cat_count: Dict[str, int] = {}
        for _, cat, _ in selected:
            per_cat_count[cat] = per_cat_count.get(cat, 0) + 1

        n_retained = len(selected)
        stats = DistillStats(
            total_input=total_input,
            total_retained=n_retained,
            mean_fitness=sum(fitnesses) / n_retained if fitnesses else 0.0,
            max_fitness=max(fitnesses) if fitnesses else 0.0,
            retention_rate=n_retained / total_input if total_input else 0.0,
            categories_covered=sorted(per_cat_count.keys()),
            per_category=per_cat_count,
        )
        timestamp = _utc_timestamp()
        return DistillResult(dataset=dataset, stats=stats, timestamp=timestamp, config=cfg)

    def _empty_result(self, total_input: int) -> DistillResult:
        stats = DistillStats(
            total_input=total_input,
            total_retained=0,
            mean_fitness=0.0,
            max_fitness=0.0,
            retention_rate=0.0,
            categories_covered=[],
            per_category={},
        )
        return DistillResult(
            dataset=AdversarialDataset(),
            stats=stats,
            timestamp=_utc_timestamp(),
            config=self._config,
        )

    @staticmethod
    def _stable_seed(text: str) -> int:
        """Deterministic seed derived from prompt text."""
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_timestamp() -> str:
    """Return a compact UTC timestamp string: YYYYMMDD_HHMMSS."""
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
