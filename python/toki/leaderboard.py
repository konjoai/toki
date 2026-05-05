"""
Multi-model adversarial leaderboard.

Runs all-pairs (round-robin) comparisons across k ≥ 2 models, applies
Bonferroni correction to the per-pair significance threshold, and ranks
models by mean safety score with asterisk markers for statistically
significant placement.

Mathematical foundation
-----------------------
For k models there are k*(k-1)/2 unique unordered pairs.  The family-wise
error rate (FWER) under the null is controlled by dividing the nominal α
by the number of comparisons (Bonferroni):

    α_bonferroni = α / (k * (k-1) / 2)

Each pair is evaluated with the paired t-test + Wilcoxon signed-rank
test from :mod:`toki.benchmark`.  A pair comparison is considered
significant when at least one of the two tests rejects H0 at α_bonferroni.

Ranking: models are sorted by descending mean score.  Rank is 1-indexed.
A model's ``significant`` flag is ``True`` when every pairwise comparison
involving that model that the leaderboard considers "won" is statistically
significant after Bonferroni correction.

Pure-stdlib core — no torch / transformers / numpy / scipy required.
The model callables can be any ``str → str`` function: real LLM clients,
mocks, or deterministic fakes.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from toki.benchmark import paired_t_test, wilcoxon_test
from toki.compare import (
    BASELINES,
    ModelSpec,
    ModelScores,
    _category_winners,
    _evaluate_model,
)
from toki.dataset import AdversarialDataset
from toki.generate import AdversarialGenerator
from toki.results import ExperimentResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LeaderboardEntry:
    """A single model's position in the leaderboard."""

    name: str
    mean_score: float
    n_comparisons: int           # number of head-to-head pairs this model is in
    wins: int                    # pairwise wins (higher mean, significant)
    losses: int                  # pairwise losses
    ties: int                    # pairwise ties (non-significant or equal mean)
    rank: int                    # 1-indexed, 1 = safest
    significant: bool            # all wins are statistically significant after Bonferroni


@dataclass
class PairResult:
    """Raw outcome of a single head-to-head comparison."""

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    winner: str             # name_a, name_b, or "tie"
    significant: bool       # at least one paired test rejects H0 at α_bonferroni
    t_stat: float
    t_p_value: float
    w_stat: float
    w_p_value: float
    alpha_bonferroni: float


@dataclass
class LeaderboardConfig:
    """Configuration for a leaderboard run."""

    name: str = "leaderboard"
    seed: int = 42
    jailbreak_count: int = 10
    injection_count: int = 10
    boundary_count: int = 5
    alpha: float = 0.05         # nominal family-wise α before Bonferroni correction
    output_dir: str = "experiments/leaderboards"


@dataclass
class LeaderboardResult:
    """Full leaderboard result — rankings, all pairwise outcomes, config snapshot."""

    name: str
    timestamp: str
    config: dict
    entries: List[LeaderboardEntry]          # ordered by rank (ascending)
    pairs: List[PairResult]                  # all k*(k-1)/2 pair comparisons
    alpha_bonferroni: float                  # corrected significance threshold
    n_models: int
    n_pairs: int

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, base_dir: Optional[str] = None) -> Path:
        """Persist to ``<base_dir>/<timestamp>_<name>/leaderboard.json``.

        Raises :exc:`FileExistsError` if the output file already exists
        (consistent with ExperimentResult.save() semantics — never overwrite).
        """
        out_dir = Path(base_dir or self.config.get("output_dir", "experiments/leaderboards"))
        run_dir = out_dir / f"{self.timestamp}_{self.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / "leaderboard.json"
        if out.exists():
            raise FileExistsError(f"Leaderboard result already exists at {out}")
        out.write_text(json.dumps(asdict(self), indent=2))
        return out

    @classmethod
    def load(cls, path) -> "LeaderboardResult":
        """Load from a previously saved ``leaderboard.json``."""
        data = json.loads(Path(path).read_text())
        data["entries"] = [LeaderboardEntry(**e) for e in data["entries"]]
        data["pairs"]   = [PairResult(**p)       for p in data["pairs"]]
        return cls(**data)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def format_table(self) -> str:
        """Return a human-readable leaderboard table (ASCII, no external deps)."""
        lines: List[str] = []
        sep = "-" * 72
        lines.append(sep)
        lines.append(
            f"  Leaderboard: {self.name}   ({self.timestamp})"
        )
        lines.append(
            f"  Models: {self.n_models}   Pairs: {self.n_pairs}   "
            f"α_bonf = {self.alpha_bonferroni:.4g}"
        )
        lines.append(sep)
        header = (
            f"  {'Rank':>4}  {'Model':>14}  {'Mean':>6}  "
            f"{'W':>3}{'L':>3}{'T':>3}  {'Sig?':>5}"
        )
        lines.append(header)
        lines.append(sep)
        for e in self.entries:
            sig_marker = "*" if e.significant else " "
            lines.append(
                f"  {e.rank:>4}  {e.name:>14}  {e.mean_score:>6.4f}  "
                f"{e.wins:>3}{e.losses:>3}{e.ties:>3}  {sig_marker:>5}"
            )
        lines.append(sep)
        lines.append("  * = all pairwise wins significant after Bonferroni correction")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bonferroni_alpha(nominal_alpha: float, n_pairs: int) -> float:
    """α_bonferroni = α / n_pairs.  Returns nominal α if n_pairs == 0."""
    if n_pairs == 0:
        return nominal_alpha
    return nominal_alpha / n_pairs


def _compare_pair(
    scores_a: ModelScores,
    scores_b: ModelScores,
    alpha_bonf: float,
) -> PairResult:
    """Run paired t-test + Wilcoxon on two pre-computed :class:`ModelScores`.

    The significance decision uses ``alpha_bonf`` (already corrected).
    """
    raw_a = scores_a.scores
    raw_b = scores_b.scores

    # Paired tests require n >= 2 and matching lengths
    if len(raw_a) >= 2 and len(raw_a) == len(raw_b):
        t_res = paired_t_test(raw_a, raw_b, alpha=alpha_bonf)
        w_res = wilcoxon_test(raw_a, raw_b, alpha=alpha_bonf)
    else:
        # Degenerate case — no statistical tests possible; treat as tie
        from toki.benchmark import StatTestResult
        t_res = StatTestResult("paired_t_test", 0.0, 1.0, False, alpha_bonf, len(raw_a))
        w_res = StatTestResult("wilcoxon",      0.0, 1.0, False, alpha_bonf, len(raw_a))

    significant = t_res.significant or w_res.significant
    eps = 1e-9
    delta = scores_b.mean_score - scores_a.mean_score
    if not significant or abs(delta) < eps:
        winner = "tie"
    else:
        winner = scores_b.name if delta > 0 else scores_a.name

    return PairResult(
        name_a=scores_a.name,
        name_b=scores_b.name,
        mean_a=scores_a.mean_score,
        mean_b=scores_b.mean_score,
        winner=winner,
        significant=significant,
        t_stat=t_res.statistic,
        t_p_value=t_res.p_value,
        w_stat=w_res.statistic,
        w_p_value=w_res.p_value,
        alpha_bonferroni=alpha_bonf,
    )


def _rank_entries(
    all_scores: Dict[str, ModelScores],
    pairs: List[PairResult],
) -> List[LeaderboardEntry]:
    """Build :class:`LeaderboardEntry` objects and assign 1-indexed ranks.

    Ranking criterion: descending mean score (higher = safer).
    Ties in mean score share the same rank; the next rank is bumped.

    A model's ``significant`` flag is True when it has ≥ 1 win and every
    one of its wins is statistically significant at the Bonferroni-corrected
    threshold.
    """
    # Tally wins / losses / ties per model
    wins:   Dict[str, int] = {n: 0 for n in all_scores}
    losses: Dict[str, int] = {n: 0 for n in all_scores}
    ties:   Dict[str, int] = {n: 0 for n in all_scores}
    # Track whether every win for each model was significant
    win_all_sig: Dict[str, bool] = {n: True for n in all_scores}

    for pair in pairs:
        if pair.winner == "tie":
            ties[pair.name_a] += 1
            ties[pair.name_b] += 1
        elif pair.winner == pair.name_a:
            wins[pair.name_a]   += 1
            losses[pair.name_b] += 1
            if not pair.significant:
                win_all_sig[pair.name_a] = False
        else:
            wins[pair.name_b]   += 1
            losses[pair.name_a] += 1
            if not pair.significant:
                win_all_sig[pair.name_b] = False

    # Sort by mean score descending, stable (name as secondary key for determinism)
    sorted_names = sorted(
        all_scores.keys(),
        key=lambda n: (-all_scores[n].mean_score, n),
    )

    entries: List[LeaderboardEntry] = []
    rank = 1
    for i, name in enumerate(sorted_names):
        if i > 0:
            prev = all_scores[sorted_names[i - 1]].mean_score
            curr = all_scores[name].mean_score
            if abs(prev - curr) > 1e-9:
                rank = i + 1   # advance rank only when scores actually differ
        n_cmp = wins[name] + losses[name] + ties[name]
        has_wins = wins[name] > 0
        entries.append(
            LeaderboardEntry(
                name=name,
                mean_score=all_scores[name].mean_score,
                n_comparisons=n_cmp,
                wins=wins[name],
                losses=losses[name],
                ties=ties[name],
                rank=rank,
                significant=has_wins and win_all_sig[name],
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Leaderboard:
    """Multi-model adversarial leaderboard.

    Usage::

        from toki.leaderboard import Leaderboard, LeaderboardConfig
        from toki.compare import BASELINES, ModelSpec

        models = [ModelSpec(name, fn) for name, fn in BASELINES.items()]
        cfg = LeaderboardConfig(name="demo", seed=42)
        lb = Leaderboard(models, cfg)
        result = lb.run(save=True)
        print(result.format_table())
    """

    def __init__(
        self,
        models: List[ModelSpec],
        config: Optional[LeaderboardConfig] = None,
    ) -> None:
        if len(models) < 2:
            raise ValueError("Leaderboard requires at least 2 models")
        names = [m.name for m in models]
        if len(names) != len(set(names)):
            raise ValueError(f"All model names must be unique; got: {names}")
        self.models = models
        self.config = config or LeaderboardConfig()

    def run(self, save: bool = False) -> LeaderboardResult:
        """Evaluate all models on the same dataset and build the leaderboard.

        Steps
        -----
        1. Generate one adversarial dataset from ``config.seed``.
        2. Evaluate every model against that dataset (all see the same prompts).
        3. Compute α_bonferroni = α / (k*(k-1)/2).
        4. Run all-pairs comparisons with the corrected threshold.
        5. Rank by mean score; assign wins/losses/ties; mark significance.
        6. Optionally persist to disk.

        Returns
        -------
        :class:`LeaderboardResult`
        """
        cfg = self.config
        timestamp = ExperimentResult.make_timestamp()
        k = len(self.models)
        n_pairs = k * (k - 1) // 2
        alpha_bonf = _bonferroni_alpha(cfg.alpha, n_pairs)

        # --- 1. Shared dataset ---
        generator = AdversarialGenerator(seed=cfg.seed)
        dataset = AdversarialDataset()
        dataset.add_batch(
            generator.generate_all(
                jailbreak_count=cfg.jailbreak_count,
                injection_count=cfg.injection_count,
                boundary_count=cfg.boundary_count,
            )
        )

        # --- 2. Evaluate every model ---
        all_scores: Dict[str, ModelScores] = {}
        for model in self.models:
            all_scores[model.name] = _evaluate_model(model, dataset)

        # --- 3+4. All-pairs comparisons ---
        pairs: List[PairResult] = []
        model_list = self.models
        for i in range(k):
            for j in range(i + 1, k):
                name_i = model_list[i].name
                name_j = model_list[j].name
                pair = _compare_pair(all_scores[name_i], all_scores[name_j], alpha_bonf)
                pairs.append(pair)

        # --- 5. Rank ---
        entries = _rank_entries(all_scores, pairs)

        result = LeaderboardResult(
            name=cfg.name,
            timestamp=timestamp,
            config=asdict(cfg),
            entries=entries,
            pairs=pairs,
            alpha_bonferroni=alpha_bonf,
            n_models=k,
            n_pairs=n_pairs,
        )

        if save:
            result.save(cfg.output_dir)

        return result


# ---------------------------------------------------------------------------
# Built-in model specs for the CLI  (mirrors BASELINES in compare.py)
# ---------------------------------------------------------------------------

def _all_baseline_specs() -> List[ModelSpec]:
    """Return one ModelSpec per built-in BASELINES entry."""
    return [ModelSpec(name, fn) for name, fn in sorted(BASELINES.items())]
