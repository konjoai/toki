"""Red Team Campaign — orchestrates generate → mutate → judge → report.

Pipeline: AdversarialGenerator → PromptMutator → MockJudge → CampaignResult.
All components are seeded from CampaignConfig.seed.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CampaignConfig:
    """Configuration for a full Red Team Campaign run.

    Attributes
    ----------
    categories:
        Attack categories to generate prompts for.
    prompts_per_category:
        Number of seed prompts generated per category.
    population_size:
        GA population size for PromptMutator.
    n_generations:
        Number of GA generations.
    mutation_rate:
        Per-individual mutation probability.
    elite_fraction:
        Fraction of population that always survives selection.
    judge_name:
        Key identifying the judge implementation ("mock" or registered name).
    adversarial_threshold:
        overall_score ≥ threshold → adversarial_success=True.
    judge_criteria:
        Which criteria to evaluate; None means all JudgeCriteria members.
    output_dir:
        Directory for saved JSON and HTML reports.
    campaign_name:
        Identifier embedded in saved file names.
    seed:
        Master RNG seed; propagated to all sub-components.
    max_prompts_to_judge:
        Hard cap on prompts sent to the judge (speed control).
    """

    categories: List[str] = field(
        default_factory=lambda: ["jailbreak", "injection", "edge_case"]
    )
    prompts_per_category: int = 10
    population_size: int = 20
    n_generations: int = 5
    mutation_rate: float = 0.3
    elite_fraction: float = 0.2
    judge_name: str = "mock"
    adversarial_threshold: float = 0.6
    judge_criteria: Optional[List[str]] = None
    output_dir: str = "results/campaigns"
    campaign_name: str = "campaign"
    seed: int = 42
    max_prompts_to_judge: int = 50


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class CampaignResult:
    """Structured output from one Red Team Campaign run.

    Attributes
    ----------
    campaign_name:
        Matches CampaignConfig.campaign_name.
    config:
        The configuration used to produce this result.
    started_at:
        ISO-8601 UTC timestamp when the run began.
    finished_at:
        ISO-8601 UTC timestamp when the run completed.
    duration_seconds:
        Wall-clock duration of the run.
    n_generated:
        Total seed prompts generated across all categories.
    n_mutated:
        Total prompts in mutated populations across all categories.
    n_judged:
        Number of prompts actually scored by the judge.
    adversarial_success_rate:
        Fraction of judged prompts that exceeded the adversarial threshold.
    mean_overall_score:
        Mean overall_score across all judged prompts.
    per_criterion_scores:
        Per-criterion mean scores: {criterion_value: mean_score}.
    top_adversarial_prompts:
        Top 5 prompts by overall_score, each truncated to 200 chars.
    json_path:
        Absolute path to the saved JSON report (set by save()).
    html_path:
        Absolute path to the saved HTML report (set by save()).
    """

    campaign_name: str
    config: CampaignConfig
    started_at: str
    finished_at: str
    duration_seconds: float
    n_generated: int
    n_mutated: int
    n_judged: int
    adversarial_success_rate: float
    mean_overall_score: float
    per_criterion_scores: Dict[str, float]
    top_adversarial_prompts: List[str]
    json_path: Optional[str] = None
    html_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "campaign_name": self.campaign_name,
            "config": {
                "categories": self.config.categories,
                "prompts_per_category": self.config.prompts_per_category,
                "population_size": self.config.population_size,
                "n_generations": self.config.n_generations,
                "mutation_rate": self.config.mutation_rate,
                "elite_fraction": self.config.elite_fraction,
                "judge_name": self.config.judge_name,
                "adversarial_threshold": self.config.adversarial_threshold,
                "judge_criteria": self.config.judge_criteria,
                "output_dir": self.config.output_dir,
                "campaign_name": self.config.campaign_name,
                "seed": self.config.seed,
                "max_prompts_to_judge": self.config.max_prompts_to_judge,
            },
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "n_generated": self.n_generated,
            "n_mutated": self.n_mutated,
            "n_judged": self.n_judged,
            "adversarial_success_rate": self.adversarial_success_rate,
            "mean_overall_score": self.mean_overall_score,
            "per_criterion_scores": self.per_criterion_scores,
            "top_adversarial_prompts": self.top_adversarial_prompts,
            "json_path": self.json_path,
            "html_path": self.html_path,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_html(self) -> str:
        """Produce a self-contained dark-mode HTML report.

        Sections: summary stats, per-criterion bar chart (ASCII), top prompts table.
        No external CSS/JS — fully inline.
        """
        bar_section = _build_bar_section(self.per_criterion_scores)
        prompts_section = _build_prompts_section(self.top_adversarial_prompts)
        summary_section = _build_summary_section(self)
        return _html_shell(self.campaign_name, summary_section, bar_section, prompts_section)

    def save(self, output_dir: str) -> tuple:
        """Write JSON and HTML to output_dir. Returns (json_path, html_path)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{self.campaign_name}"
        json_path = str(out / f"{stem}.json")
        html_path = str(out / f"{stem}.html")
        Path(json_path).write_text(self.to_json(), encoding="utf-8")
        Path(html_path).write_text(self.to_html(), encoding="utf-8")
        self.json_path = json_path
        self.html_path = html_path
        return json_path, html_path


# ---------------------------------------------------------------------------
# HTML helpers — each ≤ 30 lines, no external deps
# ---------------------------------------------------------------------------


def _build_summary_section(result: CampaignResult) -> str:
    """Render the summary stats block as an HTML fragment."""
    rows = [
        ("Campaign", result.campaign_name),
        ("Started", result.started_at),
        ("Finished", result.finished_at),
        ("Duration (s)", f"{result.duration_seconds:.2f}"),
        ("Prompts generated", str(result.n_generated)),
        ("Prompts mutated", str(result.n_mutated)),
        ("Prompts judged", str(result.n_judged)),
        ("Adversarial success rate", f"{result.adversarial_success_rate:.2%}"),
        ("Mean overall score", f"{result.mean_overall_score:.4f}"),
        ("Seed", str(result.config.seed)),
    ]
    trs = "".join(
        f"<tr><td style='padding:4px 12px;color:#aaa'>{k}</td>"
        f"<td style='padding:4px 12px;color:#fff'>{v}</td></tr>"
        for k, v in rows
    )
    return f"<h2 style='color:#7dd3fc'>Summary</h2><table>{trs}</table>"


def _bar(score: float, width: int = 30) -> str:
    """Render an ASCII bar for a score in [0.0, 1.0]."""
    filled = max(0, min(width, round(score * width)))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {score:.3f}"


def _build_bar_section(per_criterion: Dict[str, float]) -> str:
    """Render per-criterion ASCII bar chart as an HTML fragment."""
    if not per_criterion:
        return "<h2 style='color:#7dd3fc'>Criteria Scores</h2><p style='color:#aaa'>No data.</p>"
    rows = "".join(
        f"<tr><td style='padding:4px 12px;color:#aaa;white-space:nowrap'>{crit[:60]}</td>"
        f"<td style='padding:4px 12px;color:#4ade80;font-family:monospace'>{_bar(score)}</td></tr>"
        for crit, score in per_criterion.items()
    )
    return f"<h2 style='color:#7dd3fc'>Criteria Scores</h2><table>{rows}</table>"


def _build_prompts_section(prompts: List[str]) -> str:
    """Render the top adversarial prompts table as an HTML fragment."""
    if not prompts:
        return "<h2 style='color:#7dd3fc'>Top Adversarial Prompts</h2><p style='color:#aaa'>None.</p>"
    rows = "".join(
        f"<tr><td style='padding:6px 12px;color:#fde68a;vertical-align:top'>{i + 1}</td>"
        f"<td style='padding:6px 12px;color:#e2e8f0;word-break:break-word'>"
        f"{_escape_html(p)}</td></tr>"
        for i, p in enumerate(prompts)
    )
    return (
        f"<h2 style='color:#7dd3fc'>Top Adversarial Prompts</h2>"
        f"<table style='width:100%;border-collapse:collapse'>{rows}</table>"
    )


def _escape_html(text: str) -> str:
    """Minimal HTML entity escaping — no stdlib html module dependency."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _html_shell(title: str, summary: str, bars: str, prompts: str) -> str:
    """Assemble the final self-contained HTML document."""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        f"<title>Toki Campaign: {_escape_html(title)}</title>"
        "<style>body{background:#0f172a;font-family:sans-serif;margin:40px}"
        "h1{color:#f8fafc}table{border-collapse:collapse}"
        "tr:nth-child(even){background:#1e293b}"
        "</style></head><body>"
        f"<h1>Toki Red Team Campaign — {_escape_html(title)}</h1>"
        f"{summary}{bars}{prompts}"
        "<footer style='color:#475569;margin-top:40px;font-size:0.85em'>"
        "Generated by toki campaign</footer></body></html>"
    )


# ---------------------------------------------------------------------------
# Campaign orchestrator
# ---------------------------------------------------------------------------


class RedTeamCampaign:
    """Runs the full toki pipeline in one call.

    Pipeline: AdversarialGenerator → PromptMutator → MockJudge → CampaignResult.
    All components are seeded from campaign_config.seed.
    """

    def __init__(self, config: Optional[CampaignConfig] = None) -> None:
        """Initialise with optional config; defaults to CampaignConfig()."""
        self._config = config or CampaignConfig()

    @property
    def config(self) -> CampaignConfig:
        """Expose the campaign configuration."""
        return self._config

    def run(self) -> CampaignResult:
        """Execute the full pipeline. Returns CampaignResult.

        Steps:
        1. Generate seed prompts via AdversarialGenerator for each category.
        2. Mutate each category with PromptMutator (fitness_fn=_judge_score).
        3. Collect top mutated prompts (population_size best per category).
        4. Score via judge up to max_prompts_to_judge.
        5. Compute aggregate stats via JudgePipeline.summary().
        6. Return CampaignResult. Never raises — exceptions are logged.
        """
        cfg = self._config
        started_at = _utc_now()
        t0 = time.monotonic()
        logger.info(
            "Campaign '%s' starting (seed=%d, categories=%s)",
            cfg.campaign_name,
            cfg.seed,
            cfg.categories,
        )
        all_generated: List = []
        all_mutated_prompts: List[str] = []

        try:
            all_generated, all_mutated_prompts = self._generate_and_mutate(cfg)
        except Exception as exc:
            logger.warning("generate/mutate phase failed: %s", exc)

        verdicts = []
        try:
            verdicts = self._score_prompts(all_mutated_prompts, cfg)
        except Exception as exc:
            logger.warning("judging phase failed: %s", exc)

        stats = _empty_stats()
        try:
            from toki.judge import JudgePipeline, MockJudge, JudgeConfig

            judge_cfg = JudgeConfig(
                criteria=_resolve_criteria(cfg.judge_criteria),
                adversarial_threshold=cfg.adversarial_threshold,
                judge_name=cfg.judge_name,
            )
            pipeline = JudgePipeline(judge=MockJudge(judge_cfg))
            stats = pipeline.summary(verdicts)
        except Exception as exc:
            logger.warning("stats aggregation failed: %s", exc)

        top = self._top_prompts(verdicts, n=5)
        finished_at = _utc_now()
        duration = time.monotonic() - t0

        return CampaignResult(
            campaign_name=cfg.campaign_name,
            config=cfg,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            n_generated=len(all_generated),
            n_mutated=len(all_mutated_prompts),
            n_judged=len(verdicts),
            adversarial_success_rate=stats["adversarial_success_rate"],
            mean_overall_score=stats["mean_overall_score"],
            per_criterion_scores=stats["per_criterion_scores"],
            top_adversarial_prompts=top,
        )

    def _generate_and_mutate(self, cfg: CampaignConfig):
        """Generate seed prompts and run GA mutation for each category."""
        from toki.generate import AdversarialGenerator
        from toki.mutator import PromptMutator, MutationConfig

        gen = AdversarialGenerator(seed=cfg.seed)
        mut_cfg = MutationConfig(
            population_size=cfg.population_size,
            n_generations=cfg.n_generations,
            mutation_rate=cfg.mutation_rate,
            elite_fraction=cfg.elite_fraction,
            seed=cfg.seed,
        )

        all_generated = []
        all_mutated: List[str] = []

        for category in cfg.categories:
            seeds = _get_seeds(gen, category, cfg.prompts_per_category)
            all_generated.extend(seeds)
            seed_texts = [p.text for p in seeds] or ["test prompt"]
            mutator = PromptMutator(config=mut_cfg)
            try:
                result = mutator.evolve(
                    initial_prompts=seed_texts,
                    category=category,
                    fitness_fn=self._judge_score,
                )
                top_inds = sorted(
                    result.final_population,
                    key=lambda ind: ind.fitness,
                    reverse=True,
                )[: cfg.population_size]
                all_mutated.extend(ind.prompt for ind in top_inds)
            except Exception as exc:
                logger.warning("mutation failed for category '%s': %s", category, exc)
                all_mutated.extend(seed_texts)

        return all_generated, all_mutated

    def _score_prompts(self, prompts: List[str], cfg: CampaignConfig):
        """Score up to max_prompts_to_judge prompts with the judge."""
        from toki.judge import MockJudge, JudgeConfig
        from toki.generate import AdversarialPrompt
        from toki.dataset import AdversarialDataset

        judge_cfg = JudgeConfig(
            criteria=_resolve_criteria(cfg.judge_criteria),
            adversarial_threshold=cfg.adversarial_threshold,
            judge_name=cfg.judge_name,
        )
        judge = MockJudge(judge_cfg)
        capped = prompts[: cfg.max_prompts_to_judge]
        verdicts = []
        for prompt in capped:
            response = f"[mock response to: {prompt}]"
            verdict = judge.judge(prompt, response)
            verdicts.append(verdict)
        return verdicts

    def _judge_score(self, prompt: str) -> float:
        """Single-prompt fitness fn for PromptMutator: judge(prompt, response).overall_score."""
        from toki.judge import MockJudge, JudgeConfig

        cfg = self._config
        judge_cfg = JudgeConfig(
            criteria=_resolve_criteria(cfg.judge_criteria),
            adversarial_threshold=cfg.adversarial_threshold,
            judge_name=cfg.judge_name,
        )
        judge = MockJudge(judge_cfg)
        response = f"[mock response to: {prompt}]"
        verdict = judge.judge(prompt, response)
        return verdict.overall_score

    def _top_prompts(self, verdicts: list, n: int = 5) -> List[str]:
        """Top n prompts by overall_score, each truncated to 200 chars."""
        if not verdicts:
            return []
        sorted_v = sorted(verdicts, key=lambda v: v.overall_score, reverse=True)
        return [v.prompt[:200] for v in sorted_v[:n]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _empty_stats() -> dict:
    """Return zero-value stats dict matching JudgePipeline.summary() shape."""
    return {
        "mean_overall_score": 0.0,
        "adversarial_success_rate": 0.0,
        "total_evaluated": 0,
        "per_criterion_scores": {},
    }


def _resolve_criteria(judge_criteria: Optional[List[str]]):
    """Resolve criteria list to JudgeCriteria members.

    None → all criteria. Strings are matched by value against JudgeCriteria.
    """
    from toki.judge import JudgeCriteria

    all_criteria = list(JudgeCriteria)
    if judge_criteria is None:
        return all_criteria
    resolved = []
    for val in judge_criteria:
        matched = next((c for c in all_criteria if c.value == val or c.name == val), None)
        if matched is not None:
            resolved.append(matched)
        else:
            logger.warning("Unknown judge criterion %r — skipping", val)
    return resolved or all_criteria


def _get_seeds(gen, category: str, count: int):
    """Generate seed prompts for the given category."""
    if category == "jailbreak":
        return gen.generate_jailbreaks(count)
    if category == "injection":
        return gen.generate_injections(count)
    if category == "edge_case":
        return gen.generate_edge_cases()
    if category == "boundary":
        return gen.generate_boundary_cases(count)
    logger.warning("Unknown category %r — using jailbreak fallback", category)
    return gen.generate_jailbreaks(count)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_campaign(config: Optional[CampaignConfig] = None) -> CampaignResult:
    """Run a full Red Team Campaign with the given config (or defaults).

    Parameters
    ----------
    config:
        Campaign configuration. Uses CampaignConfig() defaults when None.

    Returns
    -------
    CampaignResult
        Structured result containing stats, paths, and top prompts.
    """
    return RedTeamCampaign(config).run()
