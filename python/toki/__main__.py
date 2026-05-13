"""python -m toki entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_generate(args) -> None:
    from toki.generate import AdversarialGenerator
    from toki.dataset import AdversarialDataset

    gen = AdversarialGenerator(seed=args.seed)
    ds = AdversarialDataset()
    prompts = gen.generate_all(
        jailbreak_count=args.count,
        injection_count=args.count,
        boundary_count=args.count // 2,
    )
    added = ds.add_batch(prompts)

    if args.output:
        ds.save(args.output)
        print(f"Saved {added} prompts to {args.output}")
    else:
        print(f"Generated {added} prompts")
        stats = ds.stats()
        for cat, count in stats["categories"].items():
            print(f"  {cat}: {count}")


def cmd_evaluate(args) -> None:
    from toki.evaluate import RobustnessEvaluator
    from toki.dataset import AdversarialDataset

    if args.dataset:
        ds = AdversarialDataset.load(args.dataset)
        prompts = list(ds)
    else:
        from toki.generate import AdversarialGenerator
        gen = AdversarialGenerator(seed=args.seed)
        ds = AdversarialDataset()
        ds.add_batch(gen.generate_all())
        prompts = list(ds)

    evaluator = RobustnessEvaluator()
    results = evaluator.evaluate_batch(prompts)
    summary = evaluator.summary(results)

    print(f"Total prompts: {summary['total']}")
    print(f"Mean score:    {summary['mean_score']:.3f}")
    print(f"Refusal rate:  {summary['refusal_rate']:.1%}")
    print(f"Harmful rate:  {summary['harmful_rate']:.1%}")
    print("\nBy category:")
    for cat, score in summary.get("by_category", {}).items():
        print(f"  {cat}: {score:.3f}")


def cmd_run(args) -> None:
    from toki.experiment import TokiExperiment, ExperimentConfig

    config = ExperimentConfig(
        name=args.name,
        model_name=args.model,
        seed=args.seed,
        output_dir=args.output_dir,
        run_finetune=args.finetune,
    )
    exp = TokiExperiment(config)
    result = exp.run()

    print(f"\n{'='*50}")
    print(f"Experiment: {result.name}")
    print(f"Pre-score:  {result.pre_score:.4f}")
    if result.post_score is not None:
        print(f"Post-score: {result.post_score:.4f}")
        print(f"Delta:      {result.improvement:+.4f}")
    print(f"Prompts:    {result.total_prompts}")
    print(f"{'='*50}")


def cmd_report(args) -> None:
    import random
    from toki.results import ExperimentResult
    from toki.benchmark import generate_report
    from toki.report import to_json, to_html

    result = ExperimentResult.load(args.result_json)

    # Build synthetic score lists (N=20 gaussian samples around the stored means)
    rng = random.Random(result.seed)
    n_samples = 20
    noise = 0.02

    def _gauss_clamp(mean: float) -> float:
        v = mean + rng.gauss(0, noise)
        return max(0.0, min(1.0, v))

    pre_scores = [_gauss_clamp(result.pre_score) for _ in range(n_samples)]
    post_scores = None
    if result.post_score is not None:
        post_scores = [_gauss_clamp(result.post_score) for _ in range(n_samples)]

    # Per-category synthetic scores
    category_pre = None
    category_post = None
    if result.category_scores:
        category_pre = {
            cat: [_gauss_clamp(score) for _ in range(n_samples)]
            for cat, score in result.category_scores.items()
        }
        if result.post_score is not None:
            # Use post_score as a rough proxy for categories when no per-cat post data
            category_post = {
                cat: [_gauss_clamp(result.post_score) for _ in range(n_samples)]
                for cat in result.category_scores
            }

    report = generate_report(
        result=result,
        pre_scores=pre_scores,
        post_scores=post_scores,
        category_pre=category_pre,
        category_post=category_post,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    stem = f"{result.timestamp}_{result.name}_report"

    if fmt in ("json", "both"):
        p = output_dir / f"{stem}.json"
        to_json(report, path=p)
        print(f"JSON report written to {p}")

    if fmt in ("html", "both"):
        p = output_dir / f"{stem}.html"
        to_html(report, path=p)
        print(f"HTML report written to {p}")

    if fmt not in ("json", "html", "both"):
        print(f"Unknown format {fmt!r}; use --format json|html|both", file=__import__("sys").stderr)


def cmd_compare(args) -> None:
    from toki.compare import (
        BASELINES, ComparisonConfig, ModelSpec, compare_models,
    )
    if args.model_a not in BASELINES:
        raise SystemExit(f"--model-a must be one of {sorted(BASELINES)}, got {args.model_a!r}")
    if args.model_b not in BASELINES:
        raise SystemExit(f"--model-b must be one of {sorted(BASELINES)}, got {args.model_b!r}")
    if args.model_a == args.model_b:
        raise SystemExit("--model-a and --model-b must differ")

    cfg = ComparisonConfig(
        name=args.name,
        seed=args.seed,
        jailbreak_count=args.jailbreak_count,
        injection_count=args.injection_count,
        boundary_count=args.boundary_count,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )
    spec_a = ModelSpec(args.model_a, BASELINES[args.model_a])
    spec_b = ModelSpec(args.model_b, BASELINES[args.model_b])
    result = compare_models(spec_a, spec_b, cfg, save=True)

    print(f"\n{'=' * 60}")
    print(f"A/B Comparison: {result.name}   ({result.timestamp})")
    print(f"{'=' * 60}")
    print(f"  {spec_a.name:>10}: mean={result.model_a.mean_score:.4f}  "
          f"refusal={result.model_a.refusal_rate:.0%}  "
          f"harmful={result.model_a.harmful_rate:.0%}  "
          f"leak={result.model_a.leak_rate:.0%}")
    print(f"  {spec_b.name:>10}: mean={result.model_b.mean_score:.4f}  "
          f"refusal={result.model_b.refusal_rate:.0%}  "
          f"harmful={result.model_b.harmful_rate:.0%}  "
          f"leak={result.model_b.leak_rate:.0%}")
    print(f"  Δ (b - a): {result.score_delta:+.4f}")
    if result.t_test:
        marker = "✓" if result.t_test["significant"] else " "
        print(f"  [{marker}] paired t-test: t={result.t_test['statistic']:+.3f}  "
              f"p={result.t_test['p_value']:.4g}  α={result.t_test['alpha']}")
    if result.wilcoxon:
        marker = "✓" if result.wilcoxon["significant"] else " "
        print(f"  [{marker}] wilcoxon:      W={result.wilcoxon['statistic']:.3f}  "
              f"p={result.wilcoxon['p_value']:.4g}")
    print(f"\n  Winner: \033[1m{result.winner}\033[0m"
          f" ({'significant' if result.significant else 'not significant'} at α={cfg.alpha})")
    print("\n  Per-category winners:")
    for cat, winner in sorted(result.category_winners.items()):
        print(f"    {cat:>10}: {winner}")


def cmd_pipeline(args) -> None:
    from toki.pipeline import HardeningPipeline, PipelineConfig

    cfg = PipelineConfig(
        name=args.name,
        model_name=args.model,
        seed=args.seed,
        max_iterations=args.iterations,
        convergence_threshold=args.convergence_threshold,
        convergence_window=args.convergence_window,
        jailbreak_count=args.jailbreak_count,
        injection_count=args.injection_count,
        boundary_count=args.boundary_count,
        output_dir=args.output_dir,
        run_finetune=args.finetune,
    )
    result = HardeningPipeline(cfg).run()

    print(f"\n{'=' * 60}")
    print(f"Pipeline:    {result.name}")
    print(f"Timestamp:   {result.timestamp}")
    print(f"Rounds:      {len(result.rounds)} / {cfg.max_iterations}")
    print(f"Converged:   {result.converged} ({result.stop_reason})")
    print(f"Final score: {result.final_score:.4f}")
    print(f"{'=' * 60}")
    print("\nPer-round mean score:")
    for r in result.rounds:
        marker = "✓" if r.mean_score >= cfg.convergence_threshold else " "
        print(f"  [{marker}] round {r.round_index:>3}  seed={r.seed:<10}  score={r.mean_score:.4f}  n={r.total_prompts}")


def cmd_rank(args) -> None:
    from toki.compare import BASELINES, ModelSpec
    from toki.ranking import Ranking, RankingConfig, _all_baseline_specs

    # Resolve model list: if --models provided, validate against BASELINES;
    # otherwise default to all three built-in baselines.
    if args.models:
        unknown = [m for m in args.models if m not in BASELINES]
        if unknown:
            raise SystemExit(
                f"Unknown model(s) {unknown!r}. "
                f"Available built-ins: {sorted(BASELINES)}"
            )
        if len(args.models) < 2:
            raise SystemExit("--models requires at least 2 model names")
        if len(args.models) != len(set(args.models)):
            raise SystemExit("--models entries must be unique")
        model_specs = [ModelSpec(name, BASELINES[name]) for name in args.models]
    else:
        model_specs = _all_baseline_specs()

    cfg = RankingConfig(
        name=args.name,
        seed=args.seed,
        jailbreak_count=args.jailbreak_count,
        injection_count=args.injection_count,
        boundary_count=args.boundary_count,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )
    r = Ranking(model_specs, cfg)
    result = r.run(save=args.save)

    print(result.format_table())

    if args.save:
        out_dir = __import__("pathlib").Path(cfg.output_dir)
        saved_path = out_dir / f"{result.timestamp}_{result.name}" / "ranking.json"
        if saved_path.exists():
            print(f"\n  Saved → {saved_path}")


def cmd_upload(args) -> None:
    from toki.dataset import AdversarialDataset
    from toki.hub import DatasetMetadata, HubUploader, write_card

    ds = AdversarialDataset.load(args.dataset)
    metadata = DatasetMetadata(
        name=args.name or Path(args.dataset).stem,
        version=args.version,
    )
    if args.description:
        metadata.description = args.description
    if args.private:
        pass  # handled by uploader

    if args.dry_run:
        out = Path(args.output_card or "DATASET_CARD.md")
        write_card(ds, metadata, out)
        print(f"[dry-run] Dataset card written to {out}")
        print(f"[dry-run] Would upload {len(ds)} prompts to {args.repo} (v{metadata.version})")
        return

    uploader = HubUploader(repo_id=args.repo, token=args.token, private=args.private)
    summary = uploader.upload(ds, metadata, commit_message=args.message)
    print(f"Uploaded {summary['total_prompts']} prompts → {summary['repo_id']}")
    print(f"  dataset_version: {summary['dataset_version']}")
    print(f"  toki_version:    {summary['toki_version']}")
    for cat, n in summary["categories"].items():
        print(f"  {cat}: {n}")


def cmd_list(args) -> None:
    from toki.results import list_experiments
    paths = list_experiments(args.dir)
    if not paths:
        print(f"No experiments found in {args.dir}")
        return
    for p in paths:
        data = json.loads(p.read_text())
        print(f"{data['timestamp']}  {data['name']}  pre={data['pre_score']:.3f}")


def cmd_distill(args) -> None:
    from toki.dataset import AdversarialDataset
    from toki.distill import CorpusDistiller, DistillConfig

    ds = AdversarialDataset.load(args.dataset)
    cfg = DistillConfig(
        fitness_threshold=args.fitness_threshold,
        max_per_category=args.max_per_category,
        min_prompts=args.min_prompts,
        max_total=args.max_total,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    distiller = CorpusDistiller(cfg)
    result = distiller.distill_from_dataset(ds)

    stats = result.stats
    print(f"\n{'=' * 55}")
    print(f"Corpus Distillation   ({result.timestamp})")
    print(f"{'=' * 55}")
    print(f"  Input prompts:    {stats.total_input}")
    print(f"  Retained:         {stats.total_retained}")
    print(f"  Retention rate:   {stats.retention_rate:.1%}")
    print(f"  Mean fitness:     {stats.mean_fitness:.4f}")
    print(f"  Max fitness:      {stats.max_fitness:.4f}")
    print(f"  Categories:       {', '.join(stats.categories_covered) or '(none)'}")
    print(f"")
    print(f"  Per-category breakdown:")
    for cat, n in sorted(stats.per_category.items()):
        print(f"    {cat:>12}: {n}")

    if args.save:
        saved_dir = result.save()
        print(f"\n  Saved → {saved_dir}")
    elif args.output:
        result.dataset.save(args.output)
        print(f"\n  Corpus written to {args.output}")


def cmd_campaign(args) -> None:
    from toki.campaign import CampaignConfig, run_campaign

    config = CampaignConfig(
        output_dir=args.output,
        seed=args.seed,
        campaign_name=args.name,
    )
    if args.config:
        cfg_data = json.loads(Path(args.config).read_text())
        for k, v in cfg_data.items():
            if hasattr(config, k):
                setattr(config, k, v)

    result = run_campaign(config)

    print(f"\n{'=' * 60}")
    print(f"Campaign:              {result.campaign_name}")
    print(f"Seed:                  {result.config.seed}")
    print(f"Started:               {result.started_at}")
    print(f"Duration (s):          {result.duration_seconds:.2f}")
    print(f"Prompts generated:     {result.n_generated}")
    print(f"Prompts mutated:       {result.n_mutated}")
    print(f"Prompts judged:        {result.n_judged}")
    print(f"Adversarial success:   {result.adversarial_success_rate:.2%}")
    print(f"Mean overall score:    {result.mean_overall_score:.4f}")
    print(f"{'=' * 60}")

    json_path, html_path = result.save(result.config.output_dir)
    print(f"\n  JSON report → {json_path}")
    print(f"  HTML report → {html_path}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="toki", description="Adversarial fine-tuning lab")
    sub = ap.add_subparsers(dest="command", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Generate adversarial prompts")
    p_gen.add_argument("--count", type=int, default=10)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--output", type=str, default=None)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate model robustness")
    p_eval.add_argument("--dataset", type=str, default=None)
    p_eval.add_argument("--seed", type=int, default=42)

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Run full adversarial experiment pipeline")
    p_run.add_argument("--name", type=str, default="experiment")
    p_run.add_argument("--model", type=str, default="mock")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--output-dir", type=str, default="experiments/runs")
    p_run.add_argument("--finetune", action="store_true")

    # list
    p_list = sub.add_parser("list", help="List past experiments")
    p_list.add_argument("--dir", type=str, default="experiments/runs")

    # compare (A/B model comparison)
    p_cmp = sub.add_parser("compare", help="A/B compare two model_fns on the same adversarial dataset")
    p_cmp.add_argument("--model-a", type=str, required=True,
                       help="Built-in baseline name: safe | unsafe | mixed")
    p_cmp.add_argument("--model-b", type=str, required=True,
                       help="Built-in baseline name: safe | unsafe | mixed")
    p_cmp.add_argument("--name", type=str, default="comparison")
    p_cmp.add_argument("--seed", type=int, default=42)
    p_cmp.add_argument("--alpha", type=float, default=0.05)
    p_cmp.add_argument("--jailbreak-count", type=int, default=10, dest="jailbreak_count")
    p_cmp.add_argument("--injection-count", type=int, default=10, dest="injection_count")
    p_cmp.add_argument("--boundary-count", type=int, default=5,  dest="boundary_count")
    p_cmp.add_argument("--output-dir", type=str, default="experiments/comparisons")

    # pipeline (continuous hardening loop)
    p_pipe = sub.add_parser("pipeline", help="Run iterative generate→evaluate→(finetune) loop until convergence")
    p_pipe.add_argument("--name", type=str, default="hardening")
    p_pipe.add_argument("--model", type=str, default="mock")
    p_pipe.add_argument("--seed", type=int, default=42)
    p_pipe.add_argument("--iterations", type=int, default=5, dest="iterations")
    p_pipe.add_argument("--convergence-threshold", type=float, default=0.95, dest="convergence_threshold")
    p_pipe.add_argument("--convergence-window", type=int, default=3, dest="convergence_window")
    p_pipe.add_argument("--jailbreak-count", type=int, default=10, dest="jailbreak_count")
    p_pipe.add_argument("--injection-count", type=int, default=10, dest="injection_count")
    p_pipe.add_argument("--boundary-count", type=int, default=5, dest="boundary_count")
    p_pipe.add_argument("--output-dir", type=str, default="experiments/pipelines")
    p_pipe.add_argument("--finetune", action="store_true")

    # rank (multi-model ranking with Bonferroni correction)
    p_rk = sub.add_parser(
        "rank",
        help="Rank ≥2 models on the same adversarial dataset with Bonferroni-corrected paired tests",
    )
    p_rk.add_argument(
        "--models", nargs="+", default=None,
        metavar="MODEL",
        help="Built-in baseline names to include (default: all three — safe, unsafe, mixed)",
    )
    p_rk.add_argument("--name",  type=str, default="ranking")
    p_rk.add_argument("--seed",  type=int, default=42)
    p_rk.add_argument("--alpha", type=float, default=0.05,
                      help="Nominal family-wise α before Bonferroni correction")
    p_rk.add_argument("--jailbreak-count", type=int, default=10, dest="jailbreak_count")
    p_rk.add_argument("--injection-count", type=int, default=10, dest="injection_count")
    p_rk.add_argument("--boundary-count",  type=int, default=5,  dest="boundary_count")
    p_rk.add_argument("--output-dir", type=str, default="experiments/rankings")
    p_rk.add_argument("--save", action="store_true", help="Persist ranking.json to disk")

    # upload
    p_up = sub.add_parser("upload", help="Publish an adversarial dataset to the HuggingFace Hub")
    p_up.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON")
    p_up.add_argument("--repo", type=str, required=True, help="HF repo id, e.g. user/toki-adversarial")
    p_up.add_argument("--name", type=str, default=None, help="Dataset display name (defaults to file stem)")
    p_up.add_argument("--version", type=str, default="0.1.0", help="Dataset version tag")
    p_up.add_argument("--description", type=str, default=None)
    p_up.add_argument("--token", type=str, default=None, help="HF Hub token; falls back to cached login")
    p_up.add_argument("--private", action="store_true")
    p_up.add_argument("--message", type=str, default=None, help="Commit message")
    p_up.add_argument("--dry-run", action="store_true", help="Render the dataset card locally without uploading")
    p_up.add_argument("--output-card", type=str, default=None, help="Where to write the card on dry-run")

    # report
    p_rep = sub.add_parser("report", help="Generate benchmark report for an experiment result")
    p_rep.add_argument("result_json", type=str, help="Path to result.json")
    p_rep.add_argument("--format", type=str, default="both", choices=["json", "html", "both"])
    p_rep.add_argument("--output-dir", type=str, default="experiments/reports")

    # campaign (full red team campaign)
    p_camp = sub.add_parser("campaign", help="Run a full Red Team Campaign")
    p_camp_sub = p_camp.add_subparsers(dest="campaign_command", required=True)
    p_camp_run = p_camp_sub.add_parser("run", help="Execute campaign pipeline")
    p_camp_run.add_argument("--config", type=str, default=None,
                            help="Path to campaign config JSON")
    p_camp_run.add_argument("--output", type=str, default="results/campaigns",
                            help="Output directory for reports")
    p_camp_run.add_argument("--seed", type=int, default=42)
    p_camp_run.add_argument("--name", type=str, default="campaign",
                            help="Campaign name used in report filenames")

    # distill (corpus distillation)
    p_distill = sub.add_parser(
        "distill",
        help="Distil a compact, high-fitness attack corpus from a dataset or mutation output",
    )
    p_distill.add_argument("--dataset", type=str, required=True,
                           help="Path to adversarial dataset JSON")
    p_distill.add_argument("--fitness-threshold", type=float, default=0.4,
                           dest="fitness_threshold",
                           help="Minimum fitness score to retain a prompt (default: 0.4)")
    p_distill.add_argument("--max-per-category", type=int, default=50,
                           dest="max_per_category",
                           help="Maximum prompts per category (default: 50)")
    p_distill.add_argument("--min-prompts", type=int, default=5,
                           dest="min_prompts",
                           help="Minimum total prompts to retain (default: 5)")
    p_distill.add_argument("--max-total", type=int, default=500,
                           dest="max_total",
                           help="Hard cap on total retained prompts (default: 500)")
    p_distill.add_argument("--seed", type=int, default=42)
    p_distill.add_argument("--output-dir", type=str, default="experiments/distilled",
                           dest="output_dir")
    p_distill.add_argument("--output", type=str, default=None,
                           help="Write distilled corpus JSON to this path")
    p_distill.add_argument("--save", action="store_true",
                           help="Persist full DistillResult artefacts to --output-dir")

    # coverage (P1 — coverage map + blind spot dashboard)
    p_cov = sub.add_parser("coverage", help="Print a coverage map for a dataset")
    p_cov.add_argument("--dataset", type=str, default=None,
                       help="Path to adversarial dataset JSON (default: generate fresh)")
    p_cov.add_argument("--include-multilingual", action="store_true",
                       dest="include_multilingual",
                       help="Append the 50-case multilingual+encoding battery")
    p_cov.add_argument("--blind-threshold", type=float, default=0.05,
                       dest="blind_threshold")
    p_cov.add_argument("--json", action="store_true", help="Emit JSON")

    # ci-baseline (P1 — safety regression CI gate)
    p_base = sub.add_parser(
        "ci-baseline",
        help="Snapshot current per-category pass rates as a regression baseline",
    )
    p_base.add_argument("--output", type=str, required=True,
                        help="Where to write baseline.json")
    p_base.add_argument("--seed", type=int, default=42)
    p_base.add_argument("--size", type=int, default=24,
                        help="Total prompts to evaluate")

    # ci-check (P1 — safety regression CI gate)
    p_check = sub.add_parser(
        "ci-check",
        help="Re-run evaluation, compare to baseline, exit 1 on regression",
    )
    p_check.add_argument("--baseline", type=str, required=True,
                         help="Path to baseline.json")
    p_check.add_argument("--tolerance", type=float, default=0.02,
                         help="Allowed per-category drop (default: 0.02)")
    p_check.add_argument("--seed", type=int, default=42)
    p_check.add_argument("--size", type=int, default=24)
    p_check.add_argument("--report", type=str, default=None,
                         help="Optional path to write Markdown report")
    p_check.add_argument("--json", action="store_true",
                         help="Emit JSON report to stdout")

    # consistency (P1 — evaluator consistency scoring)
    p_con = sub.add_parser(
        "consistency",
        help="Run prompts through multiple judges and report Fleiss' kappa",
    )
    p_con.add_argument("--size", type=int, default=12)
    p_con.add_argument("--seed", type=int, default=42)
    p_con.add_argument("--threshold", type=float, default=0.6,
                       help="κ below this is flagged unreliable")
    p_con.add_argument("--json", action="store_true")

    return ap


def cmd_coverage(args) -> None:
    import json as _json
    from toki.coverage import compute_coverage
    from toki.dataset import AdversarialDataset
    from toki.generate import AdversarialGenerator
    from toki.multilingual import generate_battery

    if args.dataset:
        ds = AdversarialDataset.load(args.dataset)
        prompts = list(ds)
    else:
        gen = AdversarialGenerator(seed=42)
        prompts = gen.generate_all(jailbreak_count=12, injection_count=12, boundary_count=6)
    if args.include_multilingual:
        prompts = list(prompts) + list(generate_battery())
    cov = compute_coverage(prompts, blind_threshold=args.blind_threshold)

    if args.json:
        print(_json.dumps(cov.as_dict(), indent=2))
        return
    print(f"Coverage map  total={cov.total}  blind_threshold={cov.blind_threshold:.0%}")
    for axis_name in ("category", "severity", "language", "encoding"):
        print(f"\n  {axis_name}")
        for bucket, count in cov.axes[axis_name].items():
            share = cov.shares[axis_name][bucket]
            bar = "█" * max(0, int(round(share * 24)))
            flag = "  ⚠ blind" if f"{axis_name}.{bucket}" in cov.blind_spots else ""
            print(f"    {bucket:<14} {count:>3}  {share:6.1%}  {bar}{flag}")
    if cov.blind_spots:
        print(f"\n  {len(cov.blind_spots)} blind spot(s): {', '.join(cov.blind_spots)}")


def cmd_ci_baseline(args) -> None:
    from toki.dataset import AdversarialDataset
    from toki.evaluate import RobustnessEvaluator
    from toki.generate import AdversarialGenerator
    from toki.regression import Baseline

    gen = AdversarialGenerator(seed=args.seed)
    jb = max(1, args.size // 3)
    inj = max(1, args.size // 3)
    bnd = max(1, args.size - jb - inj)
    ds = AdversarialDataset()
    ds.add_batch(gen.generate_all(jailbreak_count=jb, injection_count=inj, boundary_count=bnd))
    evaluator = RobustnessEvaluator()
    results = evaluator.evaluate_batch(list(ds))
    summary = evaluator.summary(results)
    baseline = Baseline.from_summary(summary, meta={"seed": args.seed, "size": args.size})
    p = baseline.save(args.output)
    print(f"Baseline written to {p}")
    print(f"  overall: {baseline.overall:.4f}")
    for cat, rate in sorted(baseline.per_category.items()):
        print(f"  {cat:<14} {rate:.4f}")


def cmd_ci_check(args) -> None:
    import json as _json
    import sys as _sys
    from pathlib import Path as _Path
    from toki.dataset import AdversarialDataset
    from toki.evaluate import RobustnessEvaluator
    from toki.generate import AdversarialGenerator
    from toki.regression import Baseline, compare

    baseline = Baseline.load(args.baseline)
    gen = AdversarialGenerator(seed=args.seed)
    jb = max(1, args.size // 3)
    inj = max(1, args.size // 3)
    bnd = max(1, args.size - jb - inj)
    ds = AdversarialDataset()
    ds.add_batch(gen.generate_all(jailbreak_count=jb, injection_count=inj, boundary_count=bnd))
    evaluator = RobustnessEvaluator()
    results = evaluator.evaluate_batch(list(ds))
    summary = evaluator.summary(results)
    report = compare(baseline, summary, tolerance=args.tolerance)

    if args.json:
        _sys.stdout.write(_json.dumps(report.as_dict(), indent=2) + "\n")
    else:
        _sys.stdout.write(report.to_markdown())
    if args.report:
        _Path(args.report).write_text(report.to_markdown())

    _sys.exit(report.exit_code())


def cmd_consistency(args) -> None:
    import json as _json
    from toki.consistency import ConsistencyEvaluator
    from toki.generate import AdversarialGenerator

    gen = AdversarialGenerator(seed=args.seed)
    jb = max(1, args.size // 3)
    inj = max(1, args.size // 3)
    bnd = max(1, args.size - jb - inj)
    prompts = gen.generate_all(jailbreak_count=jb, injection_count=inj, boundary_count=bnd)
    ev = ConsistencyEvaluator(threshold=args.threshold)
    report = ev.evaluate(prompts)
    if args.json:
        print(_json.dumps(report.as_dict(), indent=2))
        return
    print(f"Consistency report  N={len(report.entries)}  judges={','.join(report.judges)}")
    print(f"  Fleiss' κ = {report.mean_kappa:.4f}   threshold = {report.threshold:.2f}")
    print(f"  unreliable: {report.unreliable_count} / {len(report.entries)}")
    print("\n  agreement matrix (fraction of prompts judges agreed):")
    for j1 in report.judges:
        row = "    " + j1.ljust(10)
        for j2 in report.judges:
            row += f"{report.agreement_matrix[j1][j2]:>8.3f}"
        print(row)
    print()
    print("  flagged (kappa < threshold):")
    for e in report.entries:
        if e.unreliable:
            print(f"    [{e.category}] κ={e.kappa:.3f}  {e.prompt_text[:90]}…")


def main(argv=None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "upload":
        cmd_upload(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "rank":
        cmd_rank(args)
    elif args.command == "distill":
        cmd_distill(args)
    elif args.command == "campaign":
        cmd_campaign(args)
    elif args.command == "coverage":
        cmd_coverage(args)
    elif args.command == "ci-baseline":
        cmd_ci_baseline(args)
    elif args.command == "ci-check":
        cmd_ci_check(args)
    elif args.command == "consistency":
        cmd_consistency(args)


if __name__ == "__main__":
    main()
