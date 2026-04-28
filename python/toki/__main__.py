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


def cmd_list(args) -> None:
    from toki.results import list_experiments
    paths = list_experiments(args.dir)
    if not paths:
        print(f"No experiments found in {args.dir}")
        return
    for p in paths:
        data = json.loads(p.read_text())
        print(f"{data['timestamp']}  {data['name']}  pre={data['pre_score']:.3f}")


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

    return ap


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


if __name__ == "__main__":
    main()
