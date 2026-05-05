# toki

Adversarial fine-tuning lab for small LLMs (1B–3B) — generate adversarial datasets (jailbreak, injection, edge-case, boundary), evaluate robustness, LoRA fine-tune, and measure what actually improves. Break the model. Fix the model. Prove it.

**v0.5.0** (Python) / **v0.1.0** (Rust CLI) — Rust CLI delegates to Python training layer.

## Stack
Rust 2021 · clap · serde · anyhow · Python 3.9+ · peft/transformers (optional, `toki[train]`) · NumPy · hatchling

## Commands
```bash
cargo build                                    # build Rust CLI
cargo test                                     # run Rust unit tests
cargo clippy -- -D warnings                    # lint
cargo run -- generate --category jailbreak     # generate adversarial prompts
cargo run -- evaluate --model mock             # evaluate robustness
cargo run -- pipeline --config config.json     # full generate → eval → finetune pipeline
python -m pytest tests/ -x                     # Python test suite
python -m toki generate --category edge_case   # Python CLI
python -m toki run --config config.json        # run experiment
python -m toki list                            # list saved experiments
```

## Critical Constraints
- No `unwrap()`/`expect()` outside tests — use `anyhow::Result` and `?`
- No silent failures — log a warning when a fallback path swallows an error
- `peft`/`transformers` are **optional** — `LoRAFinetuner` must raise `ImportError` cleanly when absent
- Statistical claims require both paired t-test and Wilcoxon signed-rank test — never claim a win on mean alone when confidence intervals overlap
- `ExperimentResult.save()` must never overwrite — always create a new timestamped directory under `experiments/runs/`
- `AdversarialDataset` deduplicates on content hash — never store duplicate prompts
- Rust `ExperimentRunner` delegates to Python subprocess — never re-implement Python logic in Rust
- Safety scores are float in [0.0, 1.0] — assert at `RobustnessEvaluator` output boundary
- Version bumps touch `pyproject.toml` + `toki/__init__.py`

## Crate / Module Map
| Component | Role |
|-----------|------|
| `src/main.rs` | Rust CLI: `generate`, `evaluate`, `finetune`, `config`, `pipeline` subcommands |
| `src/config.rs` | `TokiConfig` — serde JSON config with save/load |
| `src/runner.rs` | `ExperimentRunner` — orchestration, delegates to Python subprocess |
| `python/toki/generator.py` | `AdversarialGenerator` — 4 categories × templates × payloads |
| `python/toki/evaluator.py` | `RobustnessEvaluator` — model-agnostic 0.0–1.0 safety scorer |
| `python/toki/dataset.py` | `AdversarialDataset` — dedup, persistence (JSON), category queries |
| `python/toki/finetune.py` | `LoRAFinetuner` + `LoRAConfig` — HF PEFT wrapper (optional) |
| `python/toki/experiment.py` | `TokiExperiment` — generate → evaluate → [finetune] → evaluate → save |
| `python/toki/results.py` | `ExperimentResult` — dataclass with save/load/list_experiments |
| `python/toki/benchmark.py` | `BenchmarkStats`, `paired_t_test`, `wilcoxon_test`, `BenchmarkReport` |

## Planning Docs
- `PLAN.md` — current phase state and version history
- `CHANGELOG.md` — all notable changes

## Skills
See `.claude/skills/` — auto-loaded when relevant.
Run `/konjo` to boot a full session (Brief + Discovery + Plan).
