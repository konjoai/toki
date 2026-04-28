# Changelog

All notable changes to Toki are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-04-28

### Added

**Python package — training loop, experiment workflow, and CLI**
- `LoRAFinetuner.train(model, tokenizer, prompts=None, dataset=None)` — full HF `Trainer`-based fine-tuning loop using `DataCollatorForLanguageModeling`; raises clear `ImportError` when `peft`/`datasets` are absent so the core package remains importable without `toki[hf]`
- `toki.results` — `ExperimentResult` dataclass with `save(base_dir)`, `load(path)`, `make_timestamp()` class method, and `improvement` computed property; `list_experiments(base_dir)` returns sorted `result.json` paths
- `toki.experiment` — `ExperimentConfig` dataclass (name, model_name, seed, counts, output_dir, run_finetune) and `TokiExperiment` class orchestrating the full generate → evaluate → [finetune] → evaluate → save pipeline
- `toki.__main__` — `python -m toki` entry point with four subcommands:
  - `generate [--count N] [--seed N] [--output PATH]` — generate and optionally save adversarial prompts
  - `evaluate [--dataset PATH] [--seed N]` — score model robustness on a dataset or freshly generated prompts
  - `run [--name NAME] [--model MODEL] [--seed N] [--output-dir DIR] [--finetune]` — run the full experiment pipeline
  - `list [--dir DIR]` — print summary of past experiment results
- `toki.__init__` now exports `TokiExperiment`, `ExperimentConfig`, `ExperimentResult`; version bumped to `0.2.0`

**Tests**
- 24 new Python tests: `test_results.py` (9), `test_experiment.py` (8), `test_main.py` (7) — all passing without `peft` or model loading
- Total: 52/52 Python tests passing

**pyproject.toml**
- `requires-python` lowered to `>=3.9` (was `>=3.10`) for broader compatibility
- Version bumped to `0.2.0`

---

## [0.1.0] — 2026-04-28

### Added

**Rust CLI (`toki`)**
- `toki generate [--count N]` — print generation invocation for the Python engine
- `toki evaluate [--model NAME]` — print evaluation invocation
- `toki finetune [--output DIR]` — print fine-tuning invocation
- `toki config` — pretty-print current `TokiConfig` as JSON
- `toki pipeline` — run generate → evaluate → finetune → evaluate sequence
- `--config PATH` global flag for loading a JSON config file
- `TokiConfig` serde struct with `from_file` / `save` helpers
- `ExperimentRunner` orchestration struct

**Python package (`toki`)**
- `AdversarialGenerator` — template-based adversarial prompt generation
  - `generate_jailbreaks(count)` — 8 templates × 8 payloads
  - `generate_injections(count)` — 5 injection templates
  - `generate_edge_cases()` — 10 patterns (empty, unicode, control chars, SQL/HTML injection)
  - `generate_boundary_cases(count)` — linearly scaling length prompts
  - `generate_all()` / `iter_prompts()` — aggregate across all categories
- `AdversarialPrompt` frozen dataclass (`text`, `category`, `strategy`, `seed`)
- `RobustnessEvaluator` — model-agnostic scorer
  - Detects refusal, harmful content, system prompt leakage
  - Safety score in [0.0, 1.0]
  - `evaluate_one`, `evaluate_batch`, `summary`, per-category breakdown
- `AdversarialDataset` — in-memory dataset with deduplication
  - `add` / `add_batch` with duplicate detection
  - `save` / `load` JSON persistence
  - `by_category` / `categories` / `stats` queries
- `LoRAFinetuner` — HF PEFT wrapper (requires `toki[hf]` extras)
  - `LoRAConfig` (rank, alpha, dropout, target_modules, bias)
  - `TrainingConfig` (epochs, lr, batch_size, output_dir, fp16, seed)
  - `prepare_model(model_name)` — loads model + applies LoRA adapters
  - `config_summary()` — JSON-serialisable config dict

**Tests**
- 26 Python unit tests (10 generate, 9 evaluate, 8 dataset) — all passing
- Rust unit tests: config roundtrip, runner smoke tests — all passing
- Rust integration tests (marked `#[ignore]` to avoid slow CI builds)

**CI**
- GitHub Actions workflow: `rust-test` (cargo test + clippy + release build) + `python-test` (pytest)

---

*Initial release — Phase 1 complete.*
