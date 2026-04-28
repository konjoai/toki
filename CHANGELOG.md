# Changelog

All notable changes to Toki are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
