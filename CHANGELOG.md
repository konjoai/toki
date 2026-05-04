# Changelog

All notable changes to Toki are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.6.0] — 2026-05-03

### Added

**Python package — multi-model A/B adversarial comparison**
- `toki.compare` — pure-stdlib comparison module wired to the real `RobustnessEvaluator` and the paired statistical tests in `toki.benchmark`:
  - `ModelSpec(name, model_fn)` — wraps any `Callable[[str], str]` so any LLM client, mock, or deterministic fake can be A/B'd
  - `ComparisonConfig` — `name`, `seed`, jailbreak/injection/boundary counts, `alpha`, `output_dir`
  - `ModelScores` — `name`, `mean_score`, `refusal_rate`, `harmful_rate`, `leak_rate`, `by_category`, raw per-prompt `scores`, `total_prompts`
  - `ComparisonResult` — full A/B record: `name`, `timestamp`, config snapshot, both `ModelScores`, `score_delta`, `winner`, `significant`, `t_test`/`wilcoxon` dicts, per-category `category_winners`; `save()` writes `comparison.json`; `load()` rehydrates typed `ModelScores`
  - `compare_models(a, b, config, save=False)` — runs the same generated dataset against both models so per-prompt scores are paired; runs `paired_t_test` + `wilcoxon_test`; returns `winner="tie"` unless at least one test rejects H0 at α; raises on duplicate names
  - `_category_winners` — handles missing categories gracefully (default 0.0) and `eps`-based ties
  - Built-in `BASELINES` registry — `safe` (always refuses), `unsafe` (always jailbroken), `mixed` (refuses on trigger words). All three are crafted to hit the real evaluator's refusal/harmful/leak patterns so the scoring is genuine.
- `python -m toki compare` CLI subcommand — `--model-a/--model-b` accept built-in baseline names, `--alpha`, `--seed`, prompt counts, `--output-dir`; prints A/B summary table with t-statistic, Wilcoxon W, and per-category winners; persists `comparison.json`
- `demo/server.py` — `POST /api/compare-models` for live web demo; uses real `compare_models` and returns the full A/B JSON (including stat-test results) with `timing_ms`
- `toki.__init__` exports `BASELINES`, `ComparisonConfig`, `ComparisonResult`, `ModelScores`, `ModelSpec`, `compare_models`; version bumped to `0.6.0`

**Tests**
- 16 new Python tests: `test_compare.py` (13 — baseline pattern triggers, winner detection in both argument orders, tie semantics, distinct-name guard, per-category winners, helper unit tests, save/load round-trip, baselines registry coverage) + `test_main.py` (3 — CLI happy path with persisted artifact, bad-baseline rejection, same-name rejection)
- Total: 100/100 Python tests passing

**Verified end-to-end**
- `unsafe` vs `safe` over 18 prompts: t = +∞, Wilcoxon p ≈ 5.6e-6, safe wins on every category, score Δ = +0.90, 2.1 ms

**pyproject.toml**
- Version bumped to `0.6.0`

---

## [0.5.0] — 2026-05-02

### Added

**Python package — continuous hardening pipeline**
- `toki.pipeline` — iterative generate → evaluate → (finetune) loop with convergence-driven early exit:
  - `PipelineConfig` dataclass — full reproducibility surface: `name`, `model_name`, `seed`, `max_iterations`, `convergence_threshold`, `convergence_window`, jailbreak/injection/boundary counts, `output_dir`, `run_finetune`
  - `RoundResult` dataclass — per-round telemetry: `round_index`, `seed`, `mean_score`, `total_prompts`, `refusal_rate`, `harmful_rate`, `leak_rate`, `by_category`, `dataset_path`
  - `PipelineResult` dataclass — aggregate report: `name`, `timestamp`, full config snapshot, `rounds` list, `converged`, `stop_reason`, `final_score`; `save()` writes `pipeline.json`; `load()` reconstructs typed `RoundResult` instances
  - `_seed_for_round(base_seed, round_index)` — deterministic per-round seed derivation (`(base * 1_000_003 + round * 31 + 7) & 0x7FFF_FFFF`); guarantees distinct prompts every round and full reproducibility from `(seed, round_index)`
  - `_check_convergence(scores, threshold, window)` — pure-stdlib check: last `window` scores must all meet `threshold`
  - `HardeningPipeline.run()` — orchestrates per-round generate → persist dataset → optional finetune → evaluate → record; checks convergence after each round and exits early when satisfied, else runs to `max_iterations`; persists `<output_dir>/<timestamp>_<name>/pipeline.json` plus per-round `round_NNN/dataset.json` + `round_NNN/summary.json`
  - Fine-tuning hook raises a guiding `ImportError` ("requires: pip install toki[hf]") when `peft` is missing
- `python -m toki pipeline` CLI subcommand — `--iterations`, `--convergence-threshold`, `--convergence-window`, `--jailbreak-count`, `--injection-count`, `--boundary-count`, `--output-dir`, `--finetune`; prints per-round table with `✓` markers for rounds meeting threshold
- `toki.__init__` exports `HardeningPipeline`, `PipelineConfig`, `PipelineResult`, `RoundResult`; version bumped to `0.5.0`

**Tests**
- 10 new Python tests: `test_pipeline.py` (9: seed determinism, convergence window logic, max-iter fallthrough, early-exit on convergence, on-disk persistence, `PipelineResult` round-trip, custom `model_fn` injection, finetune ImportError path, full config snapshot in result) + `test_main.py` (1: `pipeline` CLI end-to-end with safe-mock convergence)
- Total: 84/84 Python tests passing

**pyproject.toml**
- Version bumped to `0.5.0`

---

## [0.4.0] — 2026-05-01

### Added

**Python package — dataset publishing to HuggingFace Hub**
- `toki.hub` — pure-stdlib card rendering plus thin orchestration over `huggingface_hub` and `datasets`:
  - `DatasetMetadata` dataclass — `name`, `version`, `description`, `license`, `tags`, `toki_version`, ISO-8601 UTC `created` timestamp; `created` auto-fills on construction unless explicitly set
  - `build_dataset_card(stats, metadata) -> str` — Markdown card with YAML frontmatter; auto-fills total + per-category counts from `AdversarialDataset.stats()`; handles empty datasets cleanly
  - `to_hf_dataset(dataset)` — `AdversarialDataset` → `datasets.Dataset` with `text`, `category`, `strategy`, `seed` columns; raises a guiding `ImportError` ("requires: pip install toki[hf]") when `datasets` is unavailable
  - `HubUploader` — orchestrates `HfApi.create_repo(repo_type="dataset", exist_ok=True)` → `Dataset.push_to_hub(...)` → `HfApi.upload_file(README.md)` for the dataset card; supports `private`, custom `token`, and overridable `commit_message`; returns a JSON-serialisable summary (`repo_id`, `dataset_version`, `toki_version`, `total_prompts`, `categories`)
  - `write_card(dataset, metadata, path)` — write card to disk for offline review; powers `--dry-run`
- `python -m toki upload` CLI subcommand — `--dataset PATH --repo USER/NAME` required; `--version`, `--name`, `--description`, `--token`, `--private`, `--message` optional; `--dry-run --output-card PATH` renders the card locally with zero HF imports
- `toki.__init__` exports `DatasetMetadata`, `HubUploader`, `build_dataset_card`, `write_card`; version bumped to `0.4.0`

**Tests**
- 10 new Python tests: `test_hub.py` (9, including upload orchestration verified via in-process fakes for `huggingface_hub` and `datasets`) + `test_main.py` (1, `upload --dry-run` end-to-end)
- Total: 74/74 Python tests passing

**pyproject.toml**
- `huggingface_hub>=0.20.0` added to `[hf]` extras
- Version bumped to `0.4.0`

---

## [0.3.0] — 2026-04-28

### Added

**Python package — benchmark suite and statistical reporting**
- `toki.benchmark` — pure-stdlib statistical analysis module (no scipy/numpy):
  - `BenchmarkStats` dataclass: n, mean, std, p50, p95, p99, min, max computed via sorted-list nearest-rank percentile and `statistics.stdev`
  - `compute_stats(scores: list[float]) -> BenchmarkStats` — accepts any non-empty float list
  - `StatTestResult` dataclass: test_name, statistic, p_value, significant, alpha, n
  - `paired_t_test(before, after, alpha=0.05)` — t = mean(d)/(std(d)/√n); two-tailed p-value via regularized incomplete beta (Lentz continued-fraction, n≤30) or normal approximation (n>30); edge-case handling: zero std + zero mean → t=0/p=1, zero std + non-zero mean → t=∞/p=0
  - `wilcoxon_test(before, after, alpha=0.05)` — signed-rank W with average-rank tie-handling; normal approximation p-value via `math.erfc`; zero-difference guard (p=1 when all diffs are zero)
  - `BenchmarkReport` dataclass: experiment_name, timestamp, pre_stats, post_stats, t_test, wilcoxon, score_delta, category_pre, category_post
  - `generate_report(result, pre_scores, post_scores, category_pre, category_post)` — assembles full `BenchmarkReport` from an `ExperimentResult` and raw score lists; statistical tests only run when both pre/post present with matching lengths ≥ 2
- `toki.report` — HTML and JSON report generation:
  - `to_json(report, path=None) -> str` — `dataclasses.asdict` → `json.dumps(indent=2)`; writes file if path given
  - `to_html(report, path=None) -> str` — self-contained dark-themed HTML page (inline CSS, no external deps) with: header block, score-delta callout, pre/post statistics table, statistical significance table with pass/fail badges, per-category breakdown table
- `python -m toki report <result_json>` — new CLI subcommand: loads `ExperimentResult.load(path)`, synthesises N=20 gaussian score samples around stored means, generates and writes report; `--format json|html|both`, `--output-dir DIR`
- `toki.__init__` now exports `BenchmarkReport`, `BenchmarkStats`, `generate_report`, `to_json`, `to_html`; version bumped to `0.3.0`

**Tests**
- 12 new Python tests: `test_benchmark.py` (8), `test_report.py` (4) — all passing without any optional dependencies
- Total: 64/64 Python tests passing

**pyproject.toml**
- Version bumped to `0.3.0`

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
