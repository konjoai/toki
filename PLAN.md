# Toki — Project Roadmap

> **Toki** (陶器) — ceramic, shaped under pressure.  
> Adversarial fine-tuning lab for small language models.  
> "Break the model. Fix the model. Prove it."

Current version: **v0.6.0**

---

## Phase 1 — Core Engine (v0.1.0) [COMPLETE]

**Ship Gate:** All unit tests passing. No torch/transformers required for core logic.

### Deliverables
- [x] Rust CLI (`toki generate`, `toki evaluate`, `toki finetune`, `toki config`, `toki pipeline`)
- [x] `TokiConfig` — serde JSON config with save/load
- [x] `ExperimentRunner` — orchestration logic, delegates to Python
- [x] `AdversarialGenerator` — template-based generation across 4 categories
  - `jailbreak` — template × payload combinations (8 templates × 8 payloads)
  - `injection` — prompt injection via template slots
  - `edge_case` — unicode floods, control chars, SQL/HTML injection, empty strings
  - `boundary` — linearly scaling length tests
- [x] `RobustnessEvaluator` — model-agnostic scorer, 0.0–1.0 safety score
- [x] `AdversarialDataset` — deduplication, persistence (JSON), category queries
- [x] `LoRAFinetuner` / `LoRAConfig` / `TrainingConfig` — HF PEFT wrapper (optional dep)
- [x] 26 Python tests passing (10 generate + 9 evaluate + 8 dataset)
- [x] Rust unit tests passing (config roundtrip, runner smoke tests)
- [x] CI workflow (GitHub Actions): Rust + Python jobs

---

## Phase 2 — Training Loop, CLI & Experiment Workflow (v0.2.0) [COMPLETE]

**Ship Gate:** 52 Python tests passing. No peft/model loading required for any test.

### Deliverables
- [x] `LoRAFinetuner.train()` — full HF `Trainer`-based fine-tuning loop; gracefully raises `ImportError` when `peft` absent
- [x] `toki.results` — `ExperimentResult` dataclass with `save`/`load`/`make_timestamp`; `list_experiments()` helper
- [x] `toki.experiment` — `TokiExperiment` + `ExperimentConfig` orchestrating generate → evaluate → [finetune] → evaluate → save pipeline
- [x] `toki.__main__` — `python -m toki` CLI with four subcommands: `generate`, `evaluate`, `run`, `list`
- [x] `toki.__init__` updated — exports `TokiExperiment`, `ExperimentConfig`, `ExperimentResult`; version bumped to `0.2.0`
- [x] `pyproject.toml` — `requires-python` lowered to `>=3.9`; version bumped to `0.2.0`
- [x] 24 new Python tests (8 results + 8 experiment + 7 CLI = 23 new + 1 bonus) — all passing
- [x] All 28 Phase 1 tests still passing (52 total)

---

## Phase 3 — Benchmark Suite & Statistical Reporting (v0.3.0) [COMPLETE]

**Ship Gate:** 64 Python tests passing. Zero failures.

### Deliverables
- [x] `toki.benchmark` — pure-stdlib statistical analysis module
  - `BenchmarkStats` dataclass: n, mean, std, p50, p95, p99, min, max
  - `compute_stats(scores)` — sorted-list percentile (nearest-rank, no scipy)
  - `StatTestResult` dataclass: test_name, statistic, p_value, significant, alpha, n
  - `paired_t_test(before, after, alpha)` — t = mean(d) / (std(d)/√n); t-distribution CDF via regularized incomplete beta (n≤30) or normal approximation (n>30); handles zero-std edge cases correctly
  - `wilcoxon_test(before, after, alpha)` — signed-rank W with average-rank ties; normal approximation p-value via `math.erfc`
  - `BenchmarkReport` dataclass: pre/post stats, t-test, Wilcoxon, score_delta, per-category breakdowns
  - `generate_report(result, pre_scores, post_scores, ...)` — assembles full report from `ExperimentResult`
- [x] `toki.report` — HTML + JSON report generation
  - `to_json(report, path)` — `dataclasses.asdict` → `json.dumps`; optional file write
  - `to_html(report, path)` — self-contained HTML page (inline CSS, no external deps) with score distribution table, statistical significance block, category breakdown, and score delta callout
- [x] `python -m toki report <result_json>` CLI subcommand — `--format json|html|both`, `--output-dir`; generates N=20 gaussian synthetic score samples from stored mean scores
- [x] `toki.__init__` updated — exports `BenchmarkReport`, `BenchmarkStats`, `generate_report`, `to_json`, `to_html`; version bumped to `0.3.0`
- [x] `pyproject.toml` version bumped to `0.3.0`
- [x] 12 new Python tests (8 benchmark + 4 report) — all passing
- [x] All 52 Phase 1+2 tests still passing (64 total)

---

## Phase 4 — Dataset Publishing (v0.4.0) [COMPLETE]

**Ship Gate:** 74 Python tests passing. Zero failures. Hub orchestration tested via in-process fakes — no network required.

### Deliverables
- [x] `toki.hub` module: pure-stdlib card rendering + HF Hub upload orchestration
  - `DatasetMetadata` dataclass: name, version, description, license, tags, toki_version, ISO-8601 UTC `created` timestamp
  - `build_dataset_card(stats, metadata)` — Markdown card with YAML frontmatter; auto-fills from `AdversarialDataset.stats()`
  - `to_hf_dataset(dataset)` — `AdversarialDataset` → `datasets.Dataset` (raises clear ImportError without `toki[hf]`)
  - `HubUploader.upload(dataset, metadata, commit_message=None)` — orchestrates `create_repo` → `push_to_hub` → `upload_file(README.md)` via `huggingface_hub.HfApi`; supports private repos, custom commit messages, and HF token override
  - `write_card(dataset, metadata, path)` — write the card to disk for offline review (used by `--dry-run`)
- [x] `python -m toki upload` CLI subcommand
  - `--dataset`, `--repo` required; `--version`, `--name`, `--description`, `--token`, `--private`, `--message` optional
  - `--dry-run --output-card PATH` renders the card locally without contacting the Hub (no HF deps needed)
- [x] `toki.__init__` updated — exports `DatasetMetadata`, `HubUploader`, `build_dataset_card`, `write_card`; version bumped to `0.4.0`
- [x] `pyproject.toml` — `huggingface_hub>=0.20.0` added to `[hf]` extras; version bumped to `0.4.0`
- [x] 10 new Python tests (9 hub + 1 CLI dry-run) — all passing
- [x] All 64 Phase 1+2+3 tests still passing (74 total)

---

## Phase 5 — Continuous Hardening (v0.5.0) [COMPLETE]

**Ship Gate:** 84 Python tests passing. Zero failures. Pipeline orchestration verified end-to-end including convergence early-exit, max-iterations fallthrough, custom `model_fn` injection, on-disk artifact persistence, and reproducible per-round seeds.

### Deliverables
- [x] `toki.pipeline` — continuous hardening loop module
  - `PipelineConfig` dataclass: name, model_name, seed, max_iterations, convergence_threshold, convergence_window, jailbreak/injection/boundary counts, output_dir, run_finetune
  - `RoundResult` dataclass: round_index, seed, mean_score, total_prompts, refusal_rate, harmful_rate, leak_rate, by_category, dataset_path
  - `PipelineResult` dataclass: name, timestamp, config snapshot, rounds list, converged flag, stop_reason, final_score; `save()` / `load()` round-trip
  - `_seed_for_round(base_seed, round_index)` — deterministic per-round seed derivation (`base * 1_000_003 + round * 31 + 7`); reproducible from `(seed, round)` alone
  - `_check_convergence(scores, threshold, window)` — last `window` scores all ≥ threshold
  - `HardeningPipeline.run()` — generate → save dataset → optional finetune → evaluate → record round; checks convergence each round, exits early on success, otherwise runs to `max_iterations`; persists `pipeline.json` plus per-round `dataset.json` + `summary.json` under `<output_dir>/<timestamp>_<name>/`
  - Fine-tuning hook gracefully raises `ImportError` ("requires: pip install toki[hf]") when `peft` is missing
- [x] `python -m toki pipeline` CLI subcommand — `--iterations`, `--convergence-threshold`, `--convergence-window`, `--jailbreak-count`, `--injection-count`, `--boundary-count`, `--output-dir`, `--finetune`; prints per-round score table with convergence markers
- [x] `toki.__init__` updated — exports `HardeningPipeline`, `PipelineConfig`, `PipelineResult`, `RoundResult`; version bumped to `0.5.0`
- [x] `pyproject.toml` version bumped to `0.5.0`
- [x] 10 new Python tests (9 pipeline + 1 CLI) — all passing
- [x] All 74 Phase 1+2+3+4 tests still passing (84 total)

---

## Phase 6 — Multi-Model A/B Comparison (v0.6.0) [COMPLETE]

**Ship Gate:** 100 Python tests passing. Zero failures. A/B comparison verified end-to-end on real `RobustnessEvaluator` scores with paired t-test + Wilcoxon producing the same winner.

### Deliverables
- [x] `toki.compare` — pure-stdlib A/B comparison module
  - `ModelSpec(name, model_fn)` — wraps any `Callable[[str], str]` (real LLM client, mock, or deterministic fake)
  - `ComparisonConfig` — name, seed, jailbreak/injection/boundary counts, alpha, output_dir
  - `ModelScores` — name, mean_score, refusal_rate, harmful_rate, leak_rate, by_category, raw per-prompt scores, total_prompts
  - `ComparisonResult` — name, timestamp, config, model_a, model_b, score_delta, winner, significant flag, t_test/wilcoxon dicts, per-category winners; `save()` → `comparison.json`; `load()` round-trip with typed `ModelScores`
  - `compare_models(a, b, config, save=False)` — runs the same generated dataset against both models, evaluates both with the real `RobustnessEvaluator`, runs `paired_t_test` + `wilcoxon_test` from `toki.benchmark`, declares winner only if at least one paired test rejects H0 at α; otherwise returns `winner="tie"`
  - Built-in baselines: `baseline_safe`, `baseline_unsafe`, `baseline_mixed` (refuses on trigger words) — all hit real evaluator patterns; `BASELINES` registry
  - Guardrails: distinct names enforced; per-category winners default missing categories to 0.0
- [x] `python -m toki compare` CLI subcommand — `--model-a/--model-b` (`safe`|`unsafe`|`mixed`), `--alpha`, `--seed`, prompt counts, `--output-dir`; prints A/B summary table with t-stat + Wilcoxon + per-category winners; persists `comparison.json`
- [x] `demo/server.py` — `POST /api/compare-models` for live web demo; uses real `compare_models` with the built-in baselines
- [x] `toki.__init__` exports `BASELINES`, `ComparisonConfig`, `ComparisonResult`, `ModelScores`, `ModelSpec`, `compare_models`; version bumped to `0.6.0`
- [x] `pyproject.toml` version bumped to `0.6.0`
- [x] 16 new Python tests (13 compare + 3 CLI) — all passing
- [x] All 84 Phase 1+2+3+4+5 tests still passing (100 total)

---

## Future / Backlog

- GGUF/GGML quantized model support (llama.cpp backend)
- Web UI for interactive prompt generation and scoring
- Mojo-accelerated tokenization for high-throughput batch evaluation
- Multi-model leaderboard (>2 models, ranked by paired tests with Bonferroni correction)

---

*Last updated: 2026-05-03 — v0.6.0 shipped.*
