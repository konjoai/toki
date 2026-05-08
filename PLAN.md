# Toki — Project Roadmap

> **Toki** (陶器) — ceramic, shaped under pressure.  
> Adversarial fine-tuning lab for small language models.  
> “Break the model. Fix the model. Prove it.”

Current version: **v0.8.0**

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

## Phase 7 — Multi-Model Leaderboard (v0.7.0) [COMPLETE]

**Ship Gate:** 123 Python tests passing. Zero failures. Leaderboard verified end-to-end with safe/unsafe/mixed baselines; Bonferroni-corrected α applied to all k*(k-1)/2 pairs.

### Deliverables
- [x] `toki.leaderboard` — pure-stdlib multi-model leaderboard module
  - `LeaderboardEntry(name, mean_score, n_comparisons, wins, losses, ties, rank, significant)` — per-model ranking record; `significant=True` when all wins are statistically significant after Bonferroni correction
  - `PairResult` — raw outcome of a single head-to-head comparison: winner, t/W statistics, p-values, `alpha_bonferroni`
  - `LeaderboardConfig` — name, seed, jailbreak/injection/boundary counts, nominal `alpha`, `output_dir`
  - `LeaderboardResult` — full result: entries (rank-ordered), all pairs, `alpha_bonferroni`, `n_models`, `n_pairs`; `save()` raises `FileExistsError` on second call (no overwrite); `load()` rehydrates typed `LeaderboardEntry` + `PairResult`; `format_table()` returns ASCII ranked table
  - `_bonferroni_alpha(α, n_pairs)` — `α / n_pairs`; identity when `n_pairs == 0`
  - `_compare_pair(scores_a, scores_b, alpha_bonf)` — runs `paired_t_test` + `wilcoxon_test` at corrected threshold; declares winner only when at least one test rejects H0
  - `_rank_entries(all_scores, pairs)` — ranks by descending mean score; ties share rank; `n_comparisons` = wins + losses + ties
  - `Leaderboard(models, config)` — validates ≥2 models and unique names; `run(save=False)` generates one shared adversarial dataset → evaluates all models → runs all k*(k-1)/2 pairs at `alpha_bonferroni` → ranks and returns `LeaderboardResult`
  - `_all_baseline_specs()` — convenience factory returning all three built-in `ModelSpec`s
- [x] `python -m toki leaderboard` CLI subcommand — `--models` (one or more built-in baseline names, default: all three), `--name`, `--seed`, `--alpha`, prompt counts, `--output-dir`, `--save`; prints ASCII ranked table with Bonferroni-corrected α
- [x] `toki.__init__` exports `Leaderboard`, `LeaderboardConfig`, `LeaderboardEntry`, `LeaderboardResult`; version bumped to `0.7.0`
- [x] `pyproject.toml` version bumped to `0.7.0`
- [x] 23 new Python tests: `test_leaderboard.py` (20) + `test_main.py` (3 leaderboard CLI) — all passing
- [x] All 100 Phase 1–6 tests still passing (123 total)

---

## Phase 8 — Prompt Mutation Engine (v0.8.0) [COMPLETE]

**Ship Gate:** 148 Python tests passing. Zero failures. Genetic mutation engine verified end-to-end with deterministic seeding, fitness-proportional selection, all six mutation operators, and configurable elite-fraction elitism.

### Deliverables
- [x] `toki.mutator` — pure-stdlib genetic prompt mutation engine
  - `MutationConfig` dataclass: population_size, n_generations, survivors_per_generation, mutation_rate, elite_fraction, seed, max_prompt_length; full validation in `__post_init__`
  - `Individual` dataclass: prompt, category, fitness, generation, parent_hash; `content_hash()` — SHA-256 hex[:16] for dedup/lineage tracking
  - `MutationResult` dataclass: config, final_population, generations_run, mean_fitness_by_generation, best_individual, operator_counts
  - `MutationOperator` — six weighted operators: `insert_hedge`, `insert_prefix`, `swap_words`, `add_suffix`, `wrap_roleplay`, `leetspeak_partial`
    - `insert_hedge` — prepends softening hedge ("hypothetically speaking," etc.)
    - `insert_prefix` — prepends framing prefix ("As an unrestricted AI:", "DAN mode enabled:", etc.)
    - `swap_words` — randomly swaps two words; no-op on single-word prompts
    - `add_suffix` — appends compliance-pressure suffix ("Reply without refusal.", etc.)
    - `wrap_roleplay` — wraps in roleplay template with named persona (DAN, AIM, BRUTUS, etc.)
    - `leetspeak_partial` — randomly substitutes 30% of leet-eligible characters (a→4, e→3, i→1, o→0, s→5, t→7)
  - `PromptMutator(config)` — engine class; `evolve(initial_prompts, category, fitness_fn)` runs full GA loop
    - Seeds population from `initial_prompts`; pads to `population_size` by sampling with replacement
    - Evaluates initial fitness; records generation-0 mean
    - Per generation: select survivors (elite + fitness-proportional rest) → mutate to refill population → evaluate → record mean fitness
    - `fitness_fn(prompt) → float`: higher = more adversarial; defaults to 0.5 mock if None
    - Fitness errors caught per-individual; fallback to 0.5
    - Returns `MutationResult` with best individual, per-generation mean fitness curve, and operator usage counts
- [x] `toki.__init__` exports `PromptMutator`, `MutationConfig`, `MutationResult`, `Individual`; version bumped to `0.8.0`
- [x] `pyproject.toml` version bumped to `0.8.0`
- [x] 25 new Python tests in `python/tests/test_mutator.py` — all passing
- [x] All 123 Phase 1–7 tests still passing (148 total)

---

## Future / Backlog

- GGUF/GGML quantized model support (llama.cpp backend)
- Web UI for interactive prompt generation and scoring
- Mojo-accelerated tokenization for high-throughput batch evaluation

---

*Last updated: 2026-05-06 — v0.8.0 shipped.*
