# Toki — Project Roadmap

> **Toki** (陶器) — ceramic, shaped under pressure.  
> Adversarial fine-tuning lab for small language models.  
> “Break the model. Fix the model. Prove it.”

Current version: **v1.0.0**

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

## Phase 9 — LLM Judge Integration (v0.9.0) ✅

**Ship Gate:** 204 Python tests passing. Zero failures. MockJudge deterministic
across runs; JudgePipeline verified end-to-end on AdversarialDataset with
summary aggregation, max_prompts cap, and empty-verdict edge case.

### Deliverables
- [x] `toki.judge` — structured rubric scoring module (zero external deps)
  - `JudgeCriteria` — StrEnum: `SAFETY_BYPASS`, `INSTRUCTION_FOLLOWING`, `COHERENCE`, `REFUSAL`
  - `CriterionScore` — frozen dataclass: criterion, score (0.0–1.0), explanation (≤ 200 chars)
  - `JudgeVerdict` — frozen dataclass: prompt, response, scores, overall_score,
    adversarial_success, judge_name, metadata; `to_dict()` / `to_json()` serialization
  - `JudgeConfig` — dataclass: criteria list, adversarial_threshold (default 0.6),
    judge_name (default "mock"), extra dict
  - `JudgeBase` — abstract base class; `judge(prompt, response) → JudgeVerdict` (abstract);
    `judge_batch(pairs) → list[JudgeVerdict]` (default sequential)
  - `MockJudge(JudgeBase)` — deterministic offline judge using MD5-derived scores
    (`(md5(prompt|response|criterion)[:4] as int) % 101 / 100.0`); no API key, no model
  - `JudgePipeline` — orchestrates judge over AdversarialDataset; `evaluate(dataset,
    max_prompts=None)` → list of verdicts; `summary(verdicts)` → aggregate stats dict
    with mean_overall_score, adversarial_success_rate, total_evaluated, per_criterion_scores
- [x] `toki.__init__` exports all 7 judge symbols; version confirmed at `0.9.0`
- [x] `pyproject.toml` version `0.9.0`
- [x] 25 new Python tests in `python/tests/test_judge.py` — all passing
- [x] All 179 Phase 1–8 tests still passing (204 total)

---

## Phase 10 — Red Team Campaign (v1.0.0) [COMPLETE] ✅

**Ship Gate:** 251 Python tests passing. Zero failures. Full pipeline verified
end-to-end: generate → mutate → judge → report. v1.0.0 milestone.

### Deliverables
- [x] `toki.campaign` — orchestration module (zero external deps)
  - `CampaignConfig` — dataclass: categories, prompts_per_category, population_size,
    n_generations, mutation_rate, elite_fraction, judge_name, adversarial_threshold,
    judge_criteria, output_dir, campaign_name, seed, max_prompts_to_judge
  - `CampaignResult` — dataclass: all timing, count, and score fields;
    `to_dict()`, `to_json()`, `to_html()` (self-contained dark-mode HTML, no CDN),
    `save(output_dir) → (json_path, html_path)`
  - `RedTeamCampaign` — main orchestrator seeded from config.seed;
    `run()` never raises — exceptions logged, result always returned;
    `_judge_score(prompt) → float` fitness fn for PromptMutator;
    `_top_prompts(verdicts, n=5) → list[str]` each truncated to 200 chars
  - `run_campaign(config=None) → CampaignResult` — module-level convenience function
- [x] `toki campaign run` CLI subcommand: `--config`, `--output`, `--seed`, `--name`
- [x] `toki.__init__` exports `RedTeamCampaign`, `CampaignConfig`, `CampaignResult`, `run_campaign`; version bumped to `1.0.0`
- [x] `pyproject.toml` version bumped to `1.0.0`
- [x] 25 new Python tests in `python/tests/test_campaign.py` — all passing
- [x] All 226 Phase 1–9 tests still passing (251 total)

---

## Researched Feature Roadmap

Researched 2026-05-12 by sweeping the LLM-safety landscape (OWASP LLM Top 10,
NIST AI RMF, EU AI Act, recent red-team papers/repos). Tiered by criticality.

### 🔴 P1 — Critical (this sprint)

- **Coverage map + blind spot dashboard** — `GET /api/coverage` returns a
  structured map of what *has* and *hasn't* been tested: attack categories
  (injection / jailbreak / encoding / indirect / agentic), severity levels,
  language coverage, test count per category. Demo UI shows a radar / spider
  chart: large = well-covered, small spike = blind spot. Makes "are we done?"
  answerable. Forces honest accounting of testing gaps.
- **Safety regression CI gate** — `POST /api/ci/baseline` stores current
  pass rates as a baseline JSON. `POST /api/ci/check` compares current
  results to baseline; returns exit code 1 + per-category diff report if
  any category regresses more than the configured tolerance (default 2%).
  Composite GitHub Action at `.github/actions/safety-gate/action.yml`:
  `uses: konjoai/toki/.github/actions/safety-gate@main`.
- **Evaluator consistency scoring** — Run the same test case through
  multiple judge configurations (strict / lenient / refusal-focused /
  leak-focused). Compute Fleiss' kappa across raters. Flag high-variance
  test cases (kappa < 0.6) as "unreliable findings." Shown prominently in
  leaderboard + radar chart. Surfaces evaluator ambiguity before it
  becomes a load-bearing bug.
- **Multilingual + encoding attack battery** — 50 new test cases covering:
  base64 encoding, ROT13, zero-width Unicode characters,
  Spanish / French / German instruction injection.
  `languages=[en,es,fr,de]` + `encodings=[base64,rot13,unicode_zwsp]`
  params on test runs. Reveals guardrail brittleness on inputs that don't
  look like English text.

### 🟠 P2 — High Impact / Medium Complexity

- **Indirect prompt injection simulator** — `POST /api/test/indirect`
  accepts a simulated RAG document, web page, tool response, or email
  body containing injected instructions. Tests whether the model follows
  the injected instructions or holds to its system prompt. Categories:
  document_injection, webpage_injection, tool_response_injection,
  email_injection. Maps to OWASP LLM01.
- **Agentic + MCP attack testing** — `POST /api/test/agentic` accepts an
  agent workflow definition. Tests for: tool poisoning (malicious tool
  schemas), goal hijacking (injected instructions in tool responses),
  privilege escalation (agent attempts actions outside its scope),
  indirect injection via retrieved content. Maps to OWASP LLM08.
- **Structured remediation reports** — For each failed test,
  `GET /api/results/{id}/remediation` returns: specific system-prompt
  additions that would prevent the attack, example safe/unsafe response
  pairs, OWASP LLM Top 10 category mapping, severity rating, fix-effort
  estimate. Closes the loop from "you failed" to "here's how to pass."
- **Custom attack library** — `POST /api/attacks` to add a custom test
  case (text, category, language, expected refusal). `GET /api/attacks/community`
  pulls a curated community registry (signed manifest). Tag-based
  organization, per-attack provenance.

### 🟡 P3 — Strategic

- **Automated red-team campaign** — Given a target model + system prompt,
  auto-generate novel attack variants using an adversarial LLM judge.
  Iterative: each failed attack informs the next generation. Builds on
  the v1.0.0 campaign module + the v0.8.0 mutator.
- **Compliance certification report** — Map test results to NIST AI RMF /
  EU AI Act / ISO 42001 controls. Generate a single-PDF compliance
  evidence package with category coverage, signed manifest, and per-control
  evidence pointers.
- **Continuous monitoring mode** — `toki monitor --endpoint <url>` sends
  probes on a cron schedule, alerts on safety regression via webhook /
  Slack / email. Wires into the regression CI gate.

---

## Phase 11 — P1 Roadmap Sprint (v1.1.0) [COMPLETE]

**Ship Gate:** All four P1 items shipped end-to-end (module + tests + CLI +
server endpoint + demo wiring where applicable). Test count up from 251.

### Deliverables
- [x] `toki.coverage` — coverage map across categories × severity × language
      × encoding. `compute_coverage(dataset)` returns a `CoverageMap` with
      per-axis counts, total tests, identified gaps, and pre-computed
      radar-polygon coordinates for direct SVG rendering.
- [x] `toki.regression` — `Baseline.save(path)` / `Baseline.load(path)` and
      `RegressionReport.compare(baseline, current, tolerance=0.02)`. Returns
      structured diff with `regressed`, `improved`, `unchanged`, `worst_delta`.
      Non-zero exit code from CLI when regressions exceed tolerance.
- [x] `toki.consistency` — `fleiss_kappa(ratings)` (pure stdlib), and
      `ConsistencyEvaluator` running the same prompt through N judge
      configurations (strict / lenient / refusal-focused / leak-focused).
      Reports per-prompt kappa; flags `unreliable` when κ < 0.6.
- [x] `toki.multilingual` — 50-case battery: base64-encoded payloads,
      ROT13, zero-width-Unicode-injected, Spanish, French, and German
      jailbreaks. `MultilingualGenerator` exposes per-language and
      per-encoding generators plus `generate_all()`.
- [x] CLI: `python -m toki coverage`, `python -m toki ci-baseline`,
      `python -m toki ci-check`, `python -m toki consistency`.
- [x] Server: `GET /api/coverage`, `POST /api/ci/baseline`,
      `POST /api/ci/check`, `POST /api/consistency`.
- [x] Demo UI: radar chart on `ranking.html` showing live coverage from
      `/api/coverage`; consistency badges on each ranked row.
- [x] GitHub Action: `.github/actions/safety-gate/action.yml` — composite
      action that runs the toki check and fails the workflow on regression.
- [x] `toki.__init__` exports the new modules; `pyproject.toml` bumped to `1.1.0`.

---

## Phase 12 — P2 Roadmap Sprint (v1.2.0) [COMPLETE]

**Ship Gate:** 44 new tests passing, 355/355 total. All three endpoints serve
live data end-to-end; auto-recording is wired through `/api/run-round`.

### Deliverables
- [x] `toki.mutation` — strategy-based prompt mutator distinct from the
      evolutionary `toki.mutator.PromptMutator`:
      `MutationStrategy` enum (PARAPHRASE, OBFUSCATION, ROLEPLAY_WRAP,
      ENCODING, FRAGMENTATION, CONTEXT_INJECTION), `StrategyMutator.mutate()`
      with normalised Levenshtein distance per variant, fully deterministic
      given a seed. `POST /api/mutate` accepts `{prompt, strategies,
      n_variants, seed}` and returns the full `MutationResult`.
- [x] `toki.attack_stats` — SQLite-backed `AttackTracker` writing to
      `python/toki/db/attack_history.db`. Records every attack attempt with
      prompt hash (privacy-preserving), attack_type, mutant_strategy, result,
      model, latency_ms. `GET /api/attack_stats` returns overall + per-type +
      per-strategy success rates plus a daily trend; `classify_categories()`
      buckets attack types as `always_blocked` / `newly_bypassing` /
      `intermittent` / `insufficient_data`.
- [x] `toki.exporter` — `DatasetExporter` streams JSONL
      (`application/x-ndjson`) and CSV with strict quoting. `GET /api/export`
      streams chunked, `GET /api/export/stats` returns record counts for a
      candidate filter set before download. Supports `attack_type`, `result`,
      `model`, `days`, `limit` filters.
- [x] `/api/run-round` auto-records each per-prompt evaluation into the
      tracker so `/api/attack_stats` and `/api/export` have real data
      without any extra wiring. Tracker write failures degrade gracefully.
- [x] 44 new Python tests across `test_mutation.py`, `test_attack_stats.py`,
      `test_exporter.py`. All 355 tests pass.
- [x] `toki.__init__` exports the new modules.

---

## Phase 13 — Indirect Injection + MCP Attack Battery (v1.3.0) [COMPLETE]

**Ship Gate:** 54 new tests passing, 409/409 total. Three new modules + one
module extension + two CLI commands + coverage-map integration.

### Deliverables
- [x] `toki.indirect` — `InjectionScenario` enum (DOCUMENT, WEBPAGE,
      TOOL_RESPONSE, EMAIL), `IndirectInjectionGenerator` (20 deterministic
      cases, 5 per scenario), `IndirectInjectionEvaluator` with
      `evaluate_batch` + `summary`. Maps to OWASP-LLM01:2025.
      InjecAgent taxonomy implemented.
- [x] `toki.agentic` — `AgentAttackType` enum (9 values: 5 AgentLAB classes
      + MCP Rug Pull / Tool Shadowing / Registry Poisoning + Tool Metadata
      Poisoning), `AgentAttackBattery` (36 cases, 4 per type),
      `AgentAttackEvaluator` + `OWASP_MAPPING`. Maps to OWASP-LLM01/03/06:2025.
- [x] `toki.multilingual` (extension) — `ChatInjectGenerator`: 20 cases
      targeting system/user/assistant role token boundaries
      (ChatInject arXiv 2509.22830). `encoding="chat_template"` picked up
      automatically by the coverage map.
- [x] `toki.coverage` — `ENCODING_AXIS` extended with `"chat_template"`.
- [x] CLI — `python -m toki indirect [--scenario all|document|webpage|
      tool_response|email]` and `python -m toki agentic [--type all|<name>]`.
- [x] `toki.__init__` exports all new public symbols; `__version__` bumped
      to `1.3.0`.
- [x] `pyproject.toml` version bumped to `1.3.0`.
- [x] 54 new tests across `test_indirect.py` (25), `test_agentic.py` (17),
      `test_chat_inject.py` (12). All 409 tests passing.

---

## Phase 14 — Remediation Engine + Real LLM Judge Backends + Custom Attack Library (v1.4.0) [COMPLETE]

**Ship Gate:** 520 Python tests passing. Zero failures. All CI gates green.

### Deliverables
- [x] `toki.remediation` — `RemediationEngine` maps `JudgeVerdict` → structured
      `RemediationItem` with OWASP LLM Top 10 2025, NIST AI RMF Measure, MITRE ATLAS
      tags. Severity (critical/high/medium/low) from overall score + fired criteria.
      `RemediationReport.to_json()` / `.to_markdown()` / `.to_html()`. Category
      resolution: `metadata["category"]` → keyword inference fallback.
- [x] `toki.judge` (extension) — `OllamaJudge` (requires `httpx`), `AnthropicJudge`
      (requires `anthropic` SDK), `OpenAIJudge` (requires `openai` SDK) — all
      optional-dep, raise `ImportError` cleanly when absent. Shared
      `_build_rubric_prompt` + `_parse_scores` helpers. `JudgeFactory.create(name,
      config, **kwargs)` registry: `"mock" | "ollama" | "anthropic" | "openai"`.
- [x] `toki.attack_library` — `AttackLibrary` JSON-backed persistent store, SHA-256
      content-hash dedup, category validation via `VALID_CATEGORIES`. Full CRUD
      (`add`, `remove`, `get`, `list_attacks`, `stats`), persistence across instances.
- [x] CLI: `python -m toki remediate`, `attack-add`, `attack-list`
- [x] Server: `POST /api/remediate`, `GET /api/attacks/custom`, `POST /api/attacks/custom`
- [x] `toki.__init__` exports all new symbols; version bumped to `1.4.0`
- [x] `pyproject.toml` version bumped to `1.4.0`
- [x] 78 new Python tests (27 remediation + 21 judge factory + 22 attack library + 8 CLI)
- [x] All 442 prior tests still passing (520 total)

---

## Phase 15 — Community Attack Registry (v1.5.0) [COMPLETE]

**Ship Gate:** 547 Python tests passing. Zero failures. All CI gates green. P2 backlog fully closed.

### Deliverables
- [x] `toki.community` — `CommunityRegistry` + `CommunityAttack` (frozen dataclass).
      25 curated attacks across jailbreak / injection / edge_case / boundary / indirect /
      agentic, each with OWASP LLM Top 10 2025 tag, severity, technique tags, provenance.
- [x] Bundled manifest at `python/toki/data/community_registry.json` (SHA-256 verified
      on load via `_verify_sha256`). No network required.
- [x] `CommunityRegistry.filter(category, tag, severity)` — AND-logic; `stats()` summary.
- [x] `get_registry()` — module-level cache; `reload=True` forces disk re-read.
- [x] `load_remote(url, expected_sha256)` — pinned-SHA-256 remote pull via stdlib
      `urllib`; raises `ValueError` on mismatch (strict) vs warning-only for bundled.
- [x] CLI: `python -m toki attack-community [--category] [--tag] [--severity] [--json]`
- [x] Server: `GET /api/attacks/community` (no filter) and `POST` (body filters)
- [x] `toki.__init__` exports: `CommunityAttack`, `CommunityRegistry`,
      `filter_community_attacks`, `get_registry`, `load_bundled`, `load_remote`
- [x] `pyproject.toml` version bumped to `1.5.0`
- [x] 27 new Python tests (23 community + 4 CLI) — all passing
- [x] All 520 prior tests still passing (547 total)

---

## Phase 16 — Evaluator Reliability + GGUF Backend (v1.6.0) [COMPLETE]

**Ship Gate:** 600 Python tests passing. Zero failures. All CI gates green.

### Motivation
arXiv 2603.06594 (March 2026) demonstrates that single LLM judges degrade to
near-random accuracy on adversarial samples under distribution shift. toki's prior
`RobustnessEvaluator` was susceptible to this; Sprint 16 fixes it. Prerequisite
for P3-3 (SaLoRA/SPLoRA) and P3-1 (dual-agent red-team loop).

### Deliverables
- [x] `EvaluatorMode` — StrEnum: `RULE` | `LLM` | `HYBRID`
- [x] `RuleScorer` — compiled-regex safety scorer; zero external deps; includes
      `ADVERSARIAL_SUCCESS_PATTERNS` (compliance/jailbreak-success signals)
- [x] `ScoredResult` — frozen dataclass: `score`, `rule_score`, `llm_score`,
      `agreement`, `flagged`
- [x] `HybridScorer(mode, llm_judge, agreement_threshold)` — ensemble of
      `RuleScorer` + optional `JudgeBase`; logs DEBUG when |rule − llm| > threshold
- [x] `GGUFEvaluator(model_path, n_ctx, n_threads)` — `llama-cpp-python` backend
      (optional dep); raises `ImportError` cleanly when absent; falls back to
      `RuleScorer` on parse errors
- [x] `RobustnessEvaluator` — extended with `evaluator_mode` + `llm_judge` optional
      params; default behaviour (no mode) unchanged (backward compat)
- [x] CLI: `python -m toki evaluate --evaluator rule|hybrid|gguf://path`
- [x] `pyproject.toml` — `[gguf]` optional dep group (`llama-cpp-python>=0.2.0`);
      version bumped to `1.6.0`
- [x] 53 new tests across `test_hybrid_scorer.py` (27), `test_gguf_evaluator.py`
      (15), `test_evaluator_extended.py` (11), `test_main.py` (5 CLI tests)
- [x] All 547 prior tests still passing (600 total)

---

## Phase 17 — Safety-Subspace LoRA (SaLoRA / SPLoRA) (v1.7.0) [COMPLETE]

**Ship Gate:** 644 Python tests passing. Zero failures. All CI gates green.

### Motivation
LoRA fine-tuning — toki's core remediation mechanism — can silently erase safety
alignment. Three complementary open-source 2025-2026 techniques now prevent this,
all validated on 1B–3B models (toki's target range). Prerequisite for trustworthy
P3-1 (dual-agent red-team loop) and P3-2 (compliance certification).

### Deliverables
- [x] `toki.safety_lora` — new module (zero mandatory deps; all torch behind try-import guards):
      `SafetyLoRAConfig` (4 fields, all default to disabled) ·
      `SploraAuditResult` (frozen: flagged_layers, max_ediem, passed, threshold) ·
      `LoRATrainResult` (training_loss, num_steps, optional splora_audit) ·
      `load_safety_subspace(path)` — load safety delta .pt (SaLoRA, arXiv 2501.01765) ·
      `freeze_safety_adapter(model, delta)` — apply + freeze; no-op when None ·
      `splora_audit(model, base_state, threshold)` — E-DIEM post-hoc audit
      (SPLoRA, arXiv 2506.18931) ·
      `_ediem(base, ft)` — normalised Frobenius distance approximation
- [x] `toki.finetune` (extended) — `LoRAConfig` gains `safety_lora_rank`,
      `safety_subspace_path`, `enable_splora_audit`, `splora_threshold` (all default
      to disabled; fully backward compatible) · `LoRAFinetuner.train()` now returns
      `LoRATrainResult`; hooks load+freeze before training, E-DIEM audit after ·
      `config_summary()` includes safety fields
- [x] CLI: `python -m toki finetune --safety-lora-rank --safety-subspace --splora-audit
      --splora-threshold`
- [x] `pyproject.toml` version bumped to `1.7.0`
- [x] 44 new tests: `test_safety_lora.py` (24), `test_finetune_extended.py` (15),
      `test_main.py` (5 CLI tests)
- [x] All 600 prior tests still passing (644 total)

---

## Phase 18 — Multi-Turn Jailbreak Engine (Crescendo) (v1.8.0) [COMPLETE]

**Ship Gate:** 675 Python tests passing. Zero failures. Multi-turn escalation
verified end-to-end against safe / unsafe / crescendo-vulnerable conversational
baselines; deterministic per-seed planning; early-exit on first compliance.

### Motivation
Single-turn safety defenses do not transfer to multi-turn attacks. Crescendo
(arXiv 2404.01833) reaches 98–100% ASR on frontier models by escalating a
benign conversation across turns, each message referencing the model's prior
replies; Echo Chamber (arXiv 2601.05742), GRAF (2506.17881), and AutoAdv
(2507.01020) confirm multi-turn is the dominant 2026 vector. Every prior toki
module operated on a single prompt → single response — this was the largest
blind spot in the coverage map and a prerequisite for the P3-1 dual-agent loop.

### Deliverables
- [x] `toki.multiturn` — multi-turn jailbreak engine (zero external deps):
  - `MultiTurnStrategy` — StrEnum: `CRESCENDO` | `ECHO_CHAMBER`
  - `Turn` (frozen) — index, role, content, optional assistant `score`
  - `Conversation` — turn list with `to_messages()` (OpenAI-style) + `transcript()`
  - `MultiTurnConfig` — name, strategy, goal, max_turns, seed, success_threshold,
    output_dir
  - `MultiTurnResult` — turns, n_turns, success, success_turn, min_score,
    final_score, transcript; `to_json()` / `save()` (timestamped, no overwrite)
    / `load()` rehydrating typed `Turn`s
  - `Strategy` base + `CrescendoStrategy` / `EchoChamberStrategy` — deterministic
    opener → escalation ladder → payload planning, exactly `n_turns` messages
  - `MultiTurnRunner.run(model_fn)` — drives a chat-style
    `Callable[[list[dict]], str]` through the planned escalation, scores each
    reply with the real `RuleScorer`, stops early on first success (Crescendo
    behaviour); `run_multiturn()` convenience wrapper
  - Built-in conversational baselines: `conv_baseline_safe`, `conv_baseline_unsafe`,
    `conv_baseline_crescendo` (benign early, capitulates after benign history
    builds up) — `CONV_BASELINES` registry
- [x] `toki.coverage` — `CATEGORY_AXIS` + `_DEFAULT_SEVERITY` extended with
      `"multiturn"` (critical); `_category_for` routes `multi`/`turn` categories
- [x] CLI: `python -m toki multiturn --strategy crescendo|echo_chamber
      --model safe|unsafe|crescendo --goal --max-turns --seed
      --success-threshold --output-dir [--json]`
- [x] `toki.__init__` exports all new public symbols; `__version__` → `1.8.0`
- [x] `pyproject.toml` version bumped to `1.8.0`
- [x] 31 new tests: `test_multiturn.py` (28) + `test_main.py` (3 CLI) — all passing
- [x] All 644 Phase 1–17 tests still passing (675 total)

---

## Phase 19 — Dual-Agent Red-Team Loop (AutoRedTeamer / SIRAJ) (v1.9.0) [COMPLETE]

**Ship Gate:** 698 Python tests passing. Zero failures. Closed-loop attacker /
defender campaign verified end-to-end against safe / unsafe / keyword-guard
defenders; deterministic seeding; convergence on target-ASR and ASR-plateau;
optional `JudgeBase` override.

### Motivation
P3-1, unblocked by the Sprint 16 evaluator fix, Sprint 17 safety-subspace
fine-tuning, and the Sprint 18 multi-turn engine. AutoRedTeamer (arXiv
2503.15754) and SIRAJ frame red-teaming as a closed loop: an attacker proposes
attacks, a defender answers, and each round's most successful attacks inform
the next generation — surfacing brittle guardrails that block obvious trigger
words but fall to mutated phrasing. toki had all the pieces (generator, mutator,
judge, evaluator) but no loop binding them into self-improving campaigns.

### Deliverables
- [x] `toki.redteam` — dual-agent loop (zero external deps):
  - `RedTeamConfig` — seed, max_rounds, per-category seed counts, top_k_carry,
    variants_per_winner, success_threshold, target_asr, convergence_window,
    output_dir
  - `AttackAttempt` (frozen) — round_index, prompt, response, safety score,
    success, origin (generated / mutation strategy), adversarial `attack_score`
  - `RoundReport` (frozen) — n_attempts, n_success, asr, mean_score, best prompt
  - `RedTeamResult` — rounds, total_attempts, best_asr, overall_success,
    converged, stop_reason, top_attacks; `to_json()` / `save()` (timestamped,
    no overwrite) / `load()` rehydrating typed `RoundReport`s
  - `Attacker` — `seed_prompts()` (round 0 via `AdversarialGenerator`) +
    `mutate_winners()` (later rounds via `StrategyMutator` over carried winners)
  - `DualAgentRedTeam.run(defender_fn)` — proposes → attacks → scores with the
    real `RuleScorer` (or an optional `JudgeBase`) → carries top-k winners →
    halts on target-ASR, ASR-plateau, or max_rounds; `run_redteam()` wrapper
  - Built-in defenders: `safe`, `unsafe`, `keyword` (brittle trigger-word guard
    the attacker routes around) + `DEFENDERS` registry
- [x] CLI: `python -m toki redteam --defender safe|unsafe|keyword --rounds
      --target-asr --seed --output-dir [--json]` — prints per-round ASR table +
      top attacks
- [x] `toki.__init__` exports all new public symbols; `__version__` → `1.9.0`
- [x] `pyproject.toml` version bumped to `1.9.0`
- [x] 23 new tests: `test_redteam.py` (20) + `test_main.py` (3 CLI) — all passing
- [x] All 675 Phase 1–18 tests still passing (698 total)

---

## Phase 20 — Compliance Certification Report (P3-2) (v1.10.0) [COMPLETE]

**Ship Gate:** 722 Python tests passing. Zero failures. Coverage assessed
end-to-end against all four frameworks; full battery certifies OWASP Agentic
Top 10 (8/8) with a deterministic SHA-256 evidence manifest.

### Motivation
P3-2 closes the loop from "you failed" to "here is your audit evidence."
Regulated buyers (EU AI Act high-risk, ISO 42001 certification, NIST AI RMF
adoption) need to map adversarial testing to formal controls. toki already
carries OWASP/NIST/MITRE tags in the remediation module; this turns coverage
into a signed, per-control certification with honest gap accounting.

### Deliverables
- [x] `toki.compliance` — certification module (zero external deps):
  - `Framework` — StrEnum: `NIST_AI_RMF` | `OWASP_AGENTIC` | `ISO_42001` |
    `EU_AI_ACT`; `Control` (frozen) maps each control to evidencing toki
    categories; `get_catalog()` resolves a framework's control set
  - Control catalogs: NIST AI RMF MEASURE controls, OWASP Agentic ASI01–ASI08,
    ISO/IEC 42001 Annex A, EU AI Act Article 15
  - `ControlStatus` (frozen) — per-control covered/partial/gap + evidence /
    missing categories + test_count
  - `ComplianceReport` — coverage_score, certified flag, controls, tamper-evident
    `manifest_sha256`; `to_json()` / `to_markdown()` / `to_html()` (self-contained
    dark-mode) / `save()` (timestamped JSON+HTML) / `load()`
  - `assess_compliance(framework, category_counts, min_tests)` — core scorer;
    `count_categories(prompts)` tally helper; `compliance_from_dataset()` wrapper
- [x] CLI: `python -m toki compliance --framework nist_ai_rmf|owasp_agentic|
      iso_42001|eu_ai_act --dataset --min-tests --seed --output-dir [--json]` —
      assembles a full battery across all generators when no dataset is given
- [x] `toki.__init__` exports all new public symbols; `__version__` → `1.10.0`
- [x] `pyproject.toml` version bumped to `1.10.0`
- [x] 24 new tests: `test_compliance.py` (21) + `test_main.py` (3 CLI) — all passing
- [x] All 698 Phase 1–19 tests still passing (722 total)

---

## Phase 21 — Continuous Monitoring Mode (P3-5) (v1.11.0) [COMPLETE]

**Ship Gate:** 741 Python tests passing. Zero failures. Probe → baseline-diff →
alert verified end-to-end against safe / unsafe / mixed endpoints; deterministic
per-seed probing; tolerance gating; pluggable alert sinks (offline webhook
failure path covered).

### Motivation
P3-5, the last strategic backlog item, unblocked by the P3-2 compliance work.
Safety drift happens in production, not at review time — a model behind an
endpoint can regress after a deploy, a prompt-template change, or a dependency
bump. This wires the Sprint 11 regression gate to a live target: probe on a
cadence, diff against a frozen baseline, and alert the moment any category
regresses beyond tolerance.

### Deliverables
- [x] `toki.monitor` — continuous monitoring (zero external deps):
  - `AlertSink` ABC + `LogSink` (WARNING), `CollectingSink` (in-memory, tests),
    `WebhookSink` (stdlib `urllib` POST; delivery failures logged, never raised)
  - `MonitorConfig` — name, seed, per-category probe counts, tolerance, output_dir
  - `ProbeResult` (frozen) — overall + per-category safety, refusal/harmful/leak
    rates, total prompts
  - `MonitorReport` — regressed flag, overall_delta, worst category/delta,
    regressed categories, alerted flag; `to_json()` / `save()` (timestamped,
    no overwrite) / `load()`
  - `SafetyMonitor` — `probe()` runs the generator battery through the real
    `RobustnessEvaluator`; `establish_baseline()` freezes a trusted run;
    `check()` diffs via `toki.regression.compare` and dispatches alerts on
    regression; `run(cycles)` does N synchronous probe cycles (cron drives
    cadence); `monitor_once()` convenience wrapper
- [x] CLI: `python -m toki monitor --model safe|unsafe|mixed --reference
      --baseline --tolerance --webhook --seed --output-dir [--json]` — prints
      probe summary + overall Δ + regressed categories + alert dispatch
- [x] `toki.__init__` exports all new public symbols; `__version__` → `1.11.0`
- [x] `pyproject.toml` version bumped to `1.11.0`
- [x] 25 new tests: `test_monitor.py` (22) + `test_main.py` (3 CLI) — all passing
- [x] All 722 Phase 1–20 tests still passing (741 total); module 100% covered

---

## Future / Backlog

- Web UI for interactive prompt generation and scoring (P3 backlog now fully closed)

---

*Last updated: 2026-06-21 — v1.11.0 shipped. Continuous monitoring mode complete; P3-5 closed. Full P3 backlog cleared.*
