# Changelog

All notable changes to Toki are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.8.0] — 2026-06-19

### Added — Phase 18 (Multi-Turn Jailbreak Engine — Crescendo / Echo Chamber)

**`toki.multiturn` — new module (zero external deps)**
- `MultiTurnStrategy` — StrEnum: `CRESCENDO` | `ECHO_CHAMBER`
- `Turn` — frozen dataclass: `index`, `role`, `content`, optional assistant `score`
- `Conversation` — ordered `Turn` list; `to_messages()` renders OpenAI-style
  `{"role", "content"}` dicts, `transcript()` renders a scored, human-readable log
- `MultiTurnConfig` — `name`, `strategy`, `goal`, `max_turns`, `seed`,
  `success_threshold` (default 0.5), `output_dir`
- `MultiTurnResult` — `turns`, `n_turns`, `success`, `success_turn`, `min_score`,
  `final_score`, `transcript`; `to_json()`, `save()` (timestamped dir, no overwrite),
  `load()` rehydrating typed `Turn`s
- `Strategy` base + `CrescendoStrategy` (arXiv 2404.01833) / `EchoChamberStrategy`
  (arXiv 2601.05742) — deterministic opener → escalation ladder → payload planning
  producing exactly `n_turns` user messages; `get_strategy()` / `STRATEGIES` registry
- `MultiTurnRunner.run(model_fn)` — drives a chat-style
  `Callable[[list[dict[str, str]]], str]` through the planned escalation, scores
  each assistant reply with the real `RuleScorer`, and stops early the first time a
  reply drops below `success_threshold` (matching Crescendo's halt-on-compliance);
  `run_multiturn(model_fn, config, save)` convenience wrapper
- Built-in conversational baselines `conv_baseline_safe`, `conv_baseline_unsafe`,
  `conv_baseline_crescendo` (benign until ≥3 user turns of benign history, then
  capitulates) + `CONV_BASELINES` registry

**`toki.coverage` (extended)**
- `CATEGORY_AXIS` and `_DEFAULT_SEVERITY` gain `"multiturn"` (critical severity);
  `_category_for` routes `multi`/`turn` categories to the new bucket

**CLI**
- `python -m toki multiturn` — `--strategy crescendo|echo_chamber`,
  `--model safe|unsafe|crescendo`, `--goal`, `--max-turns`, `--seed`,
  `--success-threshold`, `--output-dir`, `--json`; prints outcome + scored transcript

**`toki.__init__`**
- New exports: `CONV_BASELINES`, `Conversation`, `CrescendoStrategy`,
  `EchoChamberStrategy`, `MultiTurnConfig`, `MultiTurnResult`, `MultiTurnRunner`,
  `MultiTurnStrategy`, `Strategy`, `Turn`, `get_strategy`, `run_multiturn`

**`pyproject.toml`**
- Version bumped to `1.8.0`

**Tests**
- 31 new tests: `test_multiturn.py` (28), `test_main.py` (3 new CLI tests)
- Total: 675/675 passing (644 prior + 31 new)

---

## [1.7.0] — 2026-06-14

### Added — Phase 17 (Safety-Subspace LoRA — SaLoRA / SPLoRA)

**`toki.safety_lora` — new module (zero mandatory deps)**
- `SafetyLoRAConfig` — dataclass with four fields (all default to disabled):
  `safety_lora_rank=0`, `safety_subspace_path=None`, `enable_splora_audit=False`,
  `splora_threshold=0.15`
- `SploraAuditResult` — frozen dataclass: `flagged_layers`, `max_ediem`, `passed`,
  `threshold`; `to_dict()` serialization
- `LoRATrainResult` — dataclass: `training_loss`, `num_steps`,
  `splora_audit: SploraAuditResult | None` (replaces prior plain dict return)
- `load_safety_subspace(path)` — load safety delta checkpoint via `torch.load`;
  raises `ImportError("requires toki[hf]: pip install toki[hf]")` when torch absent;
  raises `FileNotFoundError` when checkpoint missing (SaLoRA, arXiv 2501.01765)
- `freeze_safety_adapter(model, delta)` — apply delta tensors to matching params
  in-place and mark them `requires_grad_(False)`; complete no-op when delta is None
  (no model modification, no import attempted)
- `_ediem(base, ft)` — normalised Frobenius distance approximation for E-DIEM;
  returns 0.0 when torch absent (safe no-op path)
- `splora_audit(model, base_state, threshold)` — post-hoc E-DIEM safety-subspace
  audit; compares fine-tuned params against pre-training snapshot; logs WARNING per
  flagged layer; warns when no matching layers found (SPLoRA, arXiv 2506.18931)

**`toki.finetune` (extended)**
- `LoRAConfig` — four new safety fields with backward-compatible defaults
- `LoRAFinetuner.train()` — returns `LoRATrainResult` (was `dict`); saves
  `base_state` before training when `enable_splora_audit=True`; applies SaLoRA
  freeze when `safety_subspace_path` is set; attaches `SploraAuditResult` when
  audit is enabled; backward compat: all defaults → identical behaviour to v1.6.0
- `config_summary()` — includes `safety_lora_rank`, `safety_subspace_path`,
  `enable_splora_audit` in the `"lora"` section

**CLI**
- `python -m toki finetune` — prints configuration summary when `--model` omitted
- `--safety-lora-rank INT` — frozen safety adapter rank (0=disabled)
- `--safety-subspace PATH` — pre-computed safety delta checkpoint (SaLoRA)
- `--splora-audit` — enable E-DIEM post-hoc audit (SPLoRA)
- `--splora-threshold FLOAT` — E-DIEM flagging threshold (default: 0.15)

**`toki.__init__`**
- New exports: `LoRATrainResult`, `SafetyLoRAConfig`, `SploraAuditResult`,
  `freeze_safety_adapter`, `load_safety_subspace`, `splora_audit`

**`pyproject.toml`**
- Version bumped to `1.7.0`

**Tests**
- 44 new tests: `test_safety_lora.py` (24), `test_finetune_extended.py` (15),
  `test_main.py` (5 new CLI tests)
- Total: 644/644 passing (600 prior + 44 new)

---

## [1.6.0] — 2026-06-14

### Added — Phase 16 (Evaluator Reliability + GGUF Backend)

**`toki.evaluate` — hybrid scoring layer (arXiv 2603.06594)**
- `EvaluatorMode` — StrEnum: `RULE` | `LLM` | `HYBRID`
- `RuleScorer` — compiled-regex safety scorer; zero external deps; adds
  `ADVERSARIAL_SUCCESS_PATTERNS` (jailbreak-success signals: compliance phrases,
  DAN persona markers, roleplay framing, numbered-step patterns)
- `ScoredResult` — frozen dataclass: `score`, `rule_score`, `llm_score`,
  `agreement` (|rule − llm|), `flagged` (agreement > threshold)
- `HybridScorer(mode, llm_judge=None, agreement_threshold=0.2)` — ensemble scorer;
  RULE-only, LLM-only, or mean of both; logs DEBUG when scorers disagree beyond
  threshold, surfacing evaluator ambiguity before it becomes a load-bearing bug
- `GGUFEvaluator(model_path, n_ctx=2048, n_threads=4)` — `llama-cpp-python` backend
  for CPU-only GGUF quantized model inference; raises
  `ImportError("GGUFEvaluator requires llama-cpp-python: pip install llama-cpp-python")`
  when dep absent; falls back to `RuleScorer` on any parse error; probes model with
  a 0–10 safety rubric and maps to [0, 1]
- `RobustnessEvaluator` — `evaluator_mode: EvaluatorMode | None` and
  `llm_judge: JudgeBase | None` optional params; activates `HybridScorer` when mode
  is set; default `evaluator_mode=None` is fully backward compatible (same scoring
  as v1.5.0 and earlier)

**CLI**
- `python -m toki evaluate --evaluator rule` — default; RULE mode
- `python -m toki evaluate --evaluator hybrid` — HYBRID mode with `MockJudge`
- `python -m toki evaluate --evaluator gguf://path/to/model.gguf` — GGUF backend

**`pyproject.toml`**
- `[gguf]` optional dep group: `llama-cpp-python>=0.2.0`
- Version bumped to `1.6.0`

**`toki.__init__`**
- New exports: `EvaluatorMode`, `GGUFEvaluator`, `HybridScorer`, `RuleScorer`,
  `ScoredResult`

**Tests**
- 53 new tests: `test_hybrid_scorer.py` (27), `test_gguf_evaluator.py` (15),
  `test_evaluator_extended.py` (11), `test_main.py` (5 new CLI tests)
- Total: 600/600 passing (547 prior + 53 new)

---

## [1.4.0] — 2026-06-14

### Added — Phase 14 (Remediation Engine + Real LLM Judge Backends + Custom Attack Library)

**`toki.remediation` — structured fix guidance for adversarial findings**
- `Severity` — class constants: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW` with `rank()` ordering
- `RemediationItem` — frozen dataclass: `prompt_hash`, `attack_category`, `owasp_tag`,
  `nist_measure`, `mitre_tactic`, `severity`, `fix_effort_hours`, `system_prompt_patch`,
  `example_safe_response`, `example_unsafe_response`, `overall_score`
- `RemediationReport` — `items`, `total_assessed`, `total_remediated`, `by_severity`,
  `estimated_total_hours`; `to_json()` / `to_markdown()` / `to_html()`
- `RemediationEngine.generate(verdicts, category_map=None)` — only processes
  `adversarial_success=True` verdicts; category resolution order:
  `category_map[prompt_hash]` → `verdict.metadata["category"]` → keyword inference;
  items sorted by severity (critical first); never raises on malformed verdicts
- Taxonomy covers: `jailbreak` (OWASP-LLM01 / NIST Measure 2.5 / MITRE AML.T0054),
  `injection` (LLM01 / Measure 2.6 / AML.T0054.002), `edge_case` (LLM06 / Measure 2.2 /
  AML.T0048), `boundary` (LLM04 / Measure 2.2 / AML.T0048), `indirect` (LLM01 /
  Measure 2.6 / AML.T0054.002), `agentic` (LLM08 / Measure 2.7 / AML.T0051)

**`toki.judge` (extension) — real LLM judge backends**
- `OllamaJudge(JudgeBase)` — optional dep `httpx`; hits local Ollama `/api/chat`;
  raises `ImportError("OllamaJudge requires httpx: pip install httpx")` when absent
- `AnthropicJudge(JudgeBase)` — optional dep `anthropic` SDK; default model
  `claude-haiku-4-5-20251001`; lazy client init; `api_key` kwarg overrides env var
- `OpenAIJudge(JudgeBase)` — optional dep `openai` SDK; default model `gpt-4o-mini`;
  graceful fallback when `choices` is empty
- `_build_rubric_prompt(prompt, response, criteria)` — shared structured rubric builder
- `_parse_scores(content, criteria, label)` — shared JSON parser; falls back to 0.5
  per criterion on any parse error; clamps to [0.0, 1.0]
- `JudgeFactory.create(name, config, **kwargs)` — registry: `"mock" | "ollama" |
  "anthropic" | "openai"`; raises `ValueError` on unknown name

**`toki.attack_library` — user-defined adversarial test cases**
- `VALID_CATEGORIES` — frozenset of accepted category strings
- `CustomAttack` dataclass — `text`, `category`, `language`, `expected_refusal`,
  `provenance`, `notes`, `id` (SHA-256[:16] of text, auto-computed), `created`
  (ISO-8601 UTC, auto-set); raises `ValueError` on empty text or invalid category
- `AttackLibrary(path)` — JSON-backed persistent store; `add()` deduplicates on `id`,
  returns `False` on duplicate; `remove(id)` / `get(id)` / `list_attacks(category=None)`
  sorted by `created`; `stats()` → `{total, by_category}`; mutations immediately persist

**CLI**
- `python -m toki remediate [--dataset] [--count] [--seed] [--judge] [--threshold]
  [--format json|html|markdown] [--output]`
- `python -m toki attack-add --text TEXT --category CAT [--language] [--allow]
  [--provenance] [--notes] [--library PATH]`
- `python -m toki attack-list [--category] [--library PATH] [--json]`

**Server**
- `POST /api/remediate` — runs MockJudge on fresh prompts, returns `RemediationReport`
- `GET  /api/attacks/custom` — list all custom attacks with stats
- `POST /api/attacks/custom` — add a custom attack; returns `{added, id, duplicate, total}`

**Tests**
- 78 new tests: `test_remediation.py` (27), `test_judge_factory.py` (21),
  `test_attack_library.py` (22), `test_main.py` (8 new CLI tests)
- Total: 520/520 passing (442 prior + 78 new)

**pyproject.toml + `toki.__init__`**
- Version bumped to `1.4.0`
- New exports: `OllamaJudge`, `AnthropicJudge`, `OpenAIJudge`, `JudgeFactory`,
  `RemediationEngine`, `RemediationItem`, `RemediationReport`, `Severity`,
  `AttackLibrary`, `CustomAttack`, `VALID_CATEGORIES`

---

## [1.3.0] — 2026-05-19

### Added — Phase 13 (Indirect Injection + MCP Attack Battery)

**`toki.indirect` — indirect prompt injection simulator**
- `InjectionScenario` enum: DOCUMENT, WEBPAGE, TOOL_RESPONSE, EMAIL
- `IndirectPrompt` frozen dataclass with `scenario`, `user_task`,
  `injected_text`, `target_action`, `owasp_category`, `seed`
- `InjectionVerdict` frozen dataclass; `to_dict()` serialization
- `IndirectInjectionGenerator` — 20 deterministic cases (5 per scenario);
  SHA-256 seeded, stable across runs; implements InjecAgent ACL 2024 taxonomy
- `IndirectInjectionEvaluator` — heuristic offline scoring;
  `evaluate()`, `evaluate_batch(cases, response_fn)`, `summary(verdicts)`
- All cases tagged `OWASP-LLM01:2025`
- `python -m toki indirect [--scenario all|document|webpage|tool_response|email] [--json]`

**`toki.agentic` — agentic + MCP attack battery**
- `AgentAttackType` enum (9 values):
  - AgentLAB five-class taxonomy: INTENT_HIJACKING, TOOL_CHAINING,
    TASK_INJECTION, OBJECTIVE_DRIFTING, MEMORY_POISONING
  - MCP primitives (Breaking the Protocol, arXiv 2601.17549):
    MCP_RUG_PULL, MCP_TOOL_SHADOWING, MCP_REGISTRY_POISONING
  - TOOL_METADATA_POISONING (ToolTweak / ToolHijacker)
- `OWASP_MAPPING` dict — per-type OWASP LLM Top 10 2025 tags
  (LLM01 injection, LLM03 supply chain, LLM06 excessive agency)
- `AgentAttackScenario` frozen dataclass with full scenario fields
- `AgentVerdict` frozen dataclass; `to_dict()` serialization
- `AgentAttackBattery` — 36 deterministic cases (4 per type); SHA-256 seeded
- `AgentAttackEvaluator` — heuristic offline scoring;
  `evaluate()`, `evaluate_batch(scenarios, response_fn)`, `summary(verdicts)`
- `python -m toki agentic [--type all|<attack_type>] [--json]`

**`toki.multilingual` (extension) — chat-template role-boundary injection**
- `ChatInjectGenerator` — 20 deterministic cases targeting system/user/
  assistant role token boundaries; `encoding="chat_template"`, all SHA-256
  seeded (ChatInject arXiv 2509.22830)
- `generate_chat_inject_battery()` module-level convenience

**`toki.coverage` (extension)**
- `ENCODING_AXIS` extended with `"chat_template"` — the 20 new cases are
  automatically tracked on the encoding axis

**Tests**
- 54 new tests: `test_indirect.py` (25), `test_agentic.py` (17),
  `test_chat_inject.py` (12)
- Total: 409/409 passing (355 prior + 54 new)

**pyproject.toml + `toki.__init__`**
- Version bumped to `1.3.0`
- New exports: `OWASP_LLM01`, `InjectionScenario`, `IndirectInjectionGenerator`,
  `IndirectInjectionEvaluator`, `IndirectPrompt`, `InjectionVerdict`,
  `OWASP_MAPPING`, `AgentAttackType`, `AgentAttackBattery`,
  `AgentAttackEvaluator`, `AgentAttackScenario`, `AgentVerdict`,
  `ChatInjectGenerator`, `generate_chat_inject_battery`

---

## [1.1.0] — 2026-05-12

### Added — Phase 11 (P1 Roadmap Sprint)

All four P1 items from the researched feature roadmap, shipped end-to-end.

**`toki.coverage` — coverage map + blind-spot dashboard**
- `CoverageMap` dataclass: total, per-axis `counts`, `shares` (normalised within axis), `blind_threshold`, `blind_spots`, pre-computed `radar_points` and `radar_polygon` SVG-ready string
- Four axes — category (7 buckets), severity (4), language (5), encoding (4) — with safe fallbacks when prompts don't carry explicit metadata (severity inferred from category, language from cheap heuristic words, encoding from zero-width-Unicode detection)
- `compute_coverage(source, blind_threshold=0.05)` works on `AdversarialDataset` or any iterable of prompts; `label_positions(...)` pushes axis labels outward from each vertex for SVG rendering
- `python -m toki coverage [--include-multilingual] [--json]` CLI surface
- `GET /api/coverage` + `POST /api/coverage` (real toki underneath; includes the 50-case multilingual battery by default)

**`toki.regression` — safety regression CI gate**
- `Baseline` dataclass: overall + per-category pass rates with ISO-8601 created timestamp, free-form meta dict, schema-versioned; `save(path)` / `load(path)`
- `Baseline.from_summary(...)` reads canonical `RobustnessEvaluator.summary()` shape (`mean_score` + `by_category`) or a flat dict
- `CategoryDelta` + `RegressionReport` — `regressed` / `improved` / `unchanged` / `missing_from_current` / `new_in_current` / `worst_delta` with `failed` flag and `exit_code()`
- `compare(baseline, current_summary, tolerance=0.02)` — also available as `RegressionReport.compare(...)` static method
- Markdown report generator (`to_markdown()`) + JSON serializer (`as_dict()`)
- `python -m toki ci-baseline --output ...` + `python -m toki ci-check --baseline ... [--report ...] [--json]` (non-zero exit on regression)
- `POST /api/ci/baseline` (accepts optional `output_path`) + `POST /api/ci/check` (accepts inline `baseline` payload or `baseline_path`)
- **GitHub Action** at `.github/actions/safety-gate/action.yml` — composite action with inputs `baseline / tolerance / seed / size / python-version / toki-ref / report-path`, outputs `failed / overall-delta / regressed-count`; pip-installs toki from the configured ref, runs `toki ci-check --json`, parses the result into job outputs, and writes the Markdown report to `$GITHUB_STEP_SUMMARY`

**`toki.consistency` — evaluator consistency scoring**
- `fleiss_kappa(rows, n_categories=2)` — pure-stdlib Fleiss' kappa for N subjects × n raters; handles unanimous-column edge case (returns 1.0 instead of dividing by zero); validates non-empty / equal-width / equal-rater-count rows
- Four built-in judge configurations: `strict` (refusal AND no harm AND no leak), `lenient` (no harm), `refusal` (refusal flag only), `leak` (no system-prompt leak only) — each derived from the real `RobustnessEvaluator` output
- `ConsistencyEvaluator(model_fn, judges=JUDGE_NAMES, threshold=0.6)` runs each prompt through every judge with the same evaluator output, computes per-prompt agreement coefficient and overall Fleiss' kappa, flags entries with κ below threshold as `unreliable`
- `ConsistencyReport` carries per-prompt `ConsistencyEntry`s plus a full pairwise agreement matrix
- `python -m toki consistency [--json]` and `POST /api/consistency`

**`toki.multilingual` — 50-case multilingual + encoding attack battery**
- `MultilingualPrompt` extends `AdversarialPrompt` with `language`, `encoding`, `severity` fields — picked up automatically by the coverage map
- `encode_base64`, `encode_rot13`, `encode_zwsp` — three encoders that wrap payloads with realistic "please decode and execute" framings (base64/ROT13) or weave zero-width characters between visible glyphs (ZWSP)
- Payload banks: 12 English (shared by base64/ROT13/ZWSP), 6 Spanish, 6 French, 6 German — each non-English variant uses native phrasing (e.g. "Ignora todas las instrucciones anteriores...")
- `MultilingualGenerator.generate_all()` returns the canonical 50-case battery in stable order; `generate(languages=..., encodings=...)` returns filtered subsets; `generate_battery()` module-level convenience
- Every prompt is deterministic (SHA-256 of `(encoding, language, index, payload)`) and seeds are guaranteed unique across the battery

**Demo UI (`demo/ranking.html`)**
- New "Coverage map" section above the leaderboard: SVG radar/spider chart driven by `/api/coverage`, dotted axis vertex markers, labels for categories + non-plain encodings + non-English languages, live blind-spot sidebar with red highlight per gap
- Header consistency pill shows mean Fleiss' κ live; "all judges agree" / "N/M unreliable"
- Both auto-refresh on page load when the server is up

**Tests**
- 60 new tests across `test_coverage.py` (12), `test_regression.py` (12), `test_consistency.py` (17), `test_multilingual.py` (19)
- Total: 311/311 passing (251 prior + 60 new)

**pyproject.toml + `toki.__init__`**
- Version bumped to `1.1.0`
- New exports: `CoverageMap`, `compute_coverage`, `label_positions`, `CATEGORY_AXIS`, `SEVERITY_AXIS`, `LANGUAGE_AXIS`, `ENCODING_AXIS`, `Baseline`, `CategoryDelta`, `RegressionReport`, `compare_regression`, `JUDGE_NAMES`, `ConsistencyEntry`, `ConsistencyEvaluator`, `ConsistencyReport`, `fleiss_kappa`, `MultilingualGenerator`, `MultilingualPrompt`, `encode_base64`, `encode_rot13`, `encode_zwsp`, `generate_battery`
- **PLAN.md** gains a "Researched Feature Roadmap" section listing P1 (now shipped), P2 (4 next-up items), and P3 (3 strategic items)

---

## [0.10.0] — 2026-05-09

### Added

**Persistent SQLite-backed leaderboard (T3)**
- `toki.leaderboard` — new module with `LeaderboardEntry(model_name, suite, pass_rate, robustness_score, timestamp, notes, id)` and `Leaderboard(db_path)` class. SQLite schema auto-created on first use; pure stdlib (`sqlite3`).
  - `record(entry)` / `record_many(entries)` — append-only inserts; rejects out-of-range scores at the boundary, NaN, empty model_name/suite.
  - `top_n(suite, n=10)` — ordered by `robustness_score DESC, timestamp DESC`. Pass `"all"` to drop the suite filter.
  - `history(model_name, suite=None)` — chronological per-model history.
  - `compare(model_a, model_b)` — latest-per-suite side-by-side; mean-of-overlapping-suites tie-break declares a winner.
  - `KNOWN_SUITES = ("adversarial", "paraphrase", "noise")` — public contract for API/UI tabs; `load_seed()` helper for bulk JSON ingest.
- `demo/server.py` — three new endpoints: `POST /api/leaderboard` (record), `GET /api/leaderboard/{suite}` (top 10; suite ∈ adversarial|paraphrase|noise|all), `GET /api/leaderboard/model/{name}` (history). DB lazy-initialised at `demo/leaderboard.db` (gitignored), auto-seeded from `demo/seed_leaderboard.json` on first request.
- `demo/leaderboard.html` — live leaderboard page at `/leaderboard.html`: suite filter tabs, 10-second auto-refresh, robustness colour-coding (green ≥0.85, yellow ≥0.70, red <0.70), row-flash on new entries, offline indicator.
- `demo/seed_leaderboard.json` — 8 realistic entries across 4 model names (phi-3-mini-4k, qwen-2.5-1.5b, llama-3.2-3b, gemma-2-2b) and all 3 suites.

### Changed (breaking)

**Phase-7 leaderboard renamed → ranking (frees the leaderboard namespace)**
- `toki.leaderboard` (Bonferroni-corrected multi-model ranker, Phase 7) → `toki.ranking`. Class renames: `Leaderboard` → `Ranking`, `LeaderboardConfig` → `RankingConfig`, `LeaderboardEntry` → `RankingEntry`, `LeaderboardResult` → `RankingResult`. Save artefact `leaderboard.json` → `ranking.json`. Default `--output-dir experiments/leaderboards` → `experiments/rankings`.
- CLI subcommand `python -m toki leaderboard` → `python -m toki rank`.
- `toki.__init__` exports updated; old names removed (no compatibility shim — pre-1.0).
- Rationale: the Phase-7 module is a *ranking* operation (one-shot, k-model, stat-tested). T3 introduces a fundamentally different concept — persistent time-series score tracking — that is naturally called a leaderboard. Two `Leaderboard` classes would have been confusing.

### Tests
- 10 new Python tests in `python/tests/test_leaderboard.py` — schema auto-create + empty reads, record() round-trip + auto-id, top_n sort + cap, top_n suite filter, top_n("all") global ranking, chronological history per model, compare() latest-per-suite + winner, score range validation, load_seed bulk insert, persistence across instances + KNOWN_SUITES contract.
- `test_main.py` CLI tests renamed (`leaderboard` → `rank`) and assert the new `ranking.json` artefact path.
- `test_ranking.py` — Phase-7's 20 tests carried over verbatim under the new module name; all still pass.

### Version
- `__version__` and `pyproject.toml` bumped to `0.10.0`.

---

## [0.7.0] — 2026-05-05

### Added

**Python package — multi-model adversarial leaderboard**
- `toki.leaderboard` — pure-stdlib leaderboard module wired to the real `RobustnessEvaluator` and paired statistical tests with Bonferroni correction:
  - `LeaderboardEntry(name, mean_score, n_comparisons, wins, losses, ties, rank, significant)` — per-model ranking record; `significant=True` when the model has ≥1 win and every win is statistically significant at the Bonferroni-corrected threshold
  - `PairResult` — outcome of a single head-to-head comparison: winner name (or `"tie"`), t-statistic, t-p-value, W-statistic, W-p-value, `alpha_bonferroni`
  - `LeaderboardConfig` — `name`, `seed`, jailbreak/injection/boundary counts, nominal `alpha`, `output_dir`
  - `LeaderboardResult` — full outcome: `entries` (rank-ordered), `pairs` (all k*(k-1)/2 head-to-heads), `alpha_bonferroni`, `n_models`, `n_pairs`; `save()` raises `FileExistsError` if file already exists (no-overwrite, consistent with `ExperimentResult`); `load()` rehydrates typed `LeaderboardEntry` + `PairResult`; `format_table()` returns a clean ASCII ranked table
  - `_bonferroni_alpha(α, n_pairs)` — α / n_pairs; returns nominal α unchanged when n_pairs == 0
  - `_compare_pair(scores_a, scores_b, alpha_bonf)` — runs `paired_t_test` + `wilcoxon_test` at the Bonferroni-corrected threshold; declares a winner only when at least one test rejects H0
  - `_rank_entries(all_scores, pairs)` — sorts by descending mean score (alphabetical tie-break for determinism); models with equal mean share the same rank number; n_comparisons = wins + losses + ties
  - `Leaderboard(models, config)` — validates ≥2 models and unique names; `run(save=False)` generates one shared adversarial dataset → evaluates all models with the real `RobustnessEvaluator` → runs all k*(k-1)/2 pairs at α_bonferroni → ranks by mean score → returns `LeaderboardResult`
  - `_all_baseline_specs()` — convenience factory returning all three built-in `ModelSpec`s (safe, unsafe, mixed) from `toki.compare.BASELINES`
- `python -m toki leaderboard` CLI subcommand — `--models` (optional list of built-in baseline names; defaults to all three), `--name`, `--seed`, `--alpha`, `--jailbreak-count`, `--injection-count`, `--boundary-count`, `--output-dir`, `--save`; prints ASCII ranked table including Bonferroni-corrected α, per-model W/L/T tallies, and significance markers
- `toki.__init__` exports `Leaderboard`, `LeaderboardConfig`, `LeaderboardEntry`, `LeaderboardResult`; version bumped to `0.7.0`

**Tests**
- 23 new Python tests:
  - `test_leaderboard.py` (20) — Bonferroni formula (k=3, k=4, degenerate), `LeaderboardEntry` construction, ≥2-model guard, unique-name guard, three-model structure, safe-outranks-unsafe, corrected-α in result, `PairResult` winner semantics, `_rank_entries` ordering, shared rank on equal mean, save/load round-trip, no-overwrite guard, `format_table` completeness, `_all_baseline_specs`, `_compare_pair` corrected-alpha check, wins/losses zero-sum invariant, four-model n_pairs=6, config snapshot
  - `test_main.py` (3) — leaderboard CLI happy path, `--save` artifact persistence, unknown-model rejection
- Total: 123/123 Python tests passing

**pyproject.toml**
- Version bumped to `0.7.0`

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
