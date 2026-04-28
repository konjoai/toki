# Toki — Project Roadmap

> **Toki** (陶器) — ceramic, shaped under pressure.  
> Adversarial fine-tuning lab for small language models.  
> "Break the model. Fix the model. Prove it."

Current version: **v0.2.0**

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

## Phase 3 — Benchmark Suite & Statistical Reporting (v0.3.0)

**Goal:** Rigorous before/after robustness reports with statistical significance.

- [ ] `toki.benchmark` module: paired t-test / Wilcoxon on score distributions
- [ ] HTML + JSON report generation per experiment run
- [ ] p50/p95/p99 latency distribution for model inference
- [ ] `pixi run benchmark` or `make benchmark` target
- [ ] Bump to v0.3.0

---

## Phase 4 — Dataset Publishing (v0.4.0)

**Goal:** Push adversarial datasets to HuggingFace Hub with versioning.

- [ ] `toki.hub` module: upload dataset to HF Hub via `datasets` library
- [ ] Dataset card generation (auto-filled from `AdversarialDataset.stats()`)
- [ ] Version tagging in dataset metadata
- [ ] `toki upload --dataset path/to/dataset.json --repo user/toki-adversarial` CLI
- [ ] Bump to v0.4.0

---

## Phase 5 — Continuous Hardening (v0.5.0)

**Goal:** Iterative generate-finetune-evaluate loop with convergence criteria.

- [ ] `toki pipeline --iterations N --convergence-threshold 0.95` end-to-end loop
- [ ] Convergence: stop when mean robustness score >= threshold across 3 consecutive rounds
- [ ] Experiment reproducibility: full config + seed saved per run
- [ ] Bump to v0.5.0

---

## Future / Backlog

- GGUF/GGML quantized model support (llama.cpp backend)
- Multi-model adversarial comparison (model A vs model B)
- Web UI for interactive prompt generation and scoring
- Mojo-accelerated tokenization for high-throughput batch evaluation

---

*Last updated: 2026-04-28 — v0.2.0 shipped.*
