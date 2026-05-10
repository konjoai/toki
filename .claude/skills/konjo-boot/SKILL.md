---
name: konjo-boot
description: Boot a Konjo session for toki — toki — adversarial fine-tuning lab (generate adversarial datasets, evaluate robustness, LoRA fine-tune small LLMs). Produces a Session Brief, runs Discovery, identifies the next sprint. Use at the start of any work session or when invoked with /konjo.
user-invocable: true
---

# Konjo Boot — toki

## Step 1 — Orient
Read CLAUDE.md, README.md, CHANGELOG.md, PLAN.md in order.

## Step 2 — Session Brief
- Current version and test count
- Last shipped (from CHANGELOG.md)
- Active blockers
- Health: Green / Yellow / Red

## Step 3 — Discovery
- `cargo test` — Rust tests green?
- `python -m pytest tests/ -x` — Python tests green?
- `cargo clippy -- -D warnings` — lint clean?
- `ruff check .` — Python lint clean?

## Step 4 — Plan
Identify the next sprint from PLAN.md and propose the first concrete task.
