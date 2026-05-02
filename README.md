# 🦀 Tōki

![Language](https://img.shields.io/badge/language-rust-blue) ![ML](https://img.shields.io/badge/core-python-yellow) ![License](https://img.shields.io/badge/license-busl--1.1-green) ![Status](https://img.shields.io/badge/status-active-brightgreen)

> 🔥 Adversarial fine-tuning lab for small LLMs (1B–3B). Break models ⚔️, harden them 🛡️, and measure what actually improves 📊.

---

## 🏺 Meaning

**Tōki (陶器)** — *ceramic, shaped under pressure.*

Models, like clay, only reveal their strength when stress-tested. Tōki is about forcing models through pressure — adversarial inputs — and reshaping them into something more robust.

---

## 🚀 What it is

Tōki is an **end-to-end adversarial ML lab**:

* Generate adversarial prompts (jailbreaks, edge cases, failure modes)
* Fine-tune models using LoRA / QLoRA (MLX or HuggingFace)
* Evaluate robustness before and after training
* Publish:

  * adversarial datasets 📦
  * hardened model weights 🧠
  * evaluation reports 📊

---

## ❗ The problem

LLMs are brittle.

* They fail under adversarial prompts
* They overfit to narrow behaviors
* There’s little systematic research on **small model robustness**

Most teams:

> test a few prompts and call it “safe”

Tōki answers:

> **Do models actually get safer — or just better at passing tests?**

---

## 🧠 What you learn

* Adversarial ML & red-teaming
* LoRA / QLoRA fine-tuning
* Dataset construction & curation
* Robustness evaluation & benchmarking

---

## ⚙️ Architecture

* 🦀 **Rust CLI** — orchestration, experiments, pipelines
* 🐍 **Python core** — training, generation, evaluation

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/toki.git
cd toki
cargo build

# Python core (no ML deps required for generate/evaluate/report/upload --dry-run)
cd python && pip install -e .
python -m toki generate --count 32 --output dataset.json
python -m toki evaluate --dataset dataset.json
python -m toki run --name baseline --output-dir experiments/runs
python -m toki report experiments/runs/<ts>_baseline/result.json --format both

# Continuous hardening loop (stops at convergence)
python -m toki pipeline \
  --name harden_v1 \
  --iterations 10 \
  --convergence-threshold 0.95 \
  --convergence-window 3

# Publish to HuggingFace Hub (requires `pip install -e ".[hf]"`)
python -m toki upload \
  --dataset dataset.json \
  --repo your-username/toki-adversarial-v1 \
  --version 0.4.0
```

---

## 🎯 Vision

> Break the model. Fix the model. Prove it.

If you want next step, I can:
→ unify all 4 under a **Konjo umbrella README + architecture diagram** (that’s what really makes this pop in interviews)
