# Unit Test Code Quality (3-Class) — LLM Baselines

This repo benchmarks locally deployable LLMs for **3-class unit-test code quality classification**.

## Label Mapping (3 classes)
We remap original scores into 3 classes:
- Class 1: score 1 or 2
- Class 2: score 3, or score 4 with **no docstring**
- Class 3: score 4 with a **docstring**, or score 5

## Data
Validation files (JSONL):
- `val_code.jsonl` : **code-only**
- `val_code_metrics.jsonl` : **code + metrics**

Each line contains:
- `instruction` (fixed prompt)
- `input` (Code: ... [+ Metrics: ...])
- `output` (gold label: "1"/"2"/"3")

## Models evaluated
- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `bigcode/starcoder2-7b`
- `codellama/CodeLlama-7b-Instruct-hf`

## Setup
```bash
pip install -U torch transformers accelerate scikit-learn bitsandbytes
