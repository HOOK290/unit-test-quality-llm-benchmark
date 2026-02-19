## Unit Test Quality (3-Class) — LLM Baselines

This repo benchmarks locally deployable LLMs for **3-class unit-test code quality classification** (prompt-only / zero-shot).

## Label Mapping (3 classes)
We remap original scores into 3 classes:
- **Class 1**: score **1** or **2**
- **Class 2**: score **3**, or score **4** with **no docstring**
- **Class 3**: score **4** with a **docstring**, or score **5**

## Data (JSONL)
Validation files:
- `val_code.jsonl` : **code-only**
- `val_code_metrics.jsonl` : **code + metrics**

Each JSONL line contains:
- `instruction` : fixed prompt
- `input` : `Code: ...` (optionally followed by `Metrics: ...`)
- `output` : gold label in `{"1","2","3"}`

## Models evaluated
- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `bigcode/starcoder2-7b`
- `codellama/CodeLlama-7b-Instruct-hf`

## Setup
```bash
pip install -U torch transformers accelerate scikit-learn bitsandbytes
````

## prompt_only_eval1.py (Prompt-only / Zero-shot Evaluation)

## What it does

`prompt_only_eval1.py` loads a Hugging Face model (optionally in 4-bit/8-bit), runs inference on a JSONL file,
extracts the predicted label in `{1,2,3}`, and reports:

* Accuracy
* Macro-F1
* Confusion Matrix
* Classification report

## Common arguments

* `--model` : HF model name or local path
* `--data` : `val_code.jsonl` or `val_code_metrics.jsonl`
* `--load_in_4bit` : 4-bit quantization (recommended on RTX 4060)
* `--load_in_8bit --cpu_offload` : fallback for larger models (slower)
* `--limit N` : evaluate only first N samples (quick test)

## Examples (code-only)

```bash
python prompt_only_eval1.py --model microsoft/Phi-3.5-mini-instruct --data val_code.jsonl --load_in_4bit --trust_remote_code
python prompt_only_eval1.py --model Qwen/Qwen2.5-Coder-7B-Instruct --data val_code.jsonl --load_in_4bit --trust_remote_code
python prompt_only_eval1.py --model bigcode/starcoder2-7b --data val_code.jsonl --load_in_4bit
python prompt_only_eval1.py --model codellama/CodeLlama-7b-Instruct-hf --data val_code.jsonl --load_in_4bit
```

## Examples (code + metrics)

```bash
python prompt_only_eval1.py --model microsoft/Phi-3.5-mini-instruct --data val_code_metrics.jsonl --load_in_4bit --trust_remote_code
python prompt_only_eval1.py --model Qwen/Qwen2.5-Coder-7B-Instruct --data val_code_metrics.jsonl --load_in_4bit --trust_remote_code
python prompt_only_eval1.py --model bigcode/starcoder2-7b --data val_code_metrics.jsonl --load_in_4bit
python prompt_only_eval1.py --model codellama/CodeLlama-7b-Instruct-hf --data val_code_metrics.jsonl --load_in_4bit
```


