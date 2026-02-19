# -*- coding: utf-8 -*-
import argparse
import json
import re
from typing import Dict, List, Tuple
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

r"""
usage examples:

# 1) normal 4-bit (fast, GPU only if it fits)
python .\prompt_only_eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct --data .\val_code.jsonl --load_in_4bit --trust_remote_code

# 2) DeepSeek-Coder-V2-Lite-Instruct on 8GB GPU: recommended 8-bit + CPU offload
python .\prompt_only_eval1.py --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --data .\val_code.jsonl --load_in_8bit --cpu_offload --trust_remote_code

# 3) run code+metrics
python .\prompt_only_eval.py --model microsoft/Phi-3.5-mini-instruct --data .\val_code_metrics.jsonl --load_in_4bit --trust_remote_code
"""

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

LABEL_RE = re.compile(r"[123]")


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_prompt(tokenizer, instruction: str, user_input: str) -> str:
    """
    Prefer chat template if available; otherwise fallback to plain prompt.
    Force the model to output ONLY 1/2/3.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"{instruction}\n\n{user_input}\n\nOutput ONLY one label: 1 or 2 or 3.",
            },
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return (
        f"{instruction}\n\n"
        f"{user_input}\n\n"
        "Output ONLY one label: 1 or 2 or 3.\n"
        "Label:"
    )


def parse_label(text: str) -> int:
    m = LABEL_RE.search(text)
    if not m:
        return 0
    return int(m.group(0))


@torch.inference_mode()
def predict_one(model, tokenizer, prompt: str, max_new_tokens: int = 3) -> Tuple[int, str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,  # reduce VRAM peak
    )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    label = parse_label(gen_text)
    return label, gen_text


def build_quant_config(args):
    if not args.load_in_4bit and not args.load_in_8bit:
        return None

    if not _HAS_BNB:
        raise RuntimeError("BitsAndBytesConfig not available. Please install bitsandbytes.")

    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # 8-bit
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=bool(args.cpu_offload),
    )


def load_model(args, quant_config):
    """
    For 8GB GPU + 16GB RAM:
    - 4bit: works only if the whole quantized model fits GPU
    - 8bit + cpu_offload: can run larger models (slower)
    """
    # You have RTX 4060 8GB + 16GB RAM -> use conservative caps
    max_memory = {
        0: args.gpu_max_memory,   # e.g. "7600MiB"
        "cpu": args.cpu_max_memory,  # e.g. "12GiB"
    }

    print(f"Loading model: {args.model} (4bit={args.load_in_4bit}, 8bit={args.load_in_8bit}, cpu_offload={args.cpu_offload})")
    print(f"max_memory={max_memory}")

    kwargs = dict(
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        max_memory=max_memory,
    )

    # When CPU offload may happen, provide offload folder for safety
    if args.cpu_offload or args.load_in_8bit:
        kwargs.update(
            offload_folder=args.offload_folder,
            offload_state_dict=True,
        )

    return AutoModelForCausalLM.from_pretrained(args.model, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--data", required=True, help="val_code.jsonl or val_code_metrics.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit; otherwise evaluate first N samples")

    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization (fast if fits GPU)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU offload (recommended for large models on 8GB GPU)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable if model requires it")

    # For your machine: RTX 4060 8GB + 16GB RAM
    parser.add_argument("--gpu_max_memory", default="7600MiB", help="GPU memory cap for accelerate device_map")
    parser.add_argument("--cpu_max_memory", default="12GiB", help="CPU RAM cap for offload")
    parser.add_argument("--offload_folder", default="offload", help="Folder used for disk offload when needed")

    args = parser.parse_args()

    if (args.load_in_4bit or args.load_in_8bit) and not _HAS_BNB:
        raise RuntimeError("You requested quantization but bitsandbytes is not available. Install bitsandbytes.")

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one: --load_in_4bit OR --load_in_8bit")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    tokenizer.truncation_side = "left"

    quant_config = build_quant_config(args)

    model = load_model(args, quant_config)
    model.eval()

    data = load_jsonl(args.data)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    y_true, y_pred = [], []
    invalid = 0

    for i, ex in enumerate(data, 1):
        instruction = ex["instruction"]
        user_input = ex["input"]
        true_label = int(str(ex["output"]).strip())

        prompt = build_prompt(tokenizer, instruction, user_input)
        pred_label, raw = predict_one(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)

        if pred_label == 0:
            invalid += 1
            # keep metrics computable; count invalid separately
            pred_label = 1

        y_true.append(true_label)
        y_pred.append(pred_label)

        if i <= 3:
            print(f"\n[Sample {i}] true={true_label} pred={pred_label} raw_output={raw!r}")

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])

    print("\n========== RESULTS ==========")
    print(f"Model: {args.model}")
    print(f"Data : {args.data}")
    print(f"N    : {len(y_true)}  (invalid_outputs={invalid})")
    print(f"Pred dist: {dict(sorted(Counter(y_pred).items()))}")
    print(f"ACC  : {acc:.4f}")
    print(f"Macro-F1: {mf1:.4f}")
    print("Confusion Matrix (rows=true, cols=pred) labels=[1,2,3]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=[1, 2, 3], digits=4, zero_division=0))


if __name__ == "__main__":
    main()
