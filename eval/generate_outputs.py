"""
Generate responses from base, indirect, and direct models on AlpacaEval and IFEval prompts.

Outputs (saved to --output-dir):
    alpaca_base.jsonl       AlpacaEval prompts + base model responses
    alpaca_indirect.jsonl   AlpacaEval prompts + indirect GRPO responses
    alpaca_direct.jsonl     AlpacaEval prompts + direct GRPO responses
    ifeval_base.jsonl       IFEval prompts + base model responses
    ifeval_indirect.jsonl   IFEval prompts + indirect GRPO responses
    ifeval_direct.jsonl     IFEval prompts + direct GRPO responses

Usage:
    python eval/generate_outputs.py \
        --indirect-adapter /content/drive/MyDrive/cs234/grpo_indirect \
        --direct-adapter   /content/drive/MyDrive/cs234/grpo_direct \
        --output-dir       ./eval_outputs
"""

import argparse
import json
import os
import gc
from pathlib import Path

import torch
import openai
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_MODEL      = "Qwen/Qwen2.5-7B-Instruct"
POLICY_API      = "Qwen/Qwen2.5-7B-Instruct-Turbo"
MAX_NEW_TOKENS  = 512
BATCH_SIZE      = 8


# ── Together API client ───────────────────────────────────────────────────

def make_together_client():
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        raise SystemExit("ERROR: TOGETHER_API_KEY not set")
    return openai.OpenAI(api_key=key, base_url="https://api.together.xyz/v1")


def generate_base_api(prompts: list[str], client, workers: int = 16) -> list[str]:
    """Generate from base model via Together API (parallel)."""
    responses = [""] * len(prompts)

    def call(i, prompt):
        resp = client.chat.completions.create(
            model=POLICY_API,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
        )
        return i, resp.choices[0].message.content

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, i, p): i for i, p in enumerate(prompts)}
        done = 0
        for f in as_completed(futures):
            i, text = f.result()
            responses[i] = text
            done += 1
            if done % 50 == 0 or done == len(prompts):
                print(f"  API: {done}/{len(prompts)}", flush=True)

    return responses


# ── Local LoRA inference ──────────────────────────────────────────────────

def load_lora_model(adapter_path: str):
    """Load Qwen 7B + LoRA adapter. Returns (model, tokenizer)."""
    print(f"  Loading base model + adapter from {adapter_path}...")
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate_local(prompts: list[str], model, tokenizer) -> list[str]:
    """Batched generation from a local model."""
    device   = next(model.parameters()).device
    responses = []

    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start : start + BATCH_SIZE]
        messages_batch = [[{"role": "user", "content": p}] for p in batch]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, seq in enumerate(out):
            prompt_len = inputs.input_ids.shape[1]
            gen = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            responses.append(gen.strip())

        done = min(start + BATCH_SIZE, len(prompts))
        print(f"  Local: {done}/{len(prompts)}", flush=True)

    return responses


# ── Dataset loaders ───────────────────────────────────────────────────────

def load_alpacaeval(n: int) -> tuple[list[str], list[str]]:
    """
    Returns (instructions, gpt4_outputs).
    Tries multiple loading strategies since tatsu-lab/alpaca_eval uses a
    custom dataset script that newer `datasets` versions block.
    """
    print("Loading AlpacaEval dataset...")

    # Strategy 1: huggingface_hub direct parquet download (no script needed)
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_tree
        repo_files = [
            f.path for f in list_repo_tree(
                "tatsu-lab/alpaca_eval", repo_type="dataset", recursive=True
            )
            if f.path.endswith(".parquet")
        ]
        if repo_files:
            local = hf_hub_download(
                "tatsu-lab/alpaca_eval", repo_files[0], repo_type="dataset"
            )
            df = pd.read_parquet(local).head(n)
            if "instruction" in df.columns and "output" in df.columns:
                print(f"  Loaded {len(df)} prompts via parquet")
                return df["instruction"].tolist(), df["output"].tolist()
    except Exception as e:
        print(f"  parquet strategy failed: {e}")

    # Strategy 2: HuggingFace Datasets Server REST API (paginated)
    try:
        import requests
        instructions, gpt4_outputs = [], []
        page = 100
        for offset in range(0, n, page):
            resp = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": "tatsu-lab/alpaca_eval",
                    "config":  "alpaca_eval_gpt4_baseline",
                    "split":   "eval",
                    "offset":  offset,
                    "limit":   min(page, n - offset),
                },
                timeout=30,
            ).json()
            rows = resp.get("rows", [])
            if not rows:
                break
            instructions.extend(r["row"]["instruction"] for r in rows)
            gpt4_outputs.extend(r["row"]["output"]      for r in rows)
        if instructions:
            print(f"  Loaded {len(instructions)} prompts via HF Datasets API")
            return instructions[:n], gpt4_outputs[:n]
    except Exception as e:
        print(f"  HF API strategy failed: {e}")

    raise RuntimeError(
        "Could not load AlpacaEval dataset. "
        "Check internet access and huggingface_hub installation."
    )


def load_ifeval(n: int) -> list[dict]:
    """Returns list of IFEval rows (key, prompt, instruction_id_list, kwargs)."""
    print("Loading IFEval dataset...")
    ds = load_dataset("google/IFEval", split="train")  # capital IFEval
    ds = ds.select(range(min(n, len(ds))))
    rows = [dict(row) for row in ds]
    print(f"  Loaded {len(rows)} IFEval prompts")
    return rows


# ── Writers ───────────────────────────────────────────────────────────────

def save_alpaca(path: str, instructions: list[str], gpt4_outputs: list[str], responses: list[str]):
    with open(path, "w") as f:
        for inst, gpt4, resp in zip(instructions, gpt4_outputs, responses):
            f.write(json.dumps({
                "instruction": inst,
                "gpt4_output": gpt4,
                "response": resp,
            }) + "\n")
    print(f"  Saved {len(responses)} rows → {path}")


def save_ifeval(path: str, rows: list[dict], responses: list[str]):
    with open(path, "w") as f:
        for row, resp in zip(rows, responses):
            out = dict(row)
            out["response"] = resp
            f.write(json.dumps(out) + "\n")
    print(f"  Saved {len(responses)} rows → {path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indirect-adapter", required=True)
    parser.add_argument("--direct-adapter",   required=True)
    parser.add_argument("--output-dir",        default="./eval_outputs")
    parser.add_argument("--n-alpaca",  type=int, default=805)
    parser.add_argument("--n-ifeval",  type=int, default=541)
    parser.add_argument("--api-workers", type=int, default=16)
    parser.add_argument("--skip-base",     action="store_true", help="Skip base model generation")
    parser.add_argument("--skip-indirect", action="store_true")
    parser.add_argument("--skip-direct",   action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    alpaca_instructions, gpt4_outputs = load_alpacaeval(args.n_alpaca)
    ifeval_rows = load_ifeval(args.n_ifeval)
    alpaca_prompts = alpaca_instructions
    ifeval_prompts = [r["prompt"] for r in ifeval_rows]

    # ── Base model (API) ──────────────────────────────────────────────────
    if not args.skip_base:
        client = make_together_client()
        print("\n[1/3] Generating base model outputs (Together API)...")
        alpaca_base = generate_base_api(alpaca_prompts, client, args.api_workers)
        ifeval_base = generate_base_api(ifeval_prompts, client, args.api_workers)
        save_alpaca(f"{args.output_dir}/alpaca_base.jsonl",   alpaca_instructions, gpt4_outputs, alpaca_base)
        save_ifeval(f"{args.output_dir}/ifeval_base.jsonl",   ifeval_rows, ifeval_base)
        del alpaca_base, ifeval_base
    else:
        print("\n[1/3] Skipping base model (--skip-base)")

    # ── Indirect model (local LoRA) ───────────────────────────────────────
    if not args.skip_indirect:
        print("\n[2/3] Generating indirect GRPO outputs (local)...")
        model, tok = load_lora_model(args.indirect_adapter)
        alpaca_ind = generate_local(alpaca_prompts, model, tok)
        ifeval_ind = generate_local(ifeval_prompts, model, tok)
        save_alpaca(f"{args.output_dir}/alpaca_indirect.jsonl", alpaca_instructions, gpt4_outputs, alpaca_ind)
        save_ifeval(f"{args.output_dir}/ifeval_indirect.jsonl",  ifeval_rows, ifeval_ind)
        del model, tok, alpaca_ind, ifeval_ind
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[2/3] Skipping indirect model (--skip-indirect)")

    # ── Direct model (local LoRA) ─────────────────────────────────────────
    if not args.skip_direct:
        print("\n[3/3] Generating direct GRPO outputs (local)...")
        model, tok = load_lora_model(args.direct_adapter)
        alpaca_dir = generate_local(alpaca_prompts, model, tok)
        ifeval_dir = generate_local(ifeval_prompts, model, tok)
        save_alpaca(f"{args.output_dir}/alpaca_direct.jsonl", alpaca_instructions, gpt4_outputs, alpaca_dir)
        save_ifeval(f"{args.output_dir}/ifeval_direct.jsonl",  ifeval_rows, ifeval_dir)
        del model, tok, alpaca_dir, ifeval_dir
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[3/3] Skipping direct model (--skip-direct)")

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
