"""
Quick head-to-head eval: base Qwen 7B vs Direct GRPO-trained model.

- Base model: queried via Together API (fast)
- Trained model: loaded locally with LoRA adapter on MPS (slow but works)
- Judge: Llama 70B via Together API
"""

import json
import os
import random
from pathlib import Path

import openai
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────
ADAPTER_CANDIDATES = [
    Path("./grpo_direct"),
    Path("/Users/zane/Downloads/grpo_direct"),
]
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
POLICY_MODEL_API = "Qwen/Qwen2.5-7B-Instruct-Turbo"
NUM_EVAL = 10  # number of prompts to evaluate
MAX_NEW_TOKENS = 200

TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
client = openai.OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")

JUDGE_TEMPLATE = """You are an impartial judge. Given an instruction and two responses, decide which response is better.

Consider: accuracy, helpfulness, clarity, and conciseness.

Respond with EXACTLY "A" or "B" (no other text).

Instruction: {instruction}

Response A:
{response_a}

Response B:
{response_b}"""


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    """Return the first existing path from a list of candidates."""
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried: {tried}")


def get_eval_prompts(path: str, n: int) -> list[str]:
    """Get n random unique prompts from preferences.jsonl."""
    prompts = []
    seen = set()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            inst = row["instruction"]
            if inst not in seen:
                seen.add(inst)
                prompts.append(inst)
    random.seed(123)
    return random.sample(prompts, min(n, len(prompts)))


def generate_base(instruction: str) -> str:
    """Generate from base model via Together API."""
    resp = client.chat.completions.create(
        model=POLICY_MODEL_API,
        messages=[{"role": "user", "content": instruction}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def judge(instruction: str, response_a: str, response_b: str) -> str:
    """Judge which response is better."""
    prompt = JUDGE_TEMPLATE.format(
        instruction=instruction, response_a=response_a, response_b=response_b,
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip().upper()
    if text in ("A", "B"):
        return text
    return "TIE"


def main():
    adapter_path = resolve_existing_path(ADAPTER_CANDIDATES, "LoRA adapter directory")
    data_path = resolve_existing_path(
        [
            Path("preferences.jsonl"),
            adapter_path / "preferences.jsonl",
        ],
        "preferences dataset",
    )

    print(f"Using adapter: {adapter_path}")
    print(f"Using data: {data_path}")
    prompts = get_eval_prompts(str(data_path), NUM_EVAL)
    print(f"Evaluating {len(prompts)} prompts\n")

    # ── Load trained model locally on MPS ────────────────────────────────
    print("Loading base model + LoRA adapter on MPS...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map=device,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("Model loaded.\n")

    # ── Generate and judge ───────────────────────────────────────────────
    trained_wins = 0
    base_wins = 0
    ties = 0

    for i, instruction in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {instruction[:80]}...")

        # Base model response (API)
        base_response = generate_base(instruction)

        # Trained model response (local)
        messages = [{"role": "user", "content": instruction}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7, do_sample=True, top_p=0.9,
            )
        trained_response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Judge both orderings to control position bias
        swap = random.random() < 0.5
        if swap:
            v = judge(instruction, trained_response, base_response)
            winner = "trained" if v == "A" else "base" if v == "B" else "tie"
        else:
            v = judge(instruction, base_response, trained_response)
            winner = "base" if v == "A" else "trained" if v == "B" else "tie"

        if winner == "trained":
            trained_wins += 1
        elif winner == "base":
            base_wins += 1
        else:
            ties += 1

        print(f"  → Winner: {winner}")
        print(f"  Base ({len(base_response)} chars): {base_response[:100]}...")
        print(f"  Trained ({len(trained_response)} chars): {trained_response[:100]}...")
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"RESULTS ({len(prompts)} prompts)")
    print(f"  Trained model wins: {trained_wins}")
    print(f"  Base model wins:    {base_wins}")
    print(f"  Ties:               {ties}")
    print(f"  Trained win rate:   {trained_wins / len(prompts):.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
