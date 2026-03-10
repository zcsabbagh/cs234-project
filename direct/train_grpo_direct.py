"""
Direct GRPO: Train Qwen policy using live Llama 70B pairwise judging.

This implements the "direct" approach from the RLAIF comparison:
  - Instead of a pre-trained reward model, each training step queries the
    Llama 70B judge to compare all pairs of completions
  - For G=4 completions, form all 6 unordered pairs (12 ordered to control
    position bias), compute each completion's win fraction as its reward
  - GRPO normalizes rewards within the group and updates the policy

Usage:
    python train_grpo_direct.py --data preferences.jsonl
    python train_grpo_direct.py --policy-model Qwen/Qwen2.5-3B-Instruct --max-steps 100
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

import openai
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]

JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
MAX_RETRIES = 5
BASE_DELAY = 1.0

JUDGE_TEMPLATE = """You are an impartial judge. Given an instruction and two responses, decide which response is better.

Consider: accuracy, helpfulness, clarity, and conciseness.

Respond with EXACTLY "A" or "B" (no other text).

Instruction: {instruction}

Response A:
{response_a}

Response B:
{response_b}"""

DEFAULTS = {
    "policy_model": "Qwen/Qwen2.5-7B-Instruct",
    "data": "preferences.jsonl",
    "output": "./grpo_direct",
    "max_steps": 200,
    "num_generations": 4,
    "max_completion_length": 256,
    "temperature": 0.8,
    "beta": 0.05,
    "lr": 5e-6,
    "batch_size": 2,
    "grad_accum": 4,
    "lora_r": 16,
    "lora_alpha": 64,
    "eval_split": 0.1,
    "judge_workers": 8,
}

client = openai.OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
)


def api_call_with_retry(fn):
    """Retry an API call with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] {type(e).__name__}, waiting {delay:.1f}s")
            time.sleep(delay)
        except openai.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                print(f"    [retry {attempt + 1}/{MAX_RETRIES}] {e.status_code}, waiting {delay:.1f}s")
                time.sleep(delay)
            else:
                raise


def parse_verdict(judgment: str) -> str | None:
    """Extract 'A' or 'B' from the judge response."""
    text = judgment.strip().upper()
    if text in ("A", "B"):
        return text
    for line in judgment.splitlines():
        line = line.strip().upper()
        if line in ("A", "B"):
            return line
        if line.startswith("WINNER:"):
            pick = line.split(":", 1)[1].strip()
            if pick in ("A", "B"):
                return pick
    return None


def judge_single_pair(instruction: str, response_a: str, response_b: str) -> str | None:
    """Query Llama 70B to judge a single ordered pair. Returns 'A' or 'B' or None."""
    prompt = JUDGE_TEMPLATE.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
    )

    def call():
        return client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
        )

    resp = api_call_with_retry(call)
    return parse_verdict(resp.choices[0].message.content)


def compute_win_fractions(instruction: str, completions: list[str], workers: int) -> list[float]:
    """Compute win fraction for each completion via pairwise judging.

    For G completions, forms all C(G,2) unordered pairs. For each pair,
    queries both orderings (A,B) and (B,A) to control position bias.
    Each completion's reward = wins / total_comparisons_involving_it.
    """
    G = len(completions)
    pairs = list(combinations(range(G), 2))  # C(G,2) unordered pairs
    wins = [0.0] * G
    total = [0] * G

    # Build all judge tasks: 2 orderings per unordered pair
    tasks = []
    for i, j in pairs:
        tasks.append((i, j))  # completion[i] as A, completion[j] as B
        tasks.append((j, i))  # completion[j] as A, completion[i] as B

    results = {}

    def run_judge(a_idx, b_idx):
        verdict = judge_single_pair(instruction, completions[a_idx], completions[b_idx])
        return (a_idx, b_idx, verdict)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_judge, a_idx, b_idx): (a_idx, b_idx)
            for a_idx, b_idx in tasks
        }
        for future in as_completed(futures):
            a_idx, b_idx, verdict = future.result()
            results[(a_idx, b_idx)] = verdict

    # Tally wins from both orderings
    for i, j in pairs:
        # Ordering 1: i=A, j=B
        v1 = results.get((i, j))
        if v1 == "A":
            wins[i] += 1
        elif v1 == "B":
            wins[j] += 1
        # else: tie / parse failure — neither gets a win

        # Ordering 2: j=A, i=B
        v2 = results.get((j, i))
        if v2 == "A":
            wins[j] += 1
        elif v2 == "B":
            wins[i] += 1

        total[i] += 2
        total[j] += 2

    # Win fraction per completion
    return [wins[k] / total[k] if total[k] > 0 else 0.0 for k in range(G)]


def load_prompts(path: str) -> list:
    """Extract unique prompts from preferences.jsonl as chat-format message lists."""
    prompts = []
    seen = set()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            inst = row["instruction"]
            if inst not in seen:
                seen.add(inst)
                prompts.append([{"role": "user", "content": inst}])
    return prompts


def make_judge_reward_fn(policy_tokenizer, judge_workers: int):
    """Create a reward function that uses live pairwise judging.

    GRPOTrainer calls reward_fn(prompts, completions) where:
      - prompts: list of strings (chat-template-formatted)
      - completions: list of strings (raw generated text)

    The prompts come in groups of G (num_generations) per original prompt.
    We need to group them, run pairwise judging per group, and return
    per-completion rewards.
    """

    def reward_fn(prompts, completions, **kwargs):
        # Group by prompt — consecutive G entries share the same prompt
        # GRPOTrainer interleaves: [p1,p1,p1,p1, p2,p2,p2,p2, ...]
        rewards = []
        batch_size = len(prompts)

        # Detect group size by counting consecutive identical prompts
        G = 1
        while G < batch_size and prompts[G] == prompts[0]:
            G += 1

        for start in range(0, batch_size, G):
            group_prompts = prompts[start:start + G]
            group_completions = completions[start:start + G]

            # Extract the original instruction from the chat-formatted prompt
            # The prompt is already formatted via chat template; extract user content
            instruction = group_prompts[0]
            # Try to extract raw instruction if it's chat-template formatted
            # Fall back to using the raw prompt string
            try:
                # Attempt to find the user message content between template markers
                # This is model-specific; for Qwen it's between <|im_start|>user\n and <|im_end|>
                if "<|im_start|>user\n" in instruction:
                    instruction = instruction.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
            except (IndexError, AttributeError):
                pass

            win_fracs = compute_win_fractions(
                instruction, group_completions, workers=judge_workers,
            )
            rewards.extend(win_fracs)

        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="Direct GRPO with live pairwise judging")
    parser.add_argument("--policy-model", default=DEFAULTS["policy_model"])
    parser.add_argument("--data", default=DEFAULTS["data"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--max-steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--num-generations", type=int, default=DEFAULTS["num_generations"])
    parser.add_argument("--max-completion-length", type=int, default=DEFAULTS["max_completion_length"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--beta", type=float, default=DEFAULTS["beta"], help="KL penalty coefficient")
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--lora-r", type=int, default=DEFAULTS["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"])
    parser.add_argument("--eval-split", type=float, default=DEFAULTS["eval_split"])
    parser.add_argument("--judge-workers", type=int, default=DEFAULTS["judge_workers"],
                        help="Parallel workers for judge API calls")
    args = parser.parse_args()

    # ── 1. Load prompts ─────────────────────────────────────────────────────

    print(f"Loading prompts from {args.data}...")
    prompts = load_prompts(args.data)
    dataset = Dataset.from_dict({"prompt": prompts})
    split = dataset.train_test_split(test_size=args.eval_split, seed=42)
    print(f"Train: {len(split['train'])} | Eval: {len(split['test'])} prompts")

    # ── 2. Policy tokenizer ─────────────────────────────────────────────────

    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = "left"

    # ── 3. Build live judge reward function ──────────────────────────────────

    print(f"Using live judge: {JUDGE_MODEL}")
    print(f"  Per prompt: {args.num_generations} completions → "
          f"{args.num_generations * (args.num_generations - 1)} judge calls "
          f"(both orderings)")
    print(f"  Per batch of {args.batch_size} prompts: "
          f"{args.batch_size * args.num_generations * (args.num_generations - 1)} judge calls")
    print(f"  Judge workers: {args.judge_workers}\n")

    reward_fn = make_judge_reward_fn(policy_tokenizer, judge_workers=args.judge_workers)

    # ── 4. LoRA config ──────────────────────────────────────────────────────

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 5. GRPO config ──────────────────────────────────────────────────────

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=20,
        logging_steps=10,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=5,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    # ── 6. Initialize trainer ───────────────────────────────────────────────

    print(f"Initializing GRPO: policy={args.policy_model}, G={args.num_generations}, β={args.beta}")

    trainer = GRPOTrainer(
        model=args.policy_model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=lora_config,
        processing_class=policy_tokenizer,
    )

    # ── 7. Train ────────────────────────────────────────────────────────────

    print("Starting GRPO training with live judging...")
    trainer.train()

    # ── 8. Save ─────────────────────────────────────────────────────────────

    trainer.save_model(args.output)
    policy_tokenizer.save_pretrained(args.output)

    print(f"\nTraining complete. Model saved to {args.output}")


if __name__ == "__main__":
    main()
