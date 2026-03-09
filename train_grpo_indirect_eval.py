"""
Indirect GRPO + evaluation pipeline.

Training:  Local Bradley-Terry reward model (no API calls during training).
           Uses G=8 generations (increased from 4) to average out reward noise
           per the 69% RM accuracy feedback.

Quick checkpoint evals (no API, ~20-40s each, fires on every save):
  1. RM score on policy-generated completions (mean ± std)
  2. Response length (mean chars)
  3. Lexical diversity (type-token ratio) — reward-hacking signal

Full final eval (Together API, same methodology as direct GRPO):
  1. Head-to-head win rate vs base Qwen 7B (default 50 prompts)
  2. Position bias controlled — both A/B orderings, averaged
  3. Judge: Llama 3.3-70B-Instruct-Turbo
  4. Results saved to {output}/eval_results.json

Usage:
    python train_grpo_indirect_eval.py --reward-model ./reward_model
    python train_grpo_indirect_eval.py --reward-model ./reward_model --num-generations 8
    python train_grpo_indirect_eval.py --reward-model ./reward_model --skip-final-eval
"""

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ── Together API (evaluation only) ───────────────────────────────────────────
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
client = openai.OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
)

BASE_MODEL_API = "Qwen/Qwen2.5-7B-Instruct-Turbo"    # base for final eval comparisons
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
    "reward_model": "./reward_model",
    "data": "preferences.jsonl",
    "output": "./grpo_indirect",
    "max_steps": 200,
    "num_generations": 8,       # increased from 4 — averages out noise from 69% RM
    "max_completion_length": 256,
    "temperature": 0.8,
    "beta": 0.05,
    "lr": 5e-6,
    "batch_size": 2,
    "grad_accum": 4,
    "lora_r": 16,
    "lora_alpha": 64,
    "eval_split": 0.1,
    "checkpoint_eval_prompts": 8,   # prompts for quick RM eval at each checkpoint
    "final_eval_prompts": 50,       # prompts for head-to-head final eval
    "judge_workers": 8,
}


# ── API helpers ───────────────────────────────────────────────────────────────

def api_call_with_retry(fn):
    """Retry an API call with exponential backoff on transient errors."""
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
                print(f"    [retry {attempt + 1}/{MAX_RETRIES}] HTTP {e.status_code}, waiting {delay:.1f}s")
                time.sleep(delay)
            else:
                raise


def parse_verdict(text: str) -> str | None:
    text = text.strip().upper()
    if text in ("A", "B"):
        return text
    for line in text.splitlines():
        line = line.strip()
        if line in ("A", "B"):
            return line
        if line.startswith("WINNER:"):
            pick = line.split(":", 1)[1].strip()
            if pick in ("A", "B"):
                return pick
    return None


def judge_pair(instruction: str, response_a: str, response_b: str) -> str | None:
    """Query Llama 70B to pick A or B. Returns 'A', 'B', or None on failure."""
    prompt = JUDGE_TEMPLATE.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
    )

    def call():
        return client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )

    resp = api_call_with_retry(call)
    return parse_verdict(resp.choices[0].message.content)


def generate_base_response(instruction: str, max_tokens: int = 256) -> str:
    """Generate a response from the base model via Together API."""
    def call():
        return client.chat.completions.create(
            model=BASE_MODEL_API,
            messages=[{"role": "user", "content": instruction}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
    resp = api_call_with_retry(call)
    return resp.choices[0].message.content


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_prompts(path: str) -> list:
    """Return unique prompts from preferences.jsonl as chat-format message lists."""
    prompts, seen = [], set()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            inst = row["instruction"]
            if inst not in seen:
                seen.add(inst)
                prompts.append([{"role": "user", "content": inst}])
    return prompts


def load_raw_instructions(path: str) -> list[str]:
    """Return unique raw instruction strings."""
    instructions, seen = [], set()
    with open(path) as f:
        for line in f:
            inst = json.loads(line)["instruction"]
            if inst not in seen:
                seen.add(inst)
                instructions.append(inst)
    return instructions


# ── Reward model ──────────────────────────────────────────────────────────────

def load_reward_model(path: str, max_length: int = 512):
    """Load BT reward model. Returns (reward_fn, rm_tokenizer, rm_model, rm_device)."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=1, torch_dtype=dtype,
    ).to(device)
    model.eval()

    def reward_fn(prompts, completions, **kwargs):
        texts = [p + c for p, c in zip(prompts, completions)]
        inputs = tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)
        with torch.no_grad():
            return model(**inputs).logits[:, 0].tolist()

    return reward_fn, tokenizer, model, device


def sanity_check_reward(reward_fn, rm_tokenizer):
    messages = [{"role": "user", "content": "What is machine learning?"}]
    prompt = rm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    good = (
        "Machine learning is a subset of artificial intelligence where systems learn "
        "patterns from data to make predictions without being explicitly programmed."
    )
    bad = "idk lol"
    scores = reward_fn(prompts=[prompt, prompt], completions=[good, bad])
    print(f"Sanity check — good: {scores[0]:.3f}, bad: {scores[1]:.3f}")
    if scores[0] > scores[1]:
        print("  PASSED\n")
    else:
        print("  WARNING: bad response scored ≥ good — check RM format\n")


# ── Quick checkpoint evaluation callback ──────────────────────────────────────

class CheckpointEvalCallback(TrainerCallback):
    """Runs lightweight evals after each checkpoint save (no API calls).

    Metrics tracked:
      rm_score_mean / rm_score_std  — how well current policy outputs score on BT RM
      response_length_mean          — chars; rising + stable RM score = padding hack
      type_token_ratio              — lexical diversity; falling = repetition/collapse
    """

    def __init__(self, reward_fn, rm_tokenizer, instructions, policy_tokenizer, n_prompts):
        self.reward_fn = reward_fn
        self.rm_tokenizer = rm_tokenizer
        self.instructions = instructions
        self.policy_tokenizer = policy_tokenizer
        self.n_prompts = n_prompts
        self.results: list[dict] = []
        self.trainer = None   # set externally after trainer is constructed

    def on_save(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        print(f"\n{'─'*60}")
        print(f"QUICK CHECKPOINT EVAL @ step {step}")

        if model is None or self.trainer is None:
            print("  (model not accessible, skipping generation eval)")
            return

        sampled = random.sample(self.instructions, min(self.n_prompts, len(self.instructions)))

        # Generate completions from the current policy
        was_training = model.training
        model.eval()

        completions, prompts_fmt, all_words = [], [], []

        for instr in sampled:
            messages = [{"role": "user", "content": instr}]
            prompt_str = self.policy_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            try:
                policy_device = next(model.parameters()).device
                inputs = self.policy_tokenizer(
                    prompt_str, return_tensors="pt",
                ).to(policy_device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.policy_tokenizer.eos_token_id,
                    )
                completion = self.policy_tokenizer.decode(
                    out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
                )
            except Exception as e:
                print(f"  [warn] generation failed for one prompt: {e}")
                completion = ""

            completions.append(completion)
            prompts_fmt.append(prompt_str)
            all_words.extend(completion.lower().split())

        if was_training:
            model.train()

        # RM scores on generated completions
        try:
            rm_scores = self.reward_fn(prompts=prompts_fmt, completions=completions)
        except Exception as e:
            print(f"  [warn] RM scoring failed: {e}")
            rm_scores = [0.0] * len(completions)

        n = len(rm_scores)
        mean_rm = sum(rm_scores) / n if n else 0.0
        var_rm = sum((s - mean_rm) ** 2 for s in rm_scores) / n if n else 0.0
        std_rm = math.sqrt(var_rm)

        # Response length
        lengths = [len(c) for c in completions]
        mean_len = sum(lengths) / len(lengths) if lengths else 0.0

        # Type-token ratio (lexical diversity; falling = collapse / format hacking)
        ttr = len(set(all_words)) / len(all_words) if all_words else 0.0

        result = {
            "step": step,
            "rm_score_mean": round(mean_rm, 4),
            "rm_score_std": round(std_rm, 4),
            "response_length_mean": round(mean_len, 1),
            "type_token_ratio": round(ttr, 4),
        }
        self.results.append(result)

        print(f"  RM score:    {mean_rm:.4f} ± {std_rm:.4f}")
        print(f"  Resp length: {mean_len:.1f} chars (mean over {n} samples)")
        print(f"  Lex div TTR: {ttr:.3f}  (↓ = collapse / padding hack)")
        print(f"{'─'*60}\n")


# ── Full final evaluation (Together API) ─────────────────────────────────────

def generate_local_response(
    model, tokenizer, instruction: str, max_new_tokens: int = 256,
) -> str:
    """Generate a response from the given local model."""
    messages = [{"role": "user", "content": instruction}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def run_full_eval(
    model,
    tokenizer,
    instructions: list[str],
    n_prompts: int,
    judge_workers: int,
    output_dir: str,
) -> dict:
    """Head-to-head: trained model vs base model, judged by Llama 70B.

    Mirrors the methodology from eval_quick.py / direct GRPO evaluation:
    - One random A/B ordering per prompt (controls position bias without
      doubling API calls; uncomment the both-ordering block below if budget allows)
    - Judge: Llama 3.3-70B-Instruct-Turbo
    """
    eval_instructions = random.sample(instructions, min(n_prompts, len(instructions)))
    n = len(eval_instructions)

    print(f"\n{'='*60}")
    print(f"FULL FINAL EVAL  ({n} prompts vs base model via Together API)")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"{'='*60}")

    model.eval()

    trained_wins = base_wins = ties = 0
    detailed = []

    def eval_one(i_instr):
        i, instr = i_instr
        # Generate from trained model (local)
        try:
            trained_resp = generate_local_response(model, tokenizer, instr)
        except Exception as e:
            return i, instr, None, None, "error", str(e)

        # Generate from base model (Together API)
        try:
            base_resp = generate_base_response(instr)
        except Exception as e:
            return i, instr, trained_resp, None, "error", str(e)

        # Judge with randomised A/B order
        swap = random.random() < 0.5
        if swap:
            verdict = judge_pair(instr, trained_resp, base_resp)
            winner = "trained" if verdict == "A" else ("base" if verdict == "B" else "tie")
        else:
            verdict = judge_pair(instr, base_resp, trained_resp)
            winner = "base" if verdict == "A" else ("trained" if verdict == "B" else "tie")

        return i, instr, trained_resp, base_resp, winner, None

    with ThreadPoolExecutor(max_workers=judge_workers) as pool:
        futures = {pool.submit(eval_one, (i, instr)): i for i, instr in enumerate(eval_instructions)}
        for future in as_completed(futures):
            i, instr, trained_resp, base_resp, winner, err = future.result()

            if err:
                print(f"  [{i+1}/{n}] ERROR: {err}")
                ties += 1
                winner = "tie"
            else:
                if winner == "trained":
                    trained_wins += 1
                elif winner == "base":
                    base_wins += 1
                else:
                    ties += 1
                print(f"  [{i+1}/{n}] {winner:8s} | {instr[:70]}...")

            detailed.append({
                "instruction": instr,
                "trained_response": trained_resp or "",
                "base_response": base_resp or "",
                "winner": winner,
            })

    results = {
        "n_prompts": n,
        "trained_wins": trained_wins,
        "base_wins": base_wins,
        "ties": ties,
        "trained_win_rate": trained_wins / n,
        "base_win_rate": base_wins / n,
        "detailed": detailed,
    }

    print(f"\n{'='*60}")
    print(f"FINAL EVAL RESULTS")
    print(f"  Trained model wins:  {trained_wins} / {n}  ({trained_wins/n:.1%})")
    print(f"  Base model wins:     {base_wins} / {n}  ({base_wins/n:.1%})")
    print(f"  Ties / errors:       {ties}")
    print(f"{'='*60}")

    out_path = os.path.join(output_dir, "eval_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Indirect GRPO with BT reward model + multi-stage evaluation"
    )
    parser.add_argument("--policy-model", default=DEFAULTS["policy_model"])
    parser.add_argument("--reward-model", default=DEFAULTS["reward_model"])
    parser.add_argument("--data", default=DEFAULTS["data"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--max-steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--num-generations", type=int, default=DEFAULTS["num_generations"],
                        help="G completions per prompt. Default 8 (increased from 4 to "
                             "average out noise from 69%% RM accuracy)")
    parser.add_argument("--max-completion-length", type=int, default=DEFAULTS["max_completion_length"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--beta", type=float, default=DEFAULTS["beta"],
                        help="KL penalty coefficient")
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--lora-r", type=int, default=DEFAULTS["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"])
    parser.add_argument("--eval-split", type=float, default=DEFAULTS["eval_split"])
    parser.add_argument("--checkpoint-eval-prompts", type=int,
                        default=DEFAULTS["checkpoint_eval_prompts"],
                        help="# prompts for quick RM eval at each checkpoint save")
    parser.add_argument("--final-eval-prompts", type=int,
                        default=DEFAULTS["final_eval_prompts"],
                        help="# prompts for full head-to-head final eval")
    parser.add_argument("--judge-workers", type=int, default=DEFAULTS["judge_workers"],
                        help="Parallel workers for final eval judge calls")
    parser.add_argument("--skip-sanity-check", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true",
                        help="Skip Together API eval at end (training + checkpoint evals only)")
    args = parser.parse_args()

    # ── 1. Load reward model ─────────────────────────────────────────────────

    print(f"Loading reward model from {args.reward_model}...")
    reward_fn, rm_tokenizer, rm_model, rm_device = load_reward_model(args.reward_model)

    if not args.skip_sanity_check:
        sanity_check_reward(reward_fn, rm_tokenizer)

    # ── 2. Load prompts / data ───────────────────────────────────────────────

    print(f"Loading prompts from {args.data}...")
    prompts = load_prompts(args.data)
    all_instructions = load_raw_instructions(args.data)
    dataset = Dataset.from_dict({"prompt": prompts})
    split = dataset.train_test_split(test_size=args.eval_split, seed=42)
    print(f"Train: {len(split['train'])} | Eval: {len(split['test'])} prompts")

    # ── 3. Policy tokenizer ──────────────────────────────────────────────────

    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = "left"

    # ── 4. Checkpoint eval callback ──────────────────────────────────────────

    ckpt_callback = CheckpointEvalCallback(
        reward_fn=reward_fn,
        rm_tokenizer=rm_tokenizer,
        instructions=all_instructions,
        policy_tokenizer=policy_tokenizer,
        n_prompts=args.checkpoint_eval_prompts,
    )

    # ── 5. LoRA config ───────────────────────────────────────────────────────

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 6. GRPO config ───────────────────────────────────────────────────────

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Disable TRL's built-in eval_strategy — it uses the RM reward on the eval
    # split which isn't very informative (within-group normalization makes mean
    # reward ~0.5 by construction). Our custom callback handles eval instead.
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
        eval_strategy="no",         # custom callback handles eval
        save_total_limit=5,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    # ── 7. Initialize trainer ────────────────────────────────────────────────

    print(f"\nInitializing GRPO:")
    print(f"  policy  = {args.policy_model}")
    print(f"  G       = {args.num_generations} (completions/prompt)")
    print(f"  β       = {args.beta}  (KL penalty)")
    print(f"  lr      = {args.lr}")
    print(f"  steps   = {args.max_steps}")
    print(f"  ckpt eval every 50 steps ({args.checkpoint_eval_prompts} prompts, no API)\n")

    trainer = GRPOTrainer(
        model=args.policy_model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=split["train"],
        peft_config=lora_config,
        processing_class=policy_tokenizer,
        callbacks=[ckpt_callback],
    )

    # Give the callback a reference to the trainer so it can access trainer.model
    ckpt_callback.trainer = trainer

    # ── 8. Train ─────────────────────────────────────────────────────────────

    print("Starting indirect GRPO training...")
    trainer.train()

    # ── 9. Save ──────────────────────────────────────────────────────────────

    trainer.save_model(args.output)
    policy_tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}")

    # ── 10. Checkpoint eval summary ──────────────────────────────────────────

    if ckpt_callback.results:
        print(f"\n{'─'*60}")
        print("CHECKPOINT EVAL SUMMARY")
        print(f"{'step':>6}  {'RM mean':>8}  {'RM std':>7}  {'len':>7}  {'TTR':>6}")
        print(f"{'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}")
        for r in ckpt_callback.results:
            print(
                f"{r['step']:>6}  "
                f"{r['rm_score_mean']:>8.4f}  "
                f"{r['rm_score_std']:>7.4f}  "
                f"{r['response_length_mean']:>7.1f}  "
                f"{r['type_token_ratio']:>6.3f}"
            )
        print(f"{'─'*60}")

        # Save checkpoint eval history
        ckpt_summary_path = os.path.join(args.output, "checkpoint_eval_summary.json")
        with open(ckpt_summary_path, "w") as f:
            json.dump(ckpt_callback.results, f, indent=2)
        print(f"Checkpoint eval summary saved to {ckpt_summary_path}")

    # ── 11. Full final eval (Together API) ───────────────────────────────────

    if args.skip_final_eval:
        print("\n--skip-final-eval set, skipping full eval.")
        return

    if not TOGETHER_API_KEY:
        print("\nWARNING: TOGETHER_API_KEY not set, skipping final eval.")
        return

    run_full_eval(
        model=trainer.model,
        tokenizer=policy_tokenizer,
        instructions=all_instructions,
        n_prompts=args.final_eval_prompts,
        judge_workers=args.judge_workers,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
