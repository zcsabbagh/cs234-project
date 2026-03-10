"""
Indirect GRPO: Train Qwen policy using a pre-trained Bradley-Terry reward model.

This implements the "indirect" approach from the RLAIF comparison:
  - A pre-trained RM (from train_reward_model.py) scores each completion
  - GRPO samples G completions per prompt, normalizes rewards within the group,
    and updates the policy with a KL penalty against the frozen reference model
  - LoRA keeps memory usage manageable for 7B-scale models

Usage:
    python train_grpo_indirect.py --reward-model ./reward_model --data preferences.jsonl
    python train_grpo_indirect.py --policy-model Qwen/Qwen2.5-3B-Instruct --max-steps 100
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

DEFAULTS = {
    "policy_model": "Qwen/Qwen2.5-7B-Instruct",
    "reward_model": "./reward_model",
    "data": "preferences.jsonl",
    "output": "./grpo_indirect",
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
}


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


def load_reward_model(path: str, max_length: int = 512):
    """Load the trained RM and return (reward_fn, tokenizer).

    The reward function expects prompts and completions as strings where
    prompt is already chat-template-formatted (as GRPOTrainer provides)
    and completion is the raw generated text. Concatenation reproduces the
    exact format used during RM training (apply_chat_template on message lists).
    """
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=1, dtype=dtype,
    ).to(device)
    model.eval()

    def reward_fn(prompts, completions, **kwargs):
        # Newer TRL passes prompts as list[list[dict]] (message dicts).
        # Convert to formatted strings if needed.
        texts = []
        for p, c in zip(prompts, completions):
            if isinstance(p, list):
                p = tokenizer.apply_chat_template(
                    p, tokenize=False, add_generation_prompt=True
                )
            if isinstance(c, list):
                # conversation format: extract last assistant turn
                c = c[-1]["content"] if c and isinstance(c[-1], dict) else ""
            texts.append(p + c)
        inputs = tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)
        with torch.no_grad():
            return model(**inputs).logits[:, 0].tolist()

    return reward_fn, tokenizer


def sanity_check_reward(reward_fn, tokenizer):
    """Verify the RM scores a good response higher than a bad one."""
    messages = [{"role": "user", "content": "What is machine learning?"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    good = (
        "Machine learning is a subset of artificial intelligence where "
        "systems learn patterns from data to make predictions without "
        "being explicitly programmed."
    )
    bad = "idk lol"

    scores = reward_fn(prompts=[prompt, prompt], completions=[good, bad])
    print(f"Sanity check — good: {scores[0]:.3f}, bad: {scores[1]:.3f}")
    if scores[0] > scores[1]:
        print("  PASSED: good response scored higher\n")
    else:
        print("  WARNING: bad response scored equal/higher — check RM format\n")


def main():
    parser = argparse.ArgumentParser(description="Indirect GRPO with pre-trained reward model")
    parser.add_argument("--policy-model", default=DEFAULTS["policy_model"])
    parser.add_argument("--reward-model", default=DEFAULTS["reward_model"])
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
    parser.add_argument("--skip-sanity-check", action="store_true")
    args = parser.parse_args()

    # ── 1. Load reward model ────────────────────────────────────────────────

    print(f"Loading reward model from {args.reward_model}...")
    reward_fn, rm_tokenizer = load_reward_model(args.reward_model)

    if not args.skip_sanity_check:
        sanity_check_reward(reward_fn, rm_tokenizer)

    # ── 2. Load prompts ─────────────────────────────────────────────────────

    print(f"Loading prompts from {args.data}...")
    prompts = load_prompts(args.data)
    dataset = Dataset.from_dict({"prompt": prompts})
    split = dataset.train_test_split(test_size=args.eval_split, seed=42)
    print(f"Train: {len(split['train'])} | Eval: {len(split['test'])} prompts")

    # ── 3. LoRA config ──────────────────────────────────────────────────────

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 4. GRPO config ──────────────────────────────────────────────────────

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

    # ── 5. Policy tokenizer ─────────────────────────────────────────────────

    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = "left"

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

    print("Starting GRPO training...")
    trainer.train()

    # ── 8. Save ─────────────────────────────────────────────────────────────

    trainer.save_model(args.output)
    policy_tokenizer.save_pretrained(args.output)

    print(f"\nTraining complete. Model saved to {args.output}")


if __name__ == "__main__":
    main()
