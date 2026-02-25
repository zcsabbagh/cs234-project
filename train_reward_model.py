"""
Train a Bradley-Terry reward model on pairwise preference data using TRL.

Base model: Qwen/Qwen2.5-1.5B-Instruct
Data: preferences.jsonl (from generate_preferences.py)
Loss: -log(sigma(r(x, y_w) - r(x, y_l)))
"""

import argparse
import json

import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

DEFAULTS = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "data": "preferences.jsonl",
    "output": "./reward_model",
    "epochs": 3,
    "batch_size": 4,
    "grad_accum": 4,
    "lr": 1e-4,
    "max_length": 512,
    "eval_split": 0.1,
}


def load_preferences(path: str) -> Dataset:
    """Load preferences.jsonl into TRL reward format (chosen/rejected chat columns)."""
    chosen, rejected = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            winner = row.get("winner")
            loser = row.get("loser")
            if winner is None or loser is None:
                continue  # Skip rows where judge couldn't determine winner/loser
            instruction = row["instruction"]
            winner_text = row[winner]
            loser_text = row[loser]
            chosen.append([
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": winner_text},
            ])
            rejected.append([
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": loser_text},
            ])
    return Dataset.from_dict({"chosen": chosen, "rejected": rejected})


def get_device_config():
    """Return (torch_dtype, bf16_flag, fp16_flag) based on available hardware."""
    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        return (torch.bfloat16 if bf16 else torch.float16), bf16, not bf16
    # MPS (Apple Silicon) and CPU: use fp32 for stability
    return torch.float32, False, False


def main():
    parser = argparse.ArgumentParser(description="Train Bradley-Terry reward model")
    parser.add_argument("--data", default=DEFAULTS["data"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--max-length", type=int, default=DEFAULTS["max_length"])
    parser.add_argument("--eval-split", type=float, default=DEFAULTS["eval_split"])
    args = parser.parse_args()

    print(f"Loading preferences from {args.data}...")
    dataset = load_preferences(args.data)
    split = dataset.train_test_split(test_size=args.eval_split, seed=42)
    print(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")

    dtype, use_bf16, use_fp16 = get_device_config()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {args.model} (dtype={dtype}, device={device})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = RewardConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        logging_steps=10,
        bf16=use_bf16,
        fp16=use_fp16,
        warmup_steps=10,
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    metrics = trainer.evaluate()
    print(f"\nFinal eval metrics: {metrics}")
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
