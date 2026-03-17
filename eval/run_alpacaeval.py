"""
AlpacaEval: Judge trained model outputs vs base model using GPT-4o (held-out judge).

Using GPT-4o instead of Llama 70B (the training judge) is critical for detecting
reward hacking — a model that exploits Llama 70B quirks will score well on Llama
but not on GPT-4o.

Outputs:
  results/alpacaeval_results.json     win rates + raw verdicts
  results/alpacaeval_table.txt        formatted table (also printed)
  results/alpacaeval_bar.png          win-rate bar chart

Usage:
    export OPENAI_API_KEY=sk-...
    python eval/run_alpacaeval.py --output-dir ./eval_outputs --results-dir ./eval_results
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JUDGE_MODEL  = "gpt-4o"
MAX_WORKERS  = 4

JUDGE_TEMPLATE = """\
You are an impartial judge evaluating two AI assistant responses.

Instruction: {instruction}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider accuracy, helpfulness, clarity, and conciseness.
Penalize: unnecessary preamble ("Great question!", "Certainly!"), restating the question, \
padding, and filler phrases.

Respond with EXACTLY one letter — A or B — and nothing else."""


# ── Judge ─────────────────────────────────────────────────────────────────

def judge_pair(
    client: openai.OpenAI,
    instruction: str,
    response_a: str,
    response_b: str,
    max_retries: int = 4,
) -> str | None:
    prompt = JUDGE_TEMPLATE.format(
        instruction=instruction,
        response_a=response_a[:2000],
        response_b=response_b[:2000],
    )
    for attempt in range(max_retries):
        try:
            msg = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.choices[0].message.content.strip().upper()
            if text in ("A", "B"):
                return text
            # Accept "Winner: A" format too
            for word in text.split():
                if word in ("A", "B"):
                    return word
            return None
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err:
                wait = 5 * (2 ** attempt)  # 5, 10, 20, 40 s
                m = re.search(r"try again in (\d+(?:\.\d+)?)s", err)
                if m:
                    wait = float(m.group(1)) + 1
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    print(f"  [JUDGE ERROR] {e}")
                    return None
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [JUDGE ERROR] {e}")
                return None


# ── Load outputs ─────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_base_responses(path: str) -> dict[str, str]:
    """Load alpaca_base.jsonl and return {instruction: response}."""
    return {row["instruction"]: row["response"] for row in load_jsonl(path)}


# ── Evaluate one model ────────────────────────────────────────────────────

def evaluate_model(
    client: openai.OpenAI,
    rows: list[dict],
    base_responses: dict[str, str],
    model_name: str,
    n_eval: int,
    workers: int = MAX_WORKERS,
) -> dict:
    """
    Compare model responses vs base model using GPT-4o judge.
    Returns win rate and per-prompt verdicts.
    """
    rows = rows[:n_eval]
    verdicts = [None] * len(rows)

    def judge_one(i, row):
        instruction = row["instruction"]
        model_resp  = row["response"]
        base_resp   = base_responses.get(instruction, "")

        # Randomize position to control A/B bias
        swap = random.random() < 0.5
        if swap:
            verdict = judge_pair(client, instruction, model_resp, base_resp)
            win = (verdict == "A")
        else:
            verdict = judge_pair(client, instruction, base_resp, model_resp)
            win = (verdict == "B")

        return i, win, verdict

    print(f"  Judging {len(rows)} pairs for {model_name}...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge_one, i, row): i for i, row in enumerate(rows)}
        done = 0
        wins = 0
        for f in as_completed(futures):
            i, win, verdict = f.result()
            verdicts[i] = {"win": win, "verdict": verdict}
            wins += int(win)
            done += 1
            if done % 50 == 0 or done == len(rows):
                print(f"    {done}/{len(rows)} judged  win_rate={wins/done:.1%}", flush=True)

    valid    = [v for v in verdicts if v is not None and v["verdict"] is not None]
    n_valid  = len(valid)
    n_wins   = sum(1 for v in valid if v["win"])
    win_rate = n_wins / n_valid if n_valid else 0.0

    # 95% CI via Wilson score
    if n_valid > 0:
        z   = 1.96
        p   = win_rate
        ci  = z * (p * (1 - p) / n_valid) ** 0.5
    else:
        ci = 0.0

    return {
        "model":     model_name,
        "n_eval":    n_valid,
        "wins":      n_wins,
        "win_rate":  win_rate,
        "ci_95":     ci,
        "verdicts":  verdicts,
    }


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_alpacaeval(results: dict[str, dict], save_path: str):
    models    = [m for m in results if m != "base"]
    win_rates = [results[m]["win_rate"] * 100 for m in models]
    cis       = [results[m]["ci_95"] * 100      for m in models]
    colors    = ["#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [m.replace("_", " ").title() for m in models],
        win_rates, yerr=cis, capsize=6,
        color=colors[:len(models)], alpha=0.85, width=0.4
    )
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% (tie vs base)")
    for bar, v, ci in zip(bars, win_rates, cis):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 1,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Win Rate vs Base Model (%)")
    ax.set_title("AlpacaEval Win Rate (GPT-4o Judge — held-out)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved figure → {save_path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",  default="./eval_outputs")
    parser.add_argument("--results-dir", default="./eval_results")
    parser.add_argument("--n-eval", type=int, default=805, help="Number of prompts to judge")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)

    base_path = f"{args.output_dir}/alpaca_base.jsonl"
    if not os.path.exists(base_path):
        raise SystemExit(f"ERROR: Base model outputs not found: {base_path}\n"
                         "Run generate_outputs.py first (or skip --skip-base).")
    base_responses = load_base_responses(base_path)
    print(f"Loaded {len(base_responses)} base model responses.")

    model_files = {
        "indirect": f"{args.output_dir}/alpaca_indirect.jsonl",
        "direct":   f"{args.output_dir}/alpaca_direct.jsonl",
    }

    all_results = {}
    for model_name, path in model_files.items():
        if not os.path.exists(path):
            print(f"  Skipping {model_name} — file not found: {path}")
            continue
        print(f"\nEvaluating {model_name} vs base (GPT-4o judge)...")
        rows   = load_jsonl(path)
        result = evaluate_model(client, rows, base_responses, model_name, args.n_eval, args.workers)
        all_results[model_name] = result

    if not all_results:
        print("No results to display.")
        return

    # ── Print table ───────────────────────────────────────────────────────
    header = f"{'Model':<12} {'Win Rate':>10} {'95% CI':>8} {'Wins':>7} {'N':>6}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print("AlpacaEval Results (GPT-4o judge, vs base model)")
    print(sep)
    print(header)
    print(sep)
    for model, r in all_results.items():
        print(
            f"{model:<12} "
            f"{r['win_rate']*100:>9.1f}% "
            f"±{r['ci_95']*100:>5.1f}% "
            f"{r['wins']:>7} "
            f"{r['n_eval']:>6}"
        )
    print(sep)

    # ── Save results ──────────────────────────────────────────────────────
    save_data = {
        m: {k: v for k, v in r.items() if k != "verdicts"}
        for m, r in all_results.items()
    }
    results_path = f"{args.results_dir}/alpacaeval_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved → {results_path}")

    verdicts_path = f"{args.results_dir}/alpacaeval_verdicts.json"
    with open(verdicts_path, "w") as f:
        json.dump({m: r["verdicts"] for m, r in all_results.items()}, f)
    print(f"Verdicts saved → {verdicts_path}")

    table_path = f"{args.results_dir}/alpacaeval_table.txt"
    with open(table_path, "w") as f:
        f.write(sep + "\nAlpacaEval Results (GPT-4o judge, vs base model)\n" + sep + "\n")
        f.write(header + "\n" + sep + "\n")
        for model, r in all_results.items():
            f.write(
                f"{model:<12} "
                f"{r['win_rate']*100:>9.1f}% "
                f"±{r['ci_95']*100:>5.1f}% "
                f"{r['wins']:>7} "
                f"{r['n_eval']:>6}\n"
            )
        f.write(sep + "\n")
    print(f"Table saved → {table_path}")

    plot_alpacaeval(all_results, f"{args.results_dir}/alpacaeval_bar.png")


if __name__ == "__main__":
    main()
