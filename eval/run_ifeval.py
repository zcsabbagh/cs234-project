"""
Score IFEval outputs using the official google-deepmind/ifeval package.

Reads *_ifeval_*.jsonl files from --output-dir and computes:
  - Prompt-level strict accuracy
  - Prompt-level loose accuracy
  - Instruction-level strict accuracy
  - Instruction-level loose accuracy

Outputs:
  results/ifeval_results.json    raw numbers
  results/ifeval_table.txt       formatted table (also printed)
  results/ifeval_bar.png         bar chart figure

Usage:
    pip install ifeval
    python eval/run_ifeval.py --output-dir ./eval_outputs --results-dir ./eval_results
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── IFEval evaluation ─────────────────────────────────────────────────────

def evaluate_ifeval(rows: list[dict]) -> dict:
    """
    Evaluate IFEval responses using instruction_following_eval package.
    Each row must have: prompt, instruction_id_list, kwargs, response.
    Returns dict with prompt_strict, prompt_loose, instr_strict, instr_loose.
    """
    try:
        from ifeval.instructions_registry import INSTRUCTION_DICT
    except ImportError:
        raise SystemExit(
            "ERROR: ifeval not installed.\n"
            "Run: pip install ifeval"
        )

    prompt_strict_correct  = 0
    prompt_loose_correct   = 0
    instr_strict_total     = 0
    instr_strict_correct   = 0
    instr_loose_total      = 0
    instr_loose_correct    = 0

    for row in rows:
        response          = row.get("response", "")
        instruction_ids   = row.get("instruction_id_list", [])
        kwargs_list       = row.get("kwargs", [])

        # Pad kwargs if shorter than instruction_ids
        while len(kwargs_list) < len(instruction_ids):
            kwargs_list.append({})

        strict_results = []
        loose_results  = []

        for instr_id, kwargs in zip(instruction_ids, kwargs_list):
            if instr_id not in INSTRUCTION_DICT:
                continue
            instr_cls = INSTRUCTION_DICT[instr_id]
            kwargs    = {k: v for k, v in kwargs.items() if v is not None}

            try:
                instr_obj = instr_cls(instr_id)
                instr_obj.build_description(**kwargs)

                strict = instr_obj.check_following(response)
                # Loose: normalize whitespace/case for some checks
                loose_resp = " ".join(response.split()).lower()
                loose  = instr_obj.check_following(loose_resp)

                strict_results.append(strict)
                loose_results.append(loose)

                instr_strict_total   += 1
                instr_loose_total    += 1
                instr_strict_correct += int(strict)
                instr_loose_correct  += int(loose)
            except Exception:
                continue

        if strict_results:
            prompt_strict_correct += int(all(strict_results))
        if loose_results:
            prompt_loose_correct  += int(all(loose_results))

    n = len(rows)
    return {
        "prompt_strict":  prompt_strict_correct  / n if n else 0.0,
        "prompt_loose":   prompt_loose_correct   / n if n else 0.0,
        "instr_strict":   instr_strict_correct   / instr_strict_total  if instr_strict_total  else 0.0,
        "instr_loose":    instr_loose_correct    / instr_loose_total   if instr_loose_total   else 0.0,
        "n_prompts":      n,
        "n_instructions": instr_strict_total,
    }


# ── Load outputs ─────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_ifeval(results: dict[str, dict], save_path: str):
    models   = list(results.keys())
    metrics  = ["prompt_strict", "prompt_loose", "instr_strict", "instr_loose"]
    labels   = ["Prompt\nStrict", "Prompt\nLoose", "Instr\nStrict", "Instr\nLoose"]
    colors   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x     = np.arange(len(metrics))
    width = 0.25
    n     = len(models)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model, offset) in enumerate(zip(models, offsets)):
        vals = [results[model][m] * 100 for m in metrics]
        bars = ax.bar(x + offset, vals, width, label=model.replace("_", " ").title(),
                      color=colors[i % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("IFEval Results: Base vs Indirect vs Direct GRPO")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
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
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    model_files = {
        "base":     f"{args.output_dir}/ifeval_base.jsonl",
        "indirect": f"{args.output_dir}/ifeval_indirect.jsonl",
        "direct":   f"{args.output_dir}/ifeval_direct.jsonl",
    }

    all_results = {}
    for model_name, path in model_files.items():
        if not os.path.exists(path):
            print(f"  Skipping {model_name} — file not found: {path}")
            continue
        print(f"Evaluating {model_name}...")
        rows   = load_jsonl(path)
        result = evaluate_ifeval(rows)
        all_results[model_name] = result
        print(f"  n_prompts={result['n_prompts']}  n_instructions={result['n_instructions']}")

    if not all_results:
        print("No results to display.")
        return

    # ── Print table ───────────────────────────────────────────────────────
    header = f"{'Model':<12} {'Prompt Strict':>14} {'Prompt Loose':>13} {'Instr Strict':>13} {'Instr Loose':>12}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print("IFEval Results")
    print(sep)
    print(header)
    print(sep)
    for model, r in all_results.items():
        print(
            f"{model:<12} "
            f"{r['prompt_strict']*100:>13.1f}% "
            f"{r['prompt_loose']*100:>12.1f}% "
            f"{r['instr_strict']*100:>12.1f}% "
            f"{r['instr_loose']*100:>11.1f}%"
        )
    print(sep)

    # ── Save results ──────────────────────────────────────────────────────
    results_path = f"{args.results_dir}/ifeval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    table_path = f"{args.results_dir}/ifeval_table.txt"
    with open(table_path, "w") as f:
        f.write(sep + "\nIFEval Results\n" + sep + "\n" + header + "\n" + sep + "\n")
        for model, r in all_results.items():
            f.write(
                f"{model:<12} "
                f"{r['prompt_strict']*100:>13.1f}% "
                f"{r['prompt_loose']*100:>12.1f}% "
                f"{r['instr_strict']*100:>12.1f}% "
                f"{r['instr_loose']*100:>11.1f}%\n"
            )
        f.write(sep + "\n")
    print(f"Table saved → {table_path}")

    plot_ifeval(all_results, f"{args.results_dir}/ifeval_bar.png")


if __name__ == "__main__":
    main()
