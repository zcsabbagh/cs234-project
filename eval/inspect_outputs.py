"""
Sample 50-100 outputs from each model and flag degenerate patterns.

Checks for:
  - Sycophantic openers ("Great question!", "Certainly!", "Of course!", etc.)
  - Verbosity: response length relative to base model
  - Bullet-point overuse (many bullet points when prose would do)
  - Repetitive structure across diverse prompts

Outputs:
  results/inspection_summary.json   counts + flagged examples
  results/inspection_summary.txt    formatted text report (also printed)
  results/inspection_lengths.png    response length distribution comparison
  results/inspection_patterns.png   degenerate pattern frequency bar chart
  results/inspection_report.html    human-readable HTML for manual review

Usage:
    python eval/inspect_outputs.py \
        --output-dir ./eval_outputs \
        --results-dir ./eval_results \
        --n-samples 75
"""

import argparse
import json
import os
import re
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Pattern definitions ───────────────────────────────────────────────────

SYCOPHANTIC_OPENERS = [
    r"^great question",
    r"^certainly[!,]",
    r"^of course[!,]",
    r"^absolutely[!,]",
    r"^sure[!,]",
    r"^happy to help",
    r"^i'd be happy",
    r"^i'd be glad",
    r"^thanks? for (asking|your question)",
    r"^what an? (interesting|great|wonderful|excellent)",
    r"^that'?s? (a )?(great|good|excellent|wonderful|interesting) question",
]
SYCOPHANTIC_RE = re.compile(
    "|".join(SYCOPHANTIC_OPENERS), re.IGNORECASE
)

FILLER_PHRASES = [
    r"in conclusion,",
    r"to summarize,",
    r"in summary,",
    r"as i mentioned (above|earlier|before),",
    r"it's (important|worth) (to )?(note|mention)",
    r"i hope (this|that) (helps|answers)",
    r"feel free to ask",
    r"please let me know if",
    r"don't hesitate to",
]
FILLER_RE = re.compile("|".join(FILLER_PHRASES), re.IGNORECASE)


def count_bullet_points(text: str) -> int:
    lines = text.split("\n")
    return sum(1 for l in lines if re.match(r"^\s*[-*•]\s", l))


def analyze_response(response: str) -> dict:
    resp_lower = response.lower().strip()
    return {
        "length_chars":   len(response),
        "length_words":   len(response.split()),
        "is_sycophantic": bool(SYCOPHANTIC_RE.match(resp_lower)),
        "filler_count":   len(FILLER_RE.findall(response)),
        "bullet_count":   count_bullet_points(response),
        "bullet_heavy":   count_bullet_points(response) >= 5,
        "is_very_long":   len(response.split()) > 400,
        "is_very_short":  len(response.split()) < 20,
    }


# ── Load outputs ─────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_lengths(samples: dict[str, list[dict]], save_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"base": "#4C72B0", "indirect": "#DD8452", "direct": "#55A868"}

    for model, rows in samples.items():
        lengths = [r["analysis"]["length_words"] for r in rows]
        ax.hist(lengths, bins=30, alpha=0.5, label=model.title(),
                color=colors.get(model, "gray"), density=True)
        ax.axvline(np.mean(lengths), color=colors.get(model, "gray"),
                   linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Response Length (words)")
    ax.set_ylabel("Density")
    ax.set_title("Response Length Distribution by Model")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_patterns(pattern_counts: dict[str, dict], save_path: str):
    models   = list(pattern_counts.keys())
    patterns = ["sycophantic", "filler_phrases", "bullet_heavy", "very_long", "very_short"]
    labels   = ["Sycophantic\nOpener", "Filler\nPhrases", "Bullet\nHeavy (≥5)", "Very\nLong (>400w)", "Very\nShort (<20w)"]
    colors   = ["#4C72B0", "#DD8452", "#55A868"]

    x     = np.arange(len(patterns))
    width = 0.25
    n     = len(models)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (model, offset) in enumerate(zip(models, offsets)):
        vals = [pattern_counts[model][p] * 100 for p in patterns]
        bars = ax.bar(x + offset, vals, width,
                      label=model.title(), color=colors[i % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Degenerate Pattern")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Degenerate Output Pattern Frequency by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


# ── HTML report ───────────────────────────────────────────────────────────

def make_html(samples: dict[str, list[dict]], n_show: int, save_path: str):
    html = ["""<!DOCTYPE html><html><head>
<style>
  body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
  h1 { color: #333; }
  .prompt { background: #f0f4f8; padding: 10px; border-radius: 5px; margin: 10px 0; font-style: italic; }
  .response { background: #fff; border: 1px solid #ddd; padding: 10px; border-radius: 5px; white-space: pre-wrap; }
  .flag { background: #fff3cd; border: 1px solid #ffc107; padding: 3px 8px; border-radius: 3px;
          font-size: 12px; margin: 2px; display: inline-block; }
  .model-header { background: #343a40; color: white; padding: 8px 15px; border-radius: 5px; margin: 20px 0 5px; }
  .pair { border-bottom: 2px solid #eee; padding: 15px 0; }
  table { border-collapse: collapse; width: 100%; margin: 20px 0; }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
  th { background: #f8f9fa; }
</style>
</head><body>
<h1>Manual Output Inspection Report</h1>"""]

    # Summary stats table
    html.append("<h2>Summary Statistics</h2><table>")
    html.append("<tr><th>Model</th><th>Avg Words</th><th>Sycophantic %</th>"
                "<th>Bullet-Heavy %</th><th>Filler %</th><th>Very Long %</th></tr>")
    for model, rows in samples.items():
        analyses = [r["analysis"] for r in rows]
        n = len(analyses)
        html.append(
            f"<tr><td>{model.title()}</td>"
            f"<td>{np.mean([a['length_words'] for a in analyses]):.0f}</td>"
            f"<td>{sum(a['is_sycophantic'] for a in analyses)/n*100:.0f}%</td>"
            f"<td>{sum(a['bullet_heavy'] for a in analyses)/n*100:.0f}%</td>"
            f"<td>{sum(a['filler_count']>0 for a in analyses)/n*100:.0f}%</td>"
            f"<td>{sum(a['is_very_long'] for a in analyses)/n*100:.0f}%</td></tr>"
        )
    html.append("</table>")

    # Sample outputs (use same prompts across models for fair comparison)
    model_list = list(samples.keys())
    first_model = model_list[0]
    n_pairs = min(n_show, len(samples[first_model]))

    html.append("<h2>Sample Outputs (same prompts across models)</h2>")
    for i in range(n_pairs):
        html.append(f'<div class="pair"><h3>Prompt {i+1}</h3>')
        instruction = samples[first_model][i]["instruction"]
        html.append(f'<div class="prompt">{instruction}</div>')

        for model in model_list:
            if i >= len(samples[model]):
                continue
            row      = samples[model][i]
            analysis = row["analysis"]
            response = row["response"]

            flags = []
            if analysis["is_sycophantic"]:
                flags.append('<span class="flag">⚠ Sycophantic opener</span>')
            if analysis["bullet_heavy"]:
                flags.append(f'<span class="flag">⚠ Bullet-heavy ({analysis["bullet_count"]} bullets)</span>')
            if analysis["filler_count"] > 0:
                flags.append(f'<span class="flag">⚠ Filler phrases ({analysis["filler_count"]})</span>')
            if analysis["is_very_long"]:
                flags.append('<span class="flag">⚠ Very long (>400 words)</span>')

            flags_html = " ".join(flags) if flags else '<span style="color:green">✓ No flags</span>'
            html.append(
                f'<div class="model-header">{model.title()} '
                f'<small>({analysis["length_words"]} words)</small></div>'
                f'{flags_html}'
                f'<div class="response">{response[:1500]}{"..." if len(response) > 1500 else ""}</div>'
            )
        html.append("</div>")

    html.append("</body></html>")
    with open(save_path, "w") as f:
        f.write("\n".join(html))
    print(f"  Saved HTML report → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",  default="./eval_outputs")
    parser.add_argument("--results-dir", default="./eval_results")
    parser.add_argument("--n-samples",   type=int, default=75)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    random.seed(args.seed)

    model_files = {
        "base":     f"{args.output_dir}/alpaca_base.jsonl",
        "indirect": f"{args.output_dir}/alpaca_indirect.jsonl",
        "direct":   f"{args.output_dir}/alpaca_direct.jsonl",
    }

    # Load and sample — use the same indices across models for fair comparison
    all_rows: dict[str, list[dict]] = {}
    for model, path in model_files.items():
        if not os.path.exists(path):
            print(f"  Skipping {model} — not found: {path}")
            continue
        rows = load_jsonl(path)
        all_rows[model] = rows

    if not all_rows:
        print("No output files found.")
        return

    # Align sample indices to the smallest available set
    n_available = min(len(rows) for rows in all_rows.values())
    n_samples   = min(args.n_samples, n_available)
    indices     = sorted(random.sample(range(n_available), n_samples))

    samples: dict[str, list[dict]] = {}
    for model, rows in all_rows.items():
        selected = [rows[i] for i in indices]
        for row in selected:
            row["analysis"] = analyze_response(row["response"])
        samples[model] = selected

    # ── Compute pattern counts ────────────────────────────────────────────
    pattern_counts: dict[str, dict] = {}
    for model, rows in samples.items():
        n = len(rows)
        pattern_counts[model] = {
            "sycophantic":  sum(r["analysis"]["is_sycophantic"] for r in rows) / n,
            "filler_phrases": sum(r["analysis"]["filler_count"] > 0 for r in rows) / n,
            "bullet_heavy":  sum(r["analysis"]["bullet_heavy"] for r in rows) / n,
            "very_long":     sum(r["analysis"]["is_very_long"] for r in rows) / n,
            "very_short":    sum(r["analysis"]["is_very_short"] for r in rows) / n,
        }

    # ── Print summary ─────────────────────────────────────────────────────
    header = f"{'Model':<12} {'Avg Words':>10} {'Syco%':>7} {'Bullets%':>9} {'Filler%':>8} {'VeryLong%':>10}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(f"Output Inspection Summary (n={n_samples} samples per model)")
    print(sep)
    print(header)
    print(sep)
    for model, rows in samples.items():
        analyses  = [r["analysis"] for r in rows]
        n         = len(analyses)
        avg_words = np.mean([a["length_words"] for a in analyses])
        pc        = pattern_counts[model]
        print(
            f"{model:<12} "
            f"{avg_words:>9.0f} "
            f"{pc['sycophantic']*100:>6.0f}% "
            f"{pc['bullet_heavy']*100:>8.0f}% "
            f"{pc['filler_phrases']*100:>7.0f}% "
            f"{pc['very_long']*100:>9.0f}%"
        )
    print(sep)

    # Flag interesting examples
    print("\nFlagged examples (sycophantic or bullet-heavy):")
    for model, rows in samples.items():
        flagged = [r for r in rows if r["analysis"]["is_sycophantic"] or r["analysis"]["bullet_heavy"]]
        print(f"\n  {model.upper()} — {len(flagged)}/{n_samples} flagged")
        for r in flagged[:3]:
            flags = []
            if r["analysis"]["is_sycophantic"]: flags.append("SYCO")
            if r["analysis"]["bullet_heavy"]:   flags.append("BULLETS")
            print(f"    [{','.join(flags)}] {r['response'][:120]}...")

    # ── Save results ──────────────────────────────────────────────────────
    summary = {
        model: {
            "n_samples":      len(rows),
            "avg_words":      float(np.mean([r["analysis"]["length_words"] for r in rows])),
            "avg_chars":      float(np.mean([r["analysis"]["length_chars"] for r in rows])),
            "pattern_rates":  pattern_counts[model],
        }
        for model, rows in samples.items()
    }
    results_path = f"{args.results_dir}/inspection_summary.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {results_path}")

    txt_path = f"{args.results_dir}/inspection_summary.txt"
    with open(txt_path, "w") as f:
        f.write(sep + f"\nOutput Inspection (n={n_samples})\n" + sep + "\n")
        f.write(header + "\n" + sep + "\n")
        for model, rows in samples.items():
            analyses  = [r["analysis"] for r in rows]
            avg_words = np.mean([a["length_words"] for a in analyses])
            pc        = pattern_counts[model]
            f.write(
                f"{model:<12} {avg_words:>9.0f} "
                f"{pc['sycophantic']*100:>6.0f}% "
                f"{pc['bullet_heavy']*100:>8.0f}% "
                f"{pc['filler_phrases']*100:>7.0f}% "
                f"{pc['very_long']*100:>9.0f}%\n"
            )
        f.write(sep + "\n")
    print(f"Table saved → {txt_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_lengths(samples,  f"{args.results_dir}/inspection_lengths.png")
    plot_patterns(pattern_counts, f"{args.results_dir}/inspection_patterns.png")
    make_html(samples, n_show=50, save_path=f"{args.results_dir}/inspection_report.html")


if __name__ == "__main__":
    main()
