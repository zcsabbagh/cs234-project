"""
Score IFEval outputs — fully self-contained, no external eval package needed.

Implements all major instruction types from the IFEval benchmark natively:
  keywords, length_constraints, detectable_format, detectable_content,
  startend, change_case, combination, punctuation.

Outputs:
  results/ifeval_results.json    raw numbers
  results/ifeval_table.txt       formatted table (also printed)
  results/ifeval_bar.png         bar chart figure

Usage:
    python eval/run_ifeval.py --output-dir ./eval_outputs --results-dir ./eval_results
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Native IFEval instruction checkers ────────────────────────────────────

def _count_words(text: str) -> int:
    return len(text.split())

def _count_sentences(text: str) -> int:
    return len([s for s in re.split(r'[.!?]+', text) if s.strip()])

def _count_paragraphs(text: str) -> int:
    return len([p for p in re.split(r'\n\s*\n', text) if p.strip()])

def _relation(count: int, target: int, rel: str) -> bool:
    rel = rel.lower()
    if rel in ("at least", ">="):  return count >= target
    if rel in ("at most",  "<="):  return count <= target
    if rel in ("exactly",  "==" ): return count == target
    if rel in ("less than", "<"):  return count < target
    if rel in ("more than", ">"):  return count > target
    return count >= target  # default


def check_instruction(instr_id: str, kwargs: dict, response: str, loose: bool = False) -> bool | None:
    """
    Returns True/False if the instruction can be evaluated, None if unknown.
    `loose` mode normalises whitespace/case for forgiving checks.
    """
    resp = " ".join(response.split()) if loose else response
    kw   = {k: v for k, v in kwargs.items() if v is not None}

    # ── keywords ──────────────────────────────────────────────────────────
    if instr_id == "keywords:include_keywords":
        keywords = kw.get("keywords", [])
        return all(k.lower() in resp.lower() for k in keywords)

    if instr_id == "keywords:forbidden_words":
        forbidden = kw.get("forbidden_words", [])
        return not any(f.lower() in resp.lower() for f in forbidden)

    if instr_id == "keywords:frequency":
        keyword   = kw.get("keyword", "")
        frequency = int(kw.get("frequency", 1))
        relation  = kw.get("relation", "at least")
        count     = len(re.findall(re.escape(keyword), resp, re.IGNORECASE))
        return _relation(count, frequency, relation)

    if instr_id == "keywords:letter_frequency":
        letter     = kw.get("letter", "")
        target     = int(kw.get("let_frequency", 1))
        relation   = kw.get("let_relation", "at least")
        count      = resp.lower().count(letter.lower())
        return _relation(count, target, relation)

    # ── length_constraints ────────────────────────────────────────────────
    if instr_id == "length_constraints:number_words":
        target   = int(kw.get("num_words", 100))
        relation = kw.get("relation", "at least")
        return _relation(_count_words(resp), target, relation)

    if instr_id == "length_constraints:number_sentences":
        target   = int(kw.get("num_sentences", 5))
        relation = kw.get("relation", "at least")
        return _relation(_count_sentences(resp), target, relation)

    if instr_id == "length_constraints:number_paragraphs":
        target   = int(kw.get("num_paragraphs", 3))
        relation = kw.get("relation", "at least")
        return _relation(_count_paragraphs(resp), target, relation)

    if instr_id == "length_constraints:nth_paragraph_first_word":
        # Check that the Nth paragraph starts with a specific word
        nth  = int(kw.get("nth_paragraph", 1))
        word = kw.get("first_word", "")
        paras = [p.strip() for p in re.split(r'\n\s*\n', resp) if p.strip()]
        if nth > len(paras):
            return False
        return paras[nth - 1].split()[0].lower().rstrip(".,!?") == word.lower() if paras[nth-1].split() else False

    # ── detectable_content ────────────────────────────────────────────────
    if instr_id == "detectable_content:number_placeholders":
        target = int(kw.get("num_placeholders", 1))
        count  = len(re.findall(r'\[[^\]]+\]', resp))
        return count >= target

    if instr_id == "detectable_content:postscript":
        marker = kw.get("postscript_marker", "P.S.")
        return marker.lower() in resp.lower()

    # ── detectable_format ─────────────────────────────────────────────────
    if instr_id == "detectable_format:number_bullet_lists":
        target = int(kw.get("num_bullets", 3))
        count  = len([l for l in resp.split("\n") if re.match(r'^\s*[-*•]\s', l)])
        return count >= target

    if instr_id == "detectable_format:number_highlighted_sections":
        target = int(kw.get("num_highlights", 1))
        count  = len(re.findall(r'\*\*[^*]+\*\*', resp))
        return count >= target

    if instr_id == "detectable_format:multiple_sections":
        splitter     = kw.get("section_spliter", "Section")
        num_sections = int(kw.get("num_sections", 3))
        count        = len(re.findall(re.escape(splitter), resp, re.IGNORECASE))
        return count >= num_sections

    if instr_id == "detectable_format:json_format":
        stripped = resp.strip()
        try:
            json.loads(stripped)
            return True
        except Exception:
            m = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', stripped)
            if m:
                try:
                    json.loads(m.group())
                    return True
                except Exception:
                    pass
            return False

    if instr_id == "detectable_format:title":
        return bool(re.search(r'<<[^>]+>>', resp))

    if instr_id == "detectable_format:constrained_response":
        # Response should be a single short constrained answer
        stripped = resp.strip().lower().rstrip(".!?")
        return len(stripped.split()) <= 5

    # ── language ──────────────────────────────────────────────────────────
    if instr_id == "language:response_language":
        # Skip — language detection requires langdetect which may not be installed
        return None

    # ── startend ──────────────────────────────────────────────────────────
    if instr_id == "startend:end_checker":
        end_phrase = kw.get("end_phrase", "")
        return resp.strip().lower().endswith(end_phrase.strip().lower())

    if instr_id == "startend:quotation":
        s = resp.strip()
        return s.startswith('"') and s.endswith('"')

    # ── change_case ───────────────────────────────────────────────────────
    if instr_id == "change_case:english_capital":
        letters = [c for c in resp if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    if instr_id == "change_case:english_lowercase":
        letters = [c for c in resp if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)

    if instr_id == "change_case:capital_word_frequency":
        target   = int(kw.get("capital_frequency", 1))
        relation = kw.get("capital_relation", "at least")
        count    = sum(1 for w in resp.split() if w and w[0].isupper())
        return _relation(count, target, relation)

    # ── combination ───────────────────────────────────────────────────────
    if instr_id == "combination:repeat_prompt":
        prompt = kw.get("prompt_to_repeat", "")
        return prompt.lower() in resp.lower() if prompt else False

    if instr_id == "combination:two_responses":
        return bool(re.search(r'\*{3,}|---+|\n{3,}', resp))

    # ── punctuation ───────────────────────────────────────────────────────
    if instr_id == "punctuation:no_comma":
        return "," not in resp

    return None  # instruction type not implemented — skip


# ── Evaluate ──────────────────────────────────────────────────────────────

def evaluate_ifeval(rows: list[dict]) -> dict:
    prompt_strict = prompt_loose = 0
    instr_strict_num = instr_strict_den = 0
    instr_loose_num  = instr_loose_den  = 0
    skipped_types: set[str] = set()

    for row in rows:
        response     = row.get("response", "")
        instr_ids    = row.get("instruction_id_list", [])
        kwargs_list  = row.get("kwargs", []) or []

        while len(kwargs_list) < len(instr_ids):
            kwargs_list.append({})

        row_strict, row_loose = [], []

        for instr_id, kwargs in zip(instr_ids, kwargs_list):
            strict = check_instruction(instr_id, kwargs or {}, response, loose=False)
            loose  = check_instruction(instr_id, kwargs or {}, response, loose=True)

            if strict is None:
                skipped_types.add(instr_id)
                continue

            row_strict.append(strict)
            row_loose.append(loose if loose is not None else strict)

            instr_strict_den += 1
            instr_loose_den  += 1
            instr_strict_num += int(strict)
            instr_loose_num  += int(loose if loose is not None else strict)

        if row_strict:
            prompt_strict += int(all(row_strict))
        if row_loose:
            prompt_loose  += int(all(row_loose))

    n = len(rows)
    if skipped_types:
        print(f"  Skipped instruction types (not implemented): {', '.join(sorted(skipped_types))}")

    return {
        "prompt_strict":  prompt_strict  / n if n else 0.0,
        "prompt_loose":   prompt_loose   / n if n else 0.0,
        "instr_strict":   instr_strict_num / instr_strict_den if instr_strict_den else 0.0,
        "instr_loose":    instr_loose_num  / instr_loose_den  if instr_loose_den  else 0.0,
        "n_prompts":      n,
        "n_instructions": instr_strict_den,
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def plot_ifeval(results: dict[str, dict], save_path: str):
    models  = list(results.keys())
    metrics = ["prompt_strict", "prompt_loose", "instr_strict", "instr_loose"]
    labels  = ["Prompt\nStrict", "Prompt\nLoose", "Instr\nStrict", "Instr\nLoose"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x       = np.arange(len(metrics))
    width   = 0.25
    n       = len(models)
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
    ax.set_title("IFEval: Base vs Indirect vs Direct GRPO")
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
        print(f"  n_prompts={result['n_prompts']}  n_instr={result['n_instructions']}")

    if not all_results:
        print("No results to display.")
        return

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
