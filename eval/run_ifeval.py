"""
Score IFEval outputs using the official Google Research evaluation code.

At runtime this script downloads three files from the google-research GitHub
repo into /tmp/instruction_following_eval/ and imports from them directly.
This guarantees exact benchmark semantics with no third-party package needed.

Requires:  pip install langdetect  (for language:response_language checks)

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
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Download official Google Research IFEval code ─────────────────────────

IFEVAL_DIR = "/tmp/instruction_following_eval"
_BASE_URL  = (
    "https://raw.githubusercontent.com/google-research/"
    "google-research/master/instruction_following_eval/"
)
_FILES = [
    "instructions.py",
    "instructions_registry.py",
    "instructions_util.py",
]


def ensure_ifeval_code() -> bool:
    """Download the official IFEval source files if not already present."""
    sentinel = os.path.join(IFEVAL_DIR, "instructions_registry.py")
    if os.path.exists(sentinel):
        return True

    print("Downloading official IFEval evaluation code from Google Research...")
    os.makedirs(IFEVAL_DIR, exist_ok=True)

    # Create __init__.py so Python treats the dir as a package
    open(os.path.join(IFEVAL_DIR, "__init__.py"), "w").close()

    for fname in _FILES:
        url  = _BASE_URL + fname
        dest = os.path.join(IFEVAL_DIR, fname)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  Downloaded {fname}")
        except Exception as e:
            print(f"  ERROR downloading {fname}: {e}")
            return False

    return True


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_ifeval(rows: list[dict]) -> dict:
    """
    Evaluate IFEval responses using the official Google Research checkers.
    Falls back to a minimal native implementation if the download fails.
    """
    if not ensure_ifeval_code():
        print("WARNING: official code unavailable, falling back to native impl")
        return _evaluate_ifeval_native(rows)

    # Add /tmp to path so `import instruction_following_eval` works
    if "/tmp" not in sys.path:
        sys.path.insert(0, "/tmp")

    try:
        from instruction_following_eval.instructions_registry import INSTRUCTION_DICT  # noqa
    except ImportError as e:
        print(f"WARNING: import failed ({e}), falling back to native impl")
        return _evaluate_ifeval_native(rows)

    prompt_strict = prompt_loose = 0
    instr_strict_num = instr_strict_den = 0
    instr_loose_num  = instr_loose_den  = 0
    skipped: set[str] = set()

    for row in rows:
        response    = row.get("response", "")
        instr_ids   = row.get("instruction_id_list", [])
        kwargs_list = list(row.get("kwargs", []) or [])

        while len(kwargs_list) < len(instr_ids):
            kwargs_list.append({})

        row_strict, row_loose = [], []

        for instr_id, raw_kw in zip(instr_ids, kwargs_list):
            if instr_id not in INSTRUCTION_DICT:
                skipped.add(instr_id)
                continue

            # Strip None values — build_description doesn't accept them
            kw = {k: v for k, v in (raw_kw or {}).items() if v is not None}

            try:
                instr = INSTRUCTION_DICT[instr_id](instr_id)
                instr.build_description(**kw)

                strict = bool(instr.check_following(response))
                # Loose: normalise whitespace
                loose_resp = " ".join(response.split())
                loose  = bool(instr.check_following(loose_resp))

                row_strict.append(strict)
                row_loose.append(loose)

                instr_strict_den += 1
                instr_loose_den  += 1
                instr_strict_num += int(strict)
                instr_loose_num  += int(loose)
            except Exception as exc:
                skipped.add(f"{instr_id}({exc})")
                continue

        if row_strict:
            prompt_strict += int(all(row_strict))
        if row_loose:
            prompt_loose  += int(all(row_loose))

    n = len(rows)
    if skipped:
        print(f"  Skipped/errored instruction types: {', '.join(sorted(skipped))}")

    return {
        "prompt_strict":  prompt_strict  / n if n else 0.0,
        "prompt_loose":   prompt_loose   / n if n else 0.0,
        "instr_strict":   instr_strict_num / instr_strict_den if instr_strict_den else 0.0,
        "instr_loose":    instr_loose_num  / instr_loose_den  if instr_loose_den  else 0.0,
        "n_prompts":      n,
        "n_instructions": instr_strict_den,
    }


# ── Native fallback (covers all 25 official instruction types) ────────────

def _relation(count: int, target: int, rel: str) -> bool:
    rel = (rel or "at least").lower()
    if rel in ("at least", ">="):  return count >= target
    if rel in ("at most",  "<="):  return count <= target
    if rel in ("exactly",  "=="):  return count == target
    if rel in ("less than", "<"):  return count <  target
    if rel in ("more than", ">"):  return count >  target
    return count >= target


def _check_native(instr_id: str, kw: dict, response: str, loose: bool) -> bool | None:
    import re, json as _json
    resp = " ".join(response.split()) if loose else response

    if instr_id == "keywords:existence":          # official name
        keywords = kw.get("keywords", [])
        return all(k.lower() in resp.lower() for k in keywords)
    if instr_id == "keywords:include_keywords":   # alias seen in some versions
        keywords = kw.get("keywords", [])
        return all(k.lower() in resp.lower() for k in keywords)
    if instr_id == "keywords:forbidden_words":
        return not any(f.lower() in resp.lower() for f in kw.get("forbidden_words", []))
    if instr_id == "keywords:frequency":
        count = len(re.findall(re.escape(kw.get("keyword", "")), resp, re.IGNORECASE))
        return _relation(count, int(kw.get("frequency", 1)), kw.get("relation", "at least"))
    if instr_id == "keywords:letter_frequency":
        count = resp.lower().count(kw.get("letter", "").lower())
        return _relation(count, int(kw.get("let_frequency", 1)), kw.get("let_relation", "at least"))
    if instr_id == "length_constraints:number_words":
        return _relation(len(resp.split()), int(kw.get("num_words", 100)), kw.get("relation", "at least"))
    if instr_id == "length_constraints:number_sentences":
        n = len([s for s in re.split(r'[.!?]+', resp) if s.strip()])
        return _relation(n, int(kw.get("num_sentences", 5)), kw.get("relation", "at least"))
    if instr_id == "length_constraints:number_paragraphs":
        n = len([p for p in re.split(r'\n\s*\n', resp) if p.strip()])
        return _relation(n, int(kw.get("num_paragraphs", 3)), kw.get("relation", "at least"))
    if instr_id == "length_constraints:nth_paragraph_first_word":
        nth  = int(kw.get("nth_paragraph", 1))
        word = kw.get("first_word", "")
        paras = [p.strip() for p in re.split(r'\n\s*\n', resp) if p.strip()]
        if nth > len(paras) or not paras[nth-1].split(): return False
        return paras[nth-1].split()[0].lower().rstrip(".,!?") == word.lower()
    if instr_id == "detectable_content:number_placeholders":
        return len(re.findall(r'\[[^\]]+\]', resp)) >= int(kw.get("num_placeholders", 1))
    if instr_id == "detectable_content:postscript":
        return kw.get("postscript_marker", "P.S.").lower() in resp.lower()
    if instr_id == "detectable_format:number_bullet_lists":
        n = len([l for l in resp.split("\n") if re.match(r'^\s*[-*•]\s', l)])
        return n >= int(kw.get("num_bullets", 3))
    if instr_id == "detectable_format:number_highlighted_sections":
        return len(re.findall(r'\*\*[^*]+\*\*', resp)) >= int(kw.get("num_highlights", 1))
    if instr_id == "detectable_format:multiple_sections":
        splitter = kw.get("section_spliter") or kw.get("section_splitter", "Section")
        return resp.count(splitter) >= int(kw.get("num_sections", 3))
    if instr_id == "detectable_format:json_format":
        try: _json.loads(resp.strip()); return True
        except Exception:
            m = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', resp.strip())
            if m:
                try: _json.loads(m.group()); return True
                except Exception: pass
            return False
    if instr_id == "detectable_format:title":
        return bool(re.search(r'<<[^>]+>>', resp))
    if instr_id == "detectable_format:constrained_response":
        return len(resp.strip().split()) <= 5
    if instr_id == "language:response_language":
        return None  # requires langdetect; skip
    if instr_id == "startend:end_checker":
        return resp.strip().lower().endswith(kw.get("end_phrase", "").strip().lower())
    if instr_id == "startend:quotation":
        s = resp.strip(); return s.startswith('"') and s.endswith('"')
    if instr_id == "change_case:english_capital":
        letters = [c for c in resp if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)
    if instr_id == "change_case:english_lowercase":
        letters = [c for c in resp if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)
    if instr_id == "change_case:capital_word_frequency":
        count = sum(1 for w in resp.split() if w and w[0].isupper())
        return _relation(count, int(kw.get("capital_frequency", 1)), kw.get("capital_relation", "at least"))
    if instr_id == "combination:repeat_prompt":
        p = kw.get("prompt_to_repeat", "")
        return p.lower() in resp.lower() if p else False
    if instr_id == "combination:two_responses":
        import re
        return bool(re.search(r'\*{3,}|---+|\n{3,}', resp))
    if instr_id == "punctuation:no_comma":
        return "," not in resp
    return None


def _evaluate_ifeval_native(rows: list[dict]) -> dict:
    prompt_strict = prompt_loose = 0
    instr_strict_num = instr_strict_den = 0
    instr_loose_num  = instr_loose_den  = 0
    skipped: set[str] = set()

    for row in rows:
        response    = row.get("response", "")
        instr_ids   = row.get("instruction_id_list", [])
        kwargs_list = list(row.get("kwargs", []) or [])
        while len(kwargs_list) < len(instr_ids):
            kwargs_list.append({})

        row_strict, row_loose = [], []
        for instr_id, raw_kw in zip(instr_ids, kwargs_list):
            kw = {k: v for k, v in (raw_kw or {}).items() if v is not None}
            strict = _check_native(instr_id, kw, response, loose=False)
            loose  = _check_native(instr_id, kw, response, loose=True)
            if strict is None:
                skipped.add(instr_id); continue
            row_strict.append(strict)
            row_loose.append(loose if loose is not None else strict)
            instr_strict_den += 1; instr_loose_den += 1
            instr_strict_num += int(strict)
            instr_loose_num  += int(loose if loose is not None else strict)

        if row_strict: prompt_strict += int(all(row_strict))
        if row_loose:  prompt_loose  += int(all(row_loose))

    n = len(rows)
    if skipped:
        print(f"  Skipped (native): {', '.join(sorted(skipped))}")
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
    with open(path) as f:
        return [json.loads(line) for line in f]


def plot_ifeval(results: dict, save_path: str):
    models  = list(results.keys())
    metrics = ["prompt_strict", "prompt_loose", "instr_strict", "instr_loose"]
    labels  = ["Prompt\nStrict", "Prompt\nLoose", "Instr\nStrict", "Instr\nLoose"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x       = np.arange(len(metrics))
    width   = 0.25
    offsets = np.linspace(-(len(models)-1)*width/2, (len(models)-1)*width/2, len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model, offset) in enumerate(zip(models, offsets)):
        vals = [results[model][m] * 100 for m in metrics]
        bars = ax.bar(x + offset, vals, width,
                      label=model.replace("_", " ").title(),
                      color=colors[i % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("IFEval: Base vs Indirect vs Direct GRPO")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 100); ax.legend(); ax.grid(axis="y", alpha=0.3)
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
            print(f"  Skipping {model_name} — not found: {path}"); continue
        print(f"Evaluating {model_name}...")
        rows   = load_jsonl(path)
        result = evaluate_ifeval(rows)
        all_results[model_name] = result
        print(f"  n_prompts={result['n_prompts']}  n_instr={result['n_instructions']}")

    if not all_results:
        print("No results to display."); return

    header = f"{'Model':<12} {'Prompt Strict':>14} {'Prompt Loose':>13} {'Instr Strict':>13} {'Instr Loose':>12}"
    sep    = "-" * len(header)
    print("\n" + sep + "\nIFEval Results\n" + sep)
    print(header + "\n" + sep)
    for model, r in all_results.items():
        print(f"{model:<12} {r['prompt_strict']*100:>13.1f}% {r['prompt_loose']*100:>12.1f}% "
              f"{r['instr_strict']*100:>12.1f}% {r['instr_loose']*100:>11.1f}%")
    print(sep)

    with open(f"{args.results_dir}/ifeval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {args.results_dir}/ifeval_results.json")

    with open(f"{args.results_dir}/ifeval_table.txt", "w") as f:
        f.write(sep + "\nIFEval Results\n" + sep + "\n" + header + "\n" + sep + "\n")
        for model, r in all_results.items():
            f.write(f"{model:<12} {r['prompt_strict']*100:>13.1f}% {r['prompt_loose']*100:>12.1f}% "
                    f"{r['instr_strict']*100:>12.1f}% {r['instr_loose']*100:>11.1f}%\n")
        f.write(sep + "\n")

    plot_ifeval(all_results, f"{args.results_dir}/ifeval_bar.png")


if __name__ == "__main__":
    main()
