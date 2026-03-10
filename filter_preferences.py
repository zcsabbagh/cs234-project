"""
Filter preferences.jsonl to high-confidence pairs by double-judging.

For each existing pair, re-judge with the REVERSE A/B ordering.
Keep only pairs where both orderings agree on the same winner.
Pairs that flip (position-biased) or were random are discarded.

Expected: ~60-70% of pairs pass (discard ~30% noisy ones).
Result: smaller but much cleaner dataset for RM training.

Usage:
    python filter_preferences.py                          # filter all pairs
    python filter_preferences.py -w 16                   # more parallel workers
    python filter_preferences.py -i prefs.jsonl -o filtered.jsonl
"""

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
client = openai.OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
)

JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
MAX_RETRIES = 5
BASE_DELAY = 1.0

# Structured CoT prompt — judge must reason before picking, reduces random choices
JUDGE_TEMPLATE = """You are an impartial judge evaluating two AI assistant responses.

Instruction: {instruction}

Response A:
{response_a}

Response B:
{response_b}

Evaluate both responses on: accuracy, helpfulness, clarity, and conciseness.
Think step by step, then end your response with EXACTLY "Winner: A" or "Winner: B".

Your evaluation:"""


def api_call_with_retry(fn):
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
        except openai.APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                raise


def parse_verdict(text: str) -> str | None:
    """Extract A or B — accepts both simple 'A'/'B' and 'Winner: A/B' format."""
    text = text.strip()
    # Check last non-empty line first (where CoT response ends with the verdict)
    for line in reversed(text.splitlines()):
        line = line.strip().upper()
        if line in ("A", "B"):
            return line
        if "WINNER:" in line:
            pick = line.split("WINNER:", 1)[1].strip()
            if pick.startswith("A"):
                return "A"
            if pick.startswith("B"):
                return "B"
    # Fallback: any line that is just A or B
    for line in text.splitlines():
        line = line.strip().upper()
        if line in ("A", "B"):
            return line
    return None


def judge_pair(instruction: str, response_a: str, response_b: str) -> str | None:
    prompt = JUDGE_TEMPLATE.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
    )
    def call():
        return client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,   # enough room for CoT reasoning
            temperature=0.0,
        )
    resp = api_call_with_retry(call)
    return parse_verdict(resp.choices[0].message.content)


def check_pair(row: dict) -> dict | None:
    """Re-judge the pair with REVERSED ordering. Return row if both orderings agree."""
    instruction = row["instruction"]
    winner_key  = row["winner"]   # "s1" or "s2"
    loser_key   = row["loser"]

    winner_text = row[winner_key]
    loser_text  = row[loser_key]

    # Original judgment said winner > loser.
    # Now present as A=loser, B=winner (reversed).
    # If judge picks "B" → winner is still better → consistent → KEEP.
    # If judge picks "A" → loser now looks better → inconsistent → DISCARD.
    second_verdict = judge_pair(instruction, loser_text, winner_text)

    if second_verdict == "B":
        # Consistent: winner is better in both orderings
        return row
    else:
        # Inconsistent or parse failure: likely position bias or random → discard
        return None


def main():
    parser = argparse.ArgumentParser(description="Filter preferences by double-judging")
    parser.add_argument("-i", "--input",   default="preferences.jsonl")
    parser.add_argument("-o", "--output",  default="preferences_filtered.jsonl")
    parser.add_argument("-w", "--workers", type=int, default=12,
                        help="Parallel workers for judge calls")
    parser.add_argument("--max",           type=int, default=None,
                        help="Max pairs to process (default: all)")
    args = parser.parse_args()

    # Load valid pairs (skip None winners)
    rows = []
    skipped_none = 0
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("winner") or not row.get("loser"):
                skipped_none += 1
                continue
            if "s1" not in row or "s2" not in row:
                continue
            rows.append(row)

    if args.max:
        rows = rows[:args.max]

    print(f"Input:   {args.input}")
    print(f"Pairs:   {len(rows)} valid  ({skipped_none} skipped — None winner)")
    print(f"Output:  {args.output}")
    print(f"Workers: {args.workers}")
    print(f"Judge:   {JUDGE_MODEL} (CoT prompt, max_tokens=512)")
    print(f"\nExpect ~60-70% pass rate. Starting...\n")

    kept = 0
    discarded = 0
    errors = 0
    lock = threading.Lock()

    # Skip already-written output lines (resume support)
    done_instructions = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_instructions.add(json.loads(line)["instruction"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_instructions)} already in output, skipping.")
        kept = len(done_instructions)

    rows_to_process = [r for r in rows if r["instruction"] not in done_instructions]
    total = kept + len(rows_to_process)

    def process(row):
        nonlocal kept, discarded, errors
        try:
            result = check_pair(row)
        except Exception as e:
            with lock:
                errors += 1
                print(f"  [ERROR] {e}")
            return

        with lock:
            if result is not None:
                with open(args.output, "a") as f:
                    f.write(json.dumps(result) + "\n")
                kept += 1
                status = "KEEP"
            else:
                discarded += 1
                status = "DISCARD"

            done = kept + discarded
            if done % 100 == 0 or done <= 20:
                rate = kept / done if done else 0
                print(f"  [{done}/{len(rows_to_process)}] kept={kept} discarded={discarded} "
                      f"pass_rate={rate:.1%}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, row) for row in rows_to_process]
        for future in as_completed(futures):
            future.result()

    total_processed = kept + discarded
    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"  Processed:  {total_processed}")
    print(f"  Kept:       {kept}  ({kept/total_processed:.1%})")
    print(f"  Discarded:  {discarded}  ({discarded/total_processed:.1%})")
    print(f"  Errors:     {errors}")
    print(f"  Output:     {args.output}")
    print(f"\nTrain RM on filtered data:")
    print(f"  python train_reward_model.py --data {args.output} --max-length 1024")


if __name__ == "__main__":
    main()
