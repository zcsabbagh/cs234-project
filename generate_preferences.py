"""
Indirect RLAIF preference data generation pipeline.

For each instruction from UltraFeedback:
1. Generate 2 completions from the policy model (Qwen 2.5 7B)
2. Judge which is better using the judge model (Llama 3.3 70B)
3. Append {instruction, s1, s2, judgment, winner, loser} to a JSONL file

Supports parallel workers with a mutex-protected counter and file lock.
Retries API calls with exponential backoff.
"""

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from datasets import load_dataset

TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]

client = openai.OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
)

POLICY_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
OUTPUT_FILE = "preferences.jsonl"
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds

JUDGE_TEMPLATE = """You are an impartial judge. Given an instruction and two responses, decide which response is better.

Consider: accuracy, helpfulness, clarity, and conciseness.

Respond with EXACTLY "A" or "B" (no other text).

Instruction: {instruction}

Response A:
{response_a}

Response B:
{response_b}"""


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


def generate_completion(instruction: str) -> str:
    def call():
        return client.chat.completions.create(
            model=POLICY_MODEL,
            messages=[{"role": "user", "content": instruction}],
            max_tokens=512,
            temperature=0.7,
        )
    resp = api_call_with_retry(call)
    return resp.choices[0].message.content


def judge_pair(instruction: str, response_a: str, response_b: str) -> str:
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
    return resp.choices[0].message.content


def parse_verdict(judgment: str) -> str | None:
    """Extract 'A' or 'B' from the judge response."""
    text = judgment.strip().upper()
    if text in ("A", "B"):
        return text
    # Fallback: check for "Winner: A/B" in case the model adds extra text
    for line in judgment.splitlines():
        line = line.strip().upper()
        if line in ("A", "B"):
            return line
        if line.startswith("WINNER:"):
            pick = line.split(":", 1)[1].strip()
            if pick in ("A", "B"):
                return pick
    return None


class PreferenceWriter:
    """Thread-safe JSONL writer with a shared progress counter."""

    def __init__(self, output_path: str, start_count: int):
        self._path = output_path
        self._lock = threading.Lock()
        self._count = start_count

    def write(self, record: dict):
        with self._lock:
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


def process_one(instruction: str, writer: PreferenceWriter) -> dict:
    # Generate two completions
    s1 = generate_completion(instruction)
    s2 = generate_completion(instruction)

    # Randomize order to control for position bias
    swap = random.random() < 0.5
    if swap:
        shown_a, shown_b = s2, s1
        label_a, label_b = "s2", "s1"
    else:
        shown_a, shown_b = s1, s2
        label_a, label_b = "s1", "s2"

    # Judge
    judgment = judge_pair(instruction, shown_a, shown_b)
    pick = parse_verdict(judgment)

    # Map judge's A/B pick back to s1/s2 labels
    if pick == "A":
        winner, loser = label_a, label_b
    elif pick == "B":
        winner, loser = label_b, label_a
    else:
        winner, loser = None, None

    record = {
        "instruction": instruction,
        "s1": s1,
        "s2": s2,
        "judgment": judgment,
        "winner": winner,
        "loser": loser,
    }

    writer.write(record)
    return record


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1, help="Number of prompts to process")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("-o", "--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    # Count existing lines to resume from where we left off
    done = 0
    if os.path.exists(args.output):
        with open(args.output) as f:
            done = sum(1 for _ in f)
        print(f"Resuming: {done} already completed, skipping to instruction {done + 1}")

    writer = PreferenceWriter(args.output, done)

    # Load instructions into a list (skip already-done, take next n)
    print(f"Loading {args.n} instructions from UltraFeedback (skipping {done})...")
    ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
    instructions = []
    for i, row in enumerate(ds):
        if i < done:
            continue
        if len(instructions) >= args.n:
            break
        instructions.append((i, row["instruction"]))
    print(f"Loaded {len(instructions)} instructions. Running with {args.workers} workers.\n")

    target = done + len(instructions)

    def worker(idx: int, instruction: str):
        record = process_one(instruction, writer)
        count = writer.count
        print(f"  [{count}/{target}] idx={idx} winner={record['winner']} ({len(instruction)} char prompt)")
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(worker, idx, inst): idx
            for idx, inst in instructions
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                idx = futures[future]
                print(f"  [ERROR] idx={idx}: {e}")

    print(f"\nDone. {writer.count - done} new record(s) written to {args.output} ({writer.count} total)")


if __name__ == "__main__":
    main()
