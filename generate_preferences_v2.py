"""
High-quality preference data generation — v2.

Key improvements over v1:
  1. Asymmetric sampling: temp=0.3 ("direct/concise") vs temp=1.1 ("creative")
     → structural diversity guaranteed before any filtering
  2. Diverse system prompts per temperature config
     → further pushes structural divergence between the two responses
  3. Pairwise A/B judging with rubric embedded in prompt (verbosity penalized)
     → judge picks winner directly — no score parsing, no brittle field matching
  4. ROUGE-L similarity filter (threshold 0.85, no external deps)
     → discards near-identical pairs regardless of judgment
  5. A/B order randomized per instruction to control position bias
  6. Randomly assigns winner to s1 or s2 to maintain label balance

API calls per instruction: 2 generation + 1 judge = 3 total.
Expected keep rate: ~70-85% (only ROUGE-L and judge failures can discard).

Usage:
    python generate_preferences_v2.py -n 5000 -w 8
    python generate_preferences_v2.py -n 5000 -w 8 -o preferences_v2.jsonl
"""

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from datasets import load_dataset

# Load .env file if present (so TOGETHER_API_KEY doesn't need to be exported).
# Always overrides shell env if the shell value is empty.
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k = _k.strip()
            _v = _v.strip().strip("'\"")   # remove surrounding quotes if any
            if _v:                          # only set if value is non-empty
                os.environ[_k] = _v

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
if not TOGETHER_API_KEY:
    raise SystemExit(
        "ERROR: TOGETHER_API_KEY is not set.\n"
        f"  Option 1: echo 'TOGETHER_API_KEY=your_key' > {_env_path}\n"
        "  Option 2: export TOGETHER_API_KEY=your_key"
    )
print(f"API key loaded: ...{TOGETHER_API_KEY[-6:]}")
client = openai.OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
)

POLICY_MODEL    = "Qwen/Qwen2.5-7B-Instruct-Turbo"
JUDGE_MODEL     = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
MAX_RETRIES     = 5
BASE_DELAY      = 1.0
ROUGE_THRESHOLD = 0.85

# Two configs: conservative vs creative — maximally different by design.
# Low temp gets a "concise" system prompt to further push it toward brevity.
# High temp gets a neutral prompt so diversity comes from temperature, not instruction.
CONCISE_CONFIG = {
    "temperature": 0.3,
    "top_p": 0.9,
    "system": "You are a helpful assistant. Answer directly and concisely.",
}
CREATIVE_CONFIG = {
    "temperature": 1.1,
    "top_p": 0.95,
    "system": "You are a helpful assistant.",
}

# Pairwise judge prompt with rubric embedded.
# Verbosity is explicitly penalized in the criteria.
JUDGE_TEMPLATE = """\
You are an impartial judge evaluating two AI assistant responses to the same instruction.

Instruction: {instruction}

Response A:
{response_a}

Response B:
{response_b}

Evaluate both responses on these criteria:
  1. Correctness: Is the information accurate and free of errors?
  2. Completeness: Does it fully address the instruction?
  3. Brevity: Is it concise and direct, without padding or filler?
     Penalize: unnecessary preamble ("Great question!", "Certainly!"), restating
     the question, excessive bullet points when prose would do, filler phrases,
     and inflating length to appear more thorough.
  4. Clarity: Is it well-organised and easy to understand?

Respond with EXACTLY one letter — A or B — for the better response. No other text."""


# ── API helpers ───────────────────────────────────────────────────────────

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


def generate_response(instruction: str, config: dict) -> str:
    def call():
        return client.chat.completions.create(
            model=POLICY_MODEL,
            messages=[
                {"role": "system", "content": config["system"]},
                {"role": "user",   "content": instruction},
            ],
            max_tokens=512,
            temperature=config["temperature"],
            top_p=config["top_p"],
        )
    return api_call_with_retry(call).choices[0].message.content


def judge_pair(instruction: str, response_a: str, response_b: str) -> str | None:
    """Ask Llama 70B to pick the better response. Returns 'A', 'B', or None."""
    prompt = JUDGE_TEMPLATE.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
    )
    def call():
        return client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
    text = api_call_with_retry(call).choices[0].message.content.strip().upper()
    # Accept bare "A"/"B" or "Winner: A/B"
    if text in ("A", "B"):
        return text
    m = re.search(r"\b([AB])\b", text)
    return m.group(1) if m else None


# ── ROUGE-L (token-level LCS, no external deps) ───────────────────────────

def rouge_l_f1(a: str, b: str, max_tokens: int = 200) -> float:
    ta = a.lower().split()[:max_tokens]
    tb = b.lower().split()[:max_tokens]
    if not ta or not tb:
        return 0.0
    m, n = len(ta), len(tb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if ta[i-1] == tb[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p, r = lcs / n, lcs / m
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── Core pipeline ─────────────────────────────────────────────────────────

def process_one(instruction: str, writer) -> tuple[dict | None, str]:
    """
    Generate 2 responses (concise vs creative), judge pairwise, filter.
    Returns (record, reason) where reason is "kept" | "gen_fail" | "rouge" | "no_verdict".
    """
    # Step 1: Generate both responses
    try:
        concise  = generate_response(instruction, CONCISE_CONFIG)
        creative = generate_response(instruction, CREATIVE_CONFIG)
    except Exception as e:
        return None, f"gen_fail:{type(e).__name__}"

    if not concise or not creative or len(concise.strip()) < 20 or len(creative.strip()) < 20:
        return None, "gen_fail"

    # Step 2: ROUGE-L filter — discard if responses are too similar
    if rouge_l_f1(concise, creative) > ROUGE_THRESHOLD:
        return None, "rouge"

    # Step 3: Randomize A/B order to control position bias
    if random.random() < 0.5:
        response_a, response_b = concise, creative
        a_is_concise = True
    else:
        response_a, response_b = creative, concise
        a_is_concise = False

    # Step 4: Judge
    verdict = judge_pair(instruction, response_a, response_b)
    if verdict is None:
        return None, "no_verdict"

    # Determine winner/loser texts
    if verdict == "A":
        winner_text, loser_text = response_a, response_b
    else:
        winner_text, loser_text = response_b, response_a

    # Step 5: Randomly assign to s1/s2 for label balance
    if random.random() < 0.5:
        s1, s2        = winner_text, loser_text
        winner, loser = "s1", "s2"
    else:
        s1, s2        = loser_text, winner_text
        winner, loser = "s2", "s1"

    record = {
        "instruction": instruction,
        "s1":          s1,
        "s2":          s2,
        "winner":      winner,
        "loser":       loser,
    }
    writer.write(record)
    return record, "kept"


# ── I/O helpers ───────────────────────────────────────────────────────────

class PreferenceWriter:
    def __init__(self, path: str, start_count: int):
        self._path  = path
        self._lock  = threading.Lock()
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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate high-quality preference pairs (v2)")
    parser.add_argument("-n",               type=int,   default=1000,           help="Instructions to process")
    parser.add_argument("-w", "--workers",  type=int,   default=8,              help="Parallel workers")
    parser.add_argument("-o", "--output",   default="preferences_v2.jsonl")
    parser.add_argument("--rouge-threshold",type=float, default=ROUGE_THRESHOLD,help="ROUGE-L threshold (default 0.85)")
    parser.add_argument("--skip",           type=int,   default=0,              help="Skip first N UltraFeedback instructions")
    args = parser.parse_args()

    # Resume: skip already-written instructions
    done_instructions: set[str] = set()
    done_count = 0
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_instructions.add(json.loads(line)["instruction"])
                    done_count += 1
                except Exception:
                    pass
        print(f"Resuming: {done_count} pairs already in {args.output}")

    writer = PreferenceWriter(args.output, done_count)

    print(f"Loading up to {args.n} new instructions from UltraFeedback (skip offset: {args.skip})...")
    ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
    instructions: list[str] = []
    global_idx = 0
    for row in ds:
        if global_idx < args.skip:
            global_idx += 1
            continue
        instr = row["instruction"]
        if instr not in done_instructions:
            instructions.append(instr)
        global_idx += 1
        if len(instructions) >= args.n:
            break

    print(f"Loaded {len(instructions)} new instructions.")
    print(f"Sampling: temp=0.3 (concise) vs temp=1.1 (creative)")
    print(f"Filter: ROUGE-L > {args.rouge_threshold} → discard")
    print(f"API calls per instruction: 2 generation + 1 judge = 3\n")

    kept = skipped = errors = 0
    reason_counts: dict[str, int] = {}
    lock = threading.Lock()
    target = done_count + len(instructions)

    def worker(instruction: str):
        nonlocal kept, skipped, errors
        try:
            record, reason = process_one(instruction, writer)
        except Exception as e:
            with lock:
                errors += 1
                print(f"  [ERROR] {e}")
            return

        with lock:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if record is not None:
                kept += 1
                print(f"  [KEPT  {writer.count:>5}/{target}] {instruction[:65]}...")
            else:
                skipped += 1
                done = kept + skipped
                if done % 25 == 0:
                    rate = kept / done if done else 0.0
                    reason_str = "  ".join(
                        f"{k}={v}" for k, v in sorted(reason_counts.items()) if k != "kept"
                    )
                    print(f"  [skip  {done:>5}] kept_rate={rate:.1%}  ({reason_str}  errors={errors})")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, instr) for instr in instructions]
        for f in as_completed(futures):
            f.result()

    total = kept + skipped
    print(f"\n{'='*55}")
    print(f"DONE")
    print(f"  Instructions processed: {total}")
    print(f"  Pairs kept:    {kept}  ({kept/total:.1%})")
    print(f"  Pairs skipped: {skipped}  ({skipped/total:.1%})")
    print(f"  Errors:        {errors}")
    if reason_counts:
        print(f"  Discard breakdown:")
        for reason, count in sorted(reason_counts.items()):
            if reason != "kept":
                print(f"    {reason}: {count}")
    print(f"  Output: {args.output}")
    print(f"\nNext step:")
    print(f"  python train_reward_model.py --data {args.output} --max-length 1024")


if __name__ == "__main__":
    main()
