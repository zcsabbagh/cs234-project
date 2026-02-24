"""
Test Together API with OpenAI-compatible client.
1. Generate a response from Qwen 2.5 7B Instruct to an UltraFeedback-style prompt
2. Judge that response using Llama 3.1 70B Instruct as a pairwise judge
"""

import os
import openai

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise RuntimeError("Set TOGETHER_API_KEY environment variable first")

client = openai.OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
)

POLICY_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# --- Step 1: Generate two completions from the policy model ---

PROMPT = (
    "Explain the key differences between nuclear fission and nuclear fusion, "
    "including their energy output, fuel sources, and current technological feasibility. "
    "Keep your answer concise but informative."
)

print("=" * 60)
print(f"PROMPT: {PROMPT}")
print("=" * 60)

print(f"\n--- Generating 2 completions from {POLICY_MODEL} ---\n")

completions = []
for i in range(2):
    resp = client.chat.completions.create(
        model=POLICY_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=512,
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    completions.append(text)
    print(f"[Completion {i+1}] ({len(text)} chars)")
    print(text)
    print()

# --- Step 2: Judge the pair with Llama 70B ---

JUDGE_PROMPT = f"""You are an impartial judge. Given an instruction and two responses, decide which response is better.

Consider: accuracy, helpfulness, clarity, and conciseness.

Respond with EXACTLY this format:
Winner: A or B
Justification: <one or two sentences>

Instruction: {PROMPT}

Response A:
{completions[0]}

Response B:
{completions[1]}"""

print(f"--- Judging with {JUDGE_MODEL} ---\n")

judge_resp = client.chat.completions.create(
    model=JUDGE_MODEL,
    messages=[{"role": "user", "content": JUDGE_PROMPT}],
    max_tokens=150,
    temperature=0.0,
)

verdict = judge_resp.choices[0].message.content
print(f"[Judge verdict]\n{verdict}\n")

# --- Summary ---
print("=" * 60)
print("Pipeline test complete.")
print(f"  Policy model: {POLICY_MODEL}")
print(f"  Judge model:  {JUDGE_MODEL}")
print(f"  Completions generated: {len(completions)}")
print(f"  Judge called: 1 pairwise comparison")
print("=" * 60)
