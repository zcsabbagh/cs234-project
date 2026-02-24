# Step-by-Step: Direct vs. Indirect Pairwise RLAIF

## 1. Gather your ingredients

You need three things before any training happens:

**Base policy**: Qwen 2.5 7B Instruct — this is the model you're improving. Load it with whatever framework you're using (vLLM for fast inference, HF Transformers for training).

**Judge**: Llama 3.1 70B Instruct — serves as your AI feedback source. You'll query this either offline (indirect) or online (direct). Host it on a separate GPU cluster or use an inference API, since you don't want judge inference to bottleneck training.

**Prompt set**: ~20K instructions from UltraFeedback. Filter for diversity — you want a mix of reasoning, creative writing, coding, factual QA, etc. Hold out maybe 1-2K for validation.

---

## 2. Shared infrastructure: GRPO setup

Both approaches use Group Relative Policy Optimization, so set this up first. GRPO works like this per training step:

- For each prompt $x$ in your batch, sample $G=4$ completions $\{y_1, y_2, y_3, y_4\}$ from your current policy $\pi_\theta$
- Obtain a reward $r_i$ for each completion (this is where the two approaches diverge)
- Normalize rewards within the group: $\hat{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$
- Update the policy to upweight high-reward completions and downweight low-reward ones, with a KL penalty against the reference policy (your frozen copy of Qwen 7B)

The key hyperparameters are the KL coefficient $\beta$, learning rate, batch size, and number of generations $G$. Keep these identical across both approaches so you're comparing apples to apples.

---

## 3. Indirect approach (reward model pipeline)

### 3a. Generate preference data

For each of your 20K prompts, sample 2 completions from the base Qwen 7B. You now have 20K pairs $(x, y_w, y_l)$ waiting to be labeled.

### 3b. Judge the pairs

Construct a pairwise prompt for Llama 70B like:

```
Given the following instruction and two responses, which response
is better? Respond with "A" or "B" and a brief justification.

Instruction: {prompt}
Response A: {completion_1}
Response B: {completion_2}
```

Randomize A/B order to control for position bias. Run all 20K through the judge. You get a dataset of $(x, y_w, y_l)$ triples.

### 3c. Train the Bradley-Terry reward model

Take a smaller model (you could fine-tune a copy of Qwen 7B, or use a separate 1-3B model) and train it as a reward model. The BT loss for each pair is:

$\mathcal{L} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$

Train until validation accuracy plateaus (usually 65-75% accuracy on held-out pairs is reasonable). This gives you a cheap scalar reward function $r_\theta(x, y)$.

### 3d. Run GRPO with the reward model

Now the RL loop is straightforward:
1. Sample a batch of prompts
2. Generate $G=4$ completions per prompt from current policy
3. Score each with $r_\theta(x, y_i)$ — this is a fast forward pass
4. Compute GRPO update
5. Repeat

This is relatively cheap per step because the reward model is small and runs locally.

---

## 4. Direct approach (live judge pipeline)

### 4a. Run GRPO with live pairwise judging

For each training step:
1. Sample a batch of prompts
2. Generate $G=4$ completions per prompt from current policy
3. For each prompt, form all $\binom{4}{2} \times 2 = 12$ ordered pairs and query Llama 70B on each: "Is $y_i$ better than $y_j$?"
4. Compute each completion's reward as its **win fraction**: $r_i = \frac{\text{wins}_i}{G-1}$ (or over all 6 comparisons involving $y_i$ — each completion appears in 6 of the 12 ordered pairs, winning some fraction)
5. Compute GRPO update using these rewards

Actually, let me be more precise on the combinatorics. With $G=4$ completions, there are $\binom{4}{2}=6$ unordered pairs. If you're querying all 12 *ordered* pairs (i.e., presenting $(y_i, y_j)$ and $(y_j, y_i)$ separately to control for position bias), then each completion participates in 6 comparisons. Its win fraction is $\frac{\text{wins}}{6}$, giving you rewards in $\{0, \frac{1}{6}, \frac{2}{6}, \ldots, 1\}$.

### 4b. Why this is expensive

Every training step requires $12 \times B$ judge calls (where $B$ is your batch size in prompts). If your batch is 64 prompts, that's 768 Llama 70B inferences *per step*. This is the core tradeoff — you get fresh, on-policy feedback but pay heavily in compute. You'll want to:

- Batch judge queries aggressively
- Use a fast serving backend (vLLM with tensor parallelism)
- Possibly reduce to 6 pairs (one ordering) and accept some position bias, or average over both orderings but cache

---

## 5. Evaluation

### 5a. IFEval

Run your checkpoints (save every N steps) on the IFEval benchmark. This is programmatic — it checks whether the model followed specific verifiable instructions (e.g., "write exactly 3 paragraphs," "include the word 'banana' at least twice"). No judge ambiguity here.

### 5b. AlpacaEval with held-out judge

This is where you detect reward hacking. Use Claude or GPT-4 as the evaluator — **not** Llama 70B. Compare your trained model's outputs against a reference (usually GPT-4 outputs or the base model outputs). If AlpacaEval win rate improves, the model genuinely got better, not just better at pleasing Llama 70B.

### 5c. Reward hacking diagnostic plots

For each checkpoint during training, plot two curves:

- **Training reward** (Llama 70B judge scores or reward model scores) — this should go up by construction
- **Held-out eval** (Claude/GPT-4 AlpacaEval win rate, IFEval accuracy)

The hacking signature is training reward climbing while held-out eval plateaus or drops. If the direct approach shows less divergence between these curves, that supports your hypothesis.

### 5d. Manual inspection

Sample 50-100 outputs from each final model. Look for:
- **Sycophancy**: excessively agreeing with the prompt, hedging everything
- **Verbosity gaming**: padding responses with unnecessary detail to seem "more thorough"
- **Format hacking**: exploiting patterns the judge likes (e.g., always using bullet points, always starting with "Great question!")
- **Collapse**: low diversity, repetitive structure across different prompts

---

## 6. Practical sequencing

Here's the actual order of operations:

1. **Week 1**: Set up infrastructure — host Llama 70B for inference, get GRPO training loop working on Qwen 7B with a dummy reward, verify UltraFeedback prompts are clean
2. **Week 2**: Run the indirect pipeline — generate pairs, judge them, train reward model, verify RM accuracy
3. **Week 3**: Run indirect GRPO training, saving checkpoints every ~50 steps
4. **Week 3-4**: Run direct GRPO training in parallel (this takes longer due to judge latency)
5. **Week 4-5**: Evaluate all checkpoints on IFEval + AlpacaEval, generate diagnostic plots, do manual inspection

The main engineering challenge is step 4 — keeping the judge inference pipeline fast enough that RL training doesn't stall. If you're using the Tinker API they may handle some of this for you, but the 12× judge call overhead per prompt is the bottleneck either way.

---

## 7. Expected findings and what to watch for

Based on your hypothesis: the direct approach should be more robust (less hacking) because the judge sees the model's *current* outputs and compares them against each other, rather than relying on a fixed reward model trained on the base model's distribution. The reward model can be exploited because the policy drifts out-of-distribution from the RM's training data.

The key number to report is the **cost-quality Pareto frontier**: at equivalent compute budgets, which approach gives better held-out eval? And at equivalent held-out eval, how much cheaper is indirect? That's the tradeoff you're quantifying.
