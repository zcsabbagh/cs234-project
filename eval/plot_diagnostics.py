"""
Plot training reward curves and reward-hacking diagnostic.

Reads trainer_state.json from each GRPO checkpoint and optionally
loads AlpacaEval results to overlay the held-out eval on the same plot.

Outputs (all saved to --results-dir):
  training_rewards.png        Training reward vs step for both methods
  reward_hacking.png          Training reward vs Claude win rate (hacking diagnostic)
  training_kl.png             KL divergence over steps
  training_stats.txt          Printed summary also written to file

Usage:
    python eval/plot_diagnostics.py \
        --indirect-ckpt /content/drive/MyDrive/cs234/grpo_indirect \
        --direct-ckpt   /content/drive/MyDrive/cs234/grpo_direct \
        --results-dir   ./eval_results
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Parse trainer_state.json ──────────────────────────────────────────────

def load_trainer_state(ckpt_dir: str) -> dict:
    """Load trainer_state.json from a checkpoint directory."""
    # Check the checkpoint dir itself and any checkpoint-* subdirs
    candidates = [
        os.path.join(ckpt_dir, "trainer_state.json"),
    ]
    for name in os.listdir(ckpt_dir):
        sub = os.path.join(ckpt_dir, name)
        if os.path.isdir(sub):
            candidates.append(os.path.join(sub, "trainer_state.json"))

    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    raise FileNotFoundError(
        f"trainer_state.json not found in {ckpt_dir} or its subdirectories.\n"
        f"Searched: {candidates}"
    )


def extract_train_metrics(log_history: list[dict]) -> tuple[list, list, list, list]:
    """Extract (steps, reward_means, reward_stds, kls) from training log entries."""
    steps, rewards, reward_stds, kls = [], [], [], []
    for entry in log_history:
        if "rewards/reward_fn/mean" in entry:
            steps.append(entry["step"])
            rewards.append(entry["rewards/reward_fn/mean"])
            reward_stds.append(entry.get("reward_std", 0.0))
            kls.append(entry.get("kl", 0.0))
    return steps, rewards, reward_stds, kls


def extract_eval_metrics(log_history: list[dict]) -> tuple[list, list]:
    """Extract (steps, eval_reward_means) from eval log entries."""
    steps, rewards = [], []
    for entry in log_history:
        if "eval_rewards/reward_fn/mean" in entry:
            steps.append(entry["step"])
            rewards.append(entry["eval_rewards/reward_fn/mean"])
    return steps, rewards


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_training_rewards(data: dict, save_path: str):
    """Training reward over steps for both methods on the same axes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"indirect": "#4C72B0", "direct": "#DD8452"}

    for method, d in data.items():
        steps   = d["train_steps"]
        rewards = d["train_rewards"]
        stds    = d["train_reward_stds"]
        color   = colors.get(method, "green")

        # Smooth with rolling window
        window = 5
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        s_steps = steps[window - 1:]

        ax.plot(steps, rewards, alpha=0.25, color=color, linewidth=1)
        ax.plot(s_steps, smooth, color=color, linewidth=2,
                label=f"{method.title()} GRPO (smoothed)")

        # Mark eval points
        if d["eval_steps"]:
            ax.scatter(d["eval_steps"], d["eval_rewards"],
                       color=color, marker="D", s=80, zorder=5,
                       label=f"{method.title()} eval (training judge)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward (training judge: Llama 70B / RM)")
    ax.set_title("Training Reward Over Steps")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_kl(data: dict, save_path: str):
    """KL divergence over steps."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"indirect": "#4C72B0", "direct": "#DD8452"}

    for method, d in data.items():
        steps = d["train_steps"]
        kls   = d["train_kls"]
        color = colors.get(method, "green")
        ax.plot(steps, kls, color=color, linewidth=1.5, label=f"{method.title()} GRPO")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("KL Divergence from Reference")
    ax.set_title("KL Divergence Over Training (policy drift)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


def plot_reward_hacking(data: dict, alpacaeval_results: dict | None, save_path: str):
    """
    Reward hacking diagnostic:
      X-axis: final training reward (Llama 70B / RM score)
      Y-axis: held-out eval win rate (Claude)
      One point per method.

    If training rewards at step 100 and 200 are available alongside Claude
    win rates at those checkpoints, plots a trajectory. Otherwise plots a
    single summary point per method.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = {"indirect": "#4C72B0", "direct": "#DD8452"}

    if alpacaeval_results:
        for method, d in data.items():
            if method not in alpacaeval_results:
                continue
            # Final training reward
            final_train_reward = np.mean(d["train_rewards"][-20:]) if d["train_rewards"] else None
            claude_win_rate    = alpacaeval_results[method]["win_rate"] * 100
            color = colors.get(method, "green")

            if final_train_reward is not None:
                ax.scatter(
                    final_train_reward, claude_win_rate,
                    color=color, s=200, zorder=5, label=method.title()
                )
                ax.annotate(
                    f" {method.title()}\n(step 200)",
                    (final_train_reward, claude_win_rate),
                    fontsize=10,
                )

        ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                   label="50% win rate (= base model)")
        ax.set_xlabel("Final Training Reward (Llama 70B / RM)", fontsize=12)
        ax.set_ylabel("AlpacaEval Win Rate vs Base Model (GPT-4o judge) %", fontsize=12)
        ax.set_title("Reward Hacking Diagnostic\n"
                     "Hacking signature: high training reward, low held-out eval",
                     fontsize=12)
        note = ("Dashed line = 50% (parity with base model). "
                "Upper-right = genuine improvement. "
                "Right + low = hacking.")
        ax.text(0.02, 0.02, note, transform=ax.transAxes,
                fontsize=8, color="gray", va="bottom")
    else:
        # No Claude results yet — plot training reward over time as a proxy
        for method, d in data.items():
            color = colors.get(method, "green")
            ax.plot(d["train_steps"], d["train_rewards"],
                    color=color, alpha=0.5, linewidth=1)
            window = 5
            smooth = np.convolve(d["train_rewards"],
                                 np.ones(window) / window, mode="valid")
            s_steps = d["train_steps"][window - 1:]
            ax.plot(s_steps, smooth, color=color, linewidth=2, label=f"{method.title()}")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Training Reward (proxy — run run_alpacaeval.py for full diagnostic)")
        ax.set_title("Reward Hacking Diagnostic (partial — Claude results not yet available)")

    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved → {save_path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indirect-ckpt", default=None)
    parser.add_argument("--direct-ckpt",   default=None)
    parser.add_argument("--results-dir",   default="./eval_results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    ckpt_map = {}
    if args.indirect_ckpt:
        ckpt_map["indirect"] = args.indirect_ckpt
    if args.direct_ckpt:
        ckpt_map["direct"] = args.direct_ckpt

    if not ckpt_map:
        raise SystemExit("ERROR: provide at least --indirect-ckpt or --direct-ckpt")

    data = {}
    for method, ckpt_dir in ckpt_map.items():
        print(f"Loading trainer_state for {method} from {ckpt_dir}...")
        state = load_trainer_state(ckpt_dir)
        log   = state.get("log_history", [])

        t_steps, t_rewards, t_stds, t_kls = extract_train_metrics(log)
        e_steps, e_rewards                 = extract_eval_metrics(log)

        data[method] = {
            "train_steps":        t_steps,
            "train_rewards":      t_rewards,
            "train_reward_stds":  t_stds,
            "train_kls":          t_kls,
            "eval_steps":         e_steps,
            "eval_rewards":       e_rewards,
        }
        print(f"  Train entries: {len(t_steps)}  Eval entries: {len(e_steps)}")

    # Load AlpacaEval results if available
    alpacaeval_path = f"{args.results_dir}/alpacaeval_results.json"
    alpacaeval_results = None
    if os.path.exists(alpacaeval_path):
        with open(alpacaeval_path) as f:
            alpacaeval_results = json.load(f)
        print(f"Loaded AlpacaEval results from {alpacaeval_path}")
    else:
        print(f"AlpacaEval results not found at {alpacaeval_path} — hacking diagnostic will be partial")

    # ── Print summary ─────────────────────────────────────────────────────
    lines = []
    lines.append("\n" + "=" * 55)
    lines.append("Training Diagnostics Summary")
    lines.append("=" * 55)
    for method, d in data.items():
        if not d["train_rewards"]:
            continue
        rw = d["train_rewards"]
        kl = d["train_kls"]
        lines.append(f"\n  {method.upper()} GRPO:")
        lines.append(f"    Steps trained:       {len(d['train_steps'])}")
        lines.append(f"    Reward (initial):    {rw[0]:.3f}")
        lines.append(f"    Reward (final):      {rw[-1]:.3f}")
        lines.append(f"    Reward (max):        {max(rw):.3f}")
        lines.append(f"    Reward Δ:            {rw[-1] - rw[0]:+.3f}")
        lines.append(f"    KL (final):          {kl[-1]:.6f}" if kl else "")
        if d["eval_steps"]:
            for es, er in zip(d["eval_steps"], d["eval_rewards"]):
                lines.append(f"    Eval reward @step {es}: {er:.3f}")
        if alpacaeval_results and method in alpacaeval_results:
            wr = alpacaeval_results[method]
            lines.append(f"    Claude win rate:     {wr['win_rate']*100:.1f}% ±{wr['ci_95']*100:.1f}%")
    lines.append("=" * 55)

    summary = "\n".join(lines)
    print(summary)

    stats_path = f"{args.results_dir}/training_stats.txt"
    with open(stats_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary saved → {stats_path}")

    # ── Generate plots ────────────────────────────────────────────────────
    plot_training_rewards(data, f"{args.results_dir}/training_rewards.png")
    plot_kl(data,               f"{args.results_dir}/training_kl.png")
    plot_reward_hacking(data, alpacaeval_results, f"{args.results_dir}/reward_hacking.png")


if __name__ == "__main__":
    main()
