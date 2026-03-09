# Direct GRPO Training Results

## Setup
- **Policy model**: Qwen/Qwen2.5-7B-Instruct (LoRA r=16, alpha=64)
- **Judge model**: meta-llama/Llama-3.3-70B-Instruct-Turbo (live pairwise judging)
- **Training algorithm**: GRPO with G=4 completions per prompt
- **Reward signal**: Win fraction from all C(4,2)=6 pairwise comparisons, both orderings (12 judge calls per prompt)
- **KL penalty (β)**: 0.05
- **Learning rate**: 5e-6
- **Batch size**: 2, gradient accumulation: 4 (effective batch = 8)
- **Max completion length**: 256 tokens
- **Temperature**: 0.8
- **Hardware**: H100 GPU (Colab)
- **Runtime**: 2842.55s (~47 minutes) for 200 steps

## Training Loss

| Step | Training Loss |
|------|--------------|
| 10   | -0.012221    |
| 20   |  0.013584    |
| 30   | -0.048038    |
| 40   | -0.017312    |
| 50   |  0.005425    |
| 60   | -0.025825    |
| 70   | -0.014248    |
| 80   |  0.010663    |
| 90   |  0.060565    |
| 100  | -0.010842    |
| 110  | -0.012335    |
| 120  | -0.028858    |
| 130  | -0.005532    |
| 140  | -0.036266    |
| 150  | -0.019089    |
| 160  |  0.016801    |
| 170  | -0.041705    |
| 180  | -0.034296    |
| 190  | -0.009571    |
| 200  |  0.003697    |

**Average training loss**: -0.0103

## Key Metrics (from trainer state logs)

| Metric | Step 10 | Step 50 | Step 200 (final) |
|--------|---------|---------|-------------------|
| Reward (mean) | 0.500 | 0.500 | — |
| Reward (std) | 0.210 | 0.161 | — |
| KL divergence | 0.000464 | 0.000598 | — |
| Grad norm | 0.333 | 0.140 | — |
| Entropy | 0.378 | 0.368 | — |
| Completions clipped ratio | 0.775 | 0.788 | — |
| Mean completion length | 227.7 | 227.3 | — |
| Frac reward zero std | 0.35 | 0.55 | — |

## Summary

```
TrainOutput(
    global_step=200,
    training_loss=-0.010270056240260601,
    metrics={
        'train_runtime': 2842.5543,
        'train_samples_per_second': 0.563,
        'train_steps_per_second': 0.07,
        'total_flos': 0.0,
        'train_loss': -0.010270056240260601
    }
)
```

## Checkpoints
- `checkpoint-50/`
- `checkpoint-100/`
- `checkpoint-150/`
- `checkpoint-200/`

## Artifacts
- LoRA adapter: `adapter_model.safetensors` (39MB)
- Base model: `Qwen/Qwen2.5-7B-Instruct`

## Notes
- Loss oscillates around zero — expected for GRPO (advantage-weighted policy gradient with KL penalty)
- Mean reward stays at 0.5 throughout — this is the win fraction, which is always centered by construction (within each group, wins are relative)
- KL divergence remains very low (~0.0005), indicating the policy hasn't drifted far from the reference
- Eval was disabled during training to avoid the bottleneck of live-judging 850 eval prompts at each checkpoint
- Judge calls parallelized across all prompts in each batch (24 concurrent API calls per step)
