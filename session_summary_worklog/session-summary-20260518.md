# Session Summary — 2026-05-18

## Overview

This session focused on three parallel workstreams: (1) launching OPD Cumulative v2 training with improved hyperparameters, (2) generating additional teacher SFT data, and (3) deep analysis of the Lightning-OPD pipeline to align our approach.

---

## 1. OPD Cumulative v2 Training (p5-2)

### Algorithm Improvement

Modified the cumulative KL penalty to normalize by actual lookahead steps:

```python
# Before: raw sum (penalty scale depends on K)
cumulative_kl[t] = (reverse_kl[t:end] * weights).sum()

# After: normalized average (penalty scale independent of K)
actual_k = end - t
cumulative_kl[t] = (reverse_kl[t:end] * weights).sum() / actual_k
```

This ensures `opd_kl_coef` has consistent meaning regardless of horizon K or position in sequence (including tail tokens with fewer than K steps remaining).

### Hyperparameters (aligned with slime official example)

| Parameter | v1 (previous) | v2 (current) | slime example |
|-----------|:---:|:---:|:---:|
| opd_kl_coef | 0.05 | **1.0** | 1.0 |
| opd_horizon (K) | 8 | **2** | N/A (instant) |
| lr | 5e-7 | **1e-6** | 1e-6 |
| temperature | 0.6 | **1.0** | 1.0 |
| max_response_len | 8192 | **16384** | 16384 |
| global_batch_size | 32 | **64** | 64 |
| rollout_batch_size | 8 | **16** | 16 |
| num_rollout | 200 | **300** | 300 |
| GPU: rollout | 2 | **5** | 4 |
| GPU: teacher | 4 (TP=4) | **1 (TP=1)** | 1 (TP=1) |
| GPU: actor | 2 (TP=2) | **2 (TP=2)** | 2 (TP=2) |

### Environment Issues Resolved

1. **transformers 5.3.0 + huggingface_hub 1.9.2 path validation bug**: New `huggingface_hub` rejects absolute paths as repo IDs. Fixed by downgrading to `huggingface_hub==1.3.0` (no path validation) while keeping `transformers==5.3.0` (required by sglang 0.5.10 for Qwen3 `rope_parameters`).

2. **Ray port conflict**: Another container (`slime-bh`) had an old Ray cluster occupying port 8265. Our `ray job submit` connected to that cluster (which lacked our volume mounts), causing `os.path.isdir()` to return False for model paths. Fixed by killing the stale Ray cluster.

3. **Batch size mismatch**: `rollout_batch_size=8 × n_samples=4 = 32 < global_batch_size=64`. Fixed by setting `rollout_batch_size=16`.

### Status

Training launched on p5-2. Config: `configs/opd-cumulative.env`, script: `scripts/train-opd.sh`.

---

## 2. Teacher Data Generation

### Setup

- **Goal**: Generate Qwen3-8B teacher responses on 100K new math prompts (from `sft_math_extra_100k_v2.jsonl`)
- **Method**: sglang DP-8 server, `generate-teacher-extra100k.sh`
- **Split**: 50K on p5-5 (shard 0), 50K on p5-2 (shard 1)
- **Parameters**: temp=0.7, top_p=0.9, max_tokens=16384

### Results

| Machine | Progress | Rate | Quality (</think> + \boxed) |
|---------|----------|------|---------------------------|
| p5-5 | ~39K/50K | 2138/hr | 71% |
| p5-2 | **50K/50K** ✅ | 3286/hr | 78% |

Speed difference explained by data distribution (p5-5's shard has longer responses) and sglang version (0.5.9 vs 0.5.10).

### Post-processing needed

Filter to keep only samples with `</think>` + `\boxed{}`. Expected yield: ~72-78K from 100K generated.

---

## 3. SFT Data Preparation

### New 200K Dataset

Created `sft_math_200k_v2.jsonl` (200,039 samples):
- **Original 100K**: `sft_math_100k_v2.jsonl` — from OpenThoughts3, filtered for math + `\boxed{}` + complete think tags + ≤16384 tokens
- **New 100K**: `sft_math_extra_100k_v2.jsonl` — same source, filtered for math + `</think>` + ≤14000 words + no exact (prompt,response) duplicate with original

### Data Quality

- 100% have complete `<think>...</think>` pairs
- Only 12/200K have prompt asking for `\boxed{}` but response missing it (0.006%)
- OpenThoughts3 math domain: ~850K total, ~50% lack `</think>` (truncated during generation)

---

## 4. Lightning-OPD Pipeline Analysis

### Key Findings

| Stage | Lightning-OPD | Our approach |
|-------|--------------|-------------|
| SFT data | 300K prompts, teacher-generated, **no filtering** | 100-200K, filtered (think+boxed) |
| SFT training | LlamaFactory, lr=8e-5, 3000 steps, packing | ms-swift, lr=8e-5 (4B) / 1e-3 LoRA (8B) |
| Rollout collection | vLLM offline, `llm.chat()`, temp=0.7, top_p=0.9 | sglang, manual prompt formatting |
| OPD training | lr=2e-6, batch=256, max_resp=4096, 3000 rollouts | lr=1e-6, batch=64, max_resp=16384, 300 rollouts |
| Teacher consistency | **SFT teacher = OPD teacher** (proven necessary) | ✅ Both use Qwen3-8B |

### Important: Lightning-OPD doesn't filter SFT data

They rely on `packing + cutoff_len=16384` to handle truncated samples, and OPD's reward model to correct errors. Our strict filtering (require `</think>` + `\boxed{}`) may be overly conservative.

### OPD Architecture in slime

```
A_t = A_t^GRPO - λ * KL_penalty_t
```

- GRPO provides reward signal (answer correct/incorrect)
- OPD KL penalty provides teacher guidance (per-token)
- PPO-style clip (ε=0.2) limits policy update magnitude
- `--include-verifiable-reward` adds task reward on top of pure distillation

---

## 5. Model Quality Assessment

### 4B Full FT (checkpoint-253) — FAILED

- Complete degeneration: reasoning starts normally then collapses into repetition loops → garbage (`.printStackTrace`, random chars)
- Even simple factoring (x²-5x+6=0) fails — gets stuck in infinite "But how?" loop
- Root cause: likely unfiltered training data (truncated samples teach model to ramble without concluding)

### 8B LoRA SFT (sft-100k-merged) — PARTIAL SUCCESS

- Without `<think>\n` prefix: model doesn't enter thinking mode (outputs garbage Unicode + raw text)
- With `<think>\n` prefix: model reasons correctly, produces `\boxed{}` answers (78 occurrences in one response), but **never outputs `</think>`** (length explosion)
- First correct `\boxed{}` appears at ~3200 chars, but model continues for 45K+ chars
- Conclusion: model learned reasoning + answer format, but not the termination signal

---

## 6. Infrastructure Notes

### Docker Image Versioning

- p5-5: `slimerl/slime:latest` (older pull) → transformers 4.57.1, huggingface_hub 0.36.2
- p5-2: `slimerl/slime:latest` (newer pull) → transformers 5.3.0, huggingface_hub 1.9.2
- **Lesson**: Pin image versions or use digest, not `:latest`

### Ray + Docker + `--net=host` Gotcha

With `--net=host`, all containers share the network namespace. A Ray cluster in one container can intercept `ray job submit` from another container if they use the same port. Always kill stale Ray clusters or use unique ports.

---

## 7. 4B LoRA Experiments (v7/v8/v9) — Definitive Conclusion

### v7 (lr=1e-3, α=256, +lm_head): Loss Explosion
- Matched 8B recipe exactly, but 4B exploded at step 170 (warmup end)
- Root cause: lm_head (389M params) trained at full lr=1e-3 is too aggressive for hidden_size=2560

### v8 (lr=5e-4, α=256, no lm_head): Stable but 0%
- Training perfect: loss 1.097→0.786, grad_norm 0.03, 6h 42m
- AIME-2024: 0/30 — never generates `</think>`, hits max_tokens every time

### v9 (lr=3e-4, α=32, no lm_head, teacher data): Stable but 0%
- Tinker recipe + teacher-consistent data: loss 0.538→0.275 (near Full FT's 0.246!)
- AIME-2024: 0/30 — same symptoms as v8
- Manual test: reasoning content is correct, but model never outputs `</think>` special token
- Even on training prompts: model reasons correctly but doesn't terminate

### Final Conclusion
`tie_word_embeddings=True` in Qwen3-4B-Base makes LoRA without lm_head fundamentally unable to change output token distribution. The `</think>` special token (id 151668) can never be selected because lm_head weights are frozen (shared with embed_tokens). Only Full FT works for 4B.

---

## 8. Teacher Data Merged (179K)

Both shards completed:
- Shard 0 (p5-5): 50K, 70.4% complete
- Shard 1 (p5-2): 50K, 75.9% complete

**Merged without filtering** (Lightning-OPD style):
```
teacher_sft_filtered.jsonl (79K) + shard0 (50K) + shard1 (50K)
= teacher_sft_179k_merged.jsonl (179,341 samples)
```
Location: p5-4 `/workspace/data/teacher_sft_179k_merged.jsonl`

~85% have complete `</think>`, ~15% truncated. Not filtering because Lightning-OPD doesn't filter either — relies on packing cutoff + OPD reward to handle.

---

## Running Tasks

| Task | Machine | Status | ETA |
|------|---------|--------|-----|
| OPD Cumulative v2 training | p5-2 | 🔄 Running | ~10-15h |
| Teacher generation | p5-5 + p5-2 | ✅ Done | — |
| LoRA v9 eval | p5-4 | ✅ Done (0%) | — |

---

## Next Steps

1. **4B Full FT with 179K data** — lr=8e-5, 3 epochs, ZeRO-1, p5-4 + p5-3 (when available)
2. Monitor OPD v2 training — check metrics
3. Evaluate OPD v2 checkpoint on AIME 2024/2025
4. Consider Lightning-OPD offline approach: precompute teacher logprobs → eliminate live teacher
