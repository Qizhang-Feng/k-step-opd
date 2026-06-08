# Session Summary — 2026-05-16

## Project: K-Step OPD (On-Policy Distillation)

### Goal
Reproduce Lightning OPD pipeline: Qwen3-4B-Base → SFT (teacher-consistent) → OPD with Qwen3-8B teacher.

---

## Key Achievement This Session

### 4B Full FT SFT Successfully Trained — AIME 50% ✅

After multiple failed attempts (LoRA v1/v3/v5 all 0%, Full FT v1 ~0%), finally got a working 4B SFT:

| Model | AIME-2024 | AIME-2025 | Method |
|-------|:---------:|:---------:|--------|
| Qwen3-8B (teacher) | 73.3% | 70.0% | post-trained |
| 8B-Base LoRA SFT | 50.0% | 40.0% | LoRA r=128 |
| **4B-Base Full FT v2** | **50.0%** | **30.0%** | **Full FT, 3 epochs, filtered data** |

**What made it work**:
1. **Full fine-tuning** (not LoRA) — 4B's `tie_word_embeddings=True` makes LoRA insufficient
2. **Filtered teacher data** — removed 20% truncated samples (no `</think>`/`\boxed{}`)
3. **3 epochs** — 1 epoch was not enough (model degenerated on long sequences)

---

## Training Details

**4B Full FT v2 Config**:
- Model: Qwen3-4B-Base
- Data: `teacher_sft_filtered.jsonl` (79,341 samples, Qwen3-8B generated, filtered for completeness)
- Framework: ms-swift 4.2.0, DeepSpeed ZeRO-1
- Multi-node: p5-3 (master) + p5-4 (worker), 16×H100
- lr=8e-5, cosine, warmup=0.1, 3 epochs
- bs=8, accum=2, global batch=256 (same as Lightning-OPD)
- packing=True, max_length=16384, gradient_checkpointing, liger_kernel, flash_attn
- 759 steps, 36s/step, 7h 35m total
- Final loss: 0.246

**Checkpoint**: p5-4: `/root/.cache/huggingface/sft-qwen3-4b-full-teacher-v2-ckpt759/`

---

## Currently Running

| Machine | Task | Status | ETA |
|---------|------|--------|-----|
| p5-9 | 8B SFT rollout collection (DAPO-Math-17k) | 🔄 Running | ~3-4h |

---

## Infrastructure State

### Machines
| Machine | Status | Notes |
|---------|--------|-------|
| p5-3 | Available | Has 4B Full FT v2 checkpoint (master) |
| p5-4 | Available | Docker on NVMe (88G root free), fresh container, has checkpoint copy |
| p5-5 | Partially occupied | ccrchen's vLLM on GPU 1 |
| p5-9 | Busy | Rollout collection running |

### Key Changes This Session
- **p5-4 Docker migrated to NVMe**: `/var/lib/docker` → `/opt/dlami/nvme/docker` (symlink), root fs 88G free
- **p5-4 container rebuilt**: Fresh `k-step-opd-sft` with ms-swift 4.2, deepspeed, flash-attn, liger
- **ms-swift 4.2 multi-node packing works**: PackingDataset broadcasts from master, no deadlock
- **SSH keys deployed**: p5-2 and p5-4 can SSH to each other via private IPs

### Software Versions (k-step-opd-sft containers)
- ms-swift 4.2.0, PyTorch 2.6.0+cu126, NCCL 2.21.5
- DeepSpeed 0.19.0, flash-attn 2.8.3, liger-kernel 0.8.0

---

## Failed Approaches (Don't Repeat)

### 4B LoRA (any variant) for AIME
- `tie_word_embeddings=True` → LoRA can't change output distribution
- Even with `modules_to_save lm_head`, LoRA capacity insufficient for long reasoning
- Simple math works (7/10) but AIME degenerate after ~13K chars

### Full FT with unfiltered data (1 epoch)
- 20% of teacher data is truncated (no `</think>`) → model learns "thinking can be infinite"
- 1 epoch not enough training → degenerate on hard problems

### Multi-node data inconsistency
- Both nodes MUST have identical data files (md5 match)
- PackingDataset broadcasts indices from master → worker IndexError if dataset sizes differ

### p5-4 root fs issues
- Other users' /home = 860GB, Docker overlay = 130GB → 0 bytes free
- Solution: migrate Docker to NVMe (`/var/lib/docker` → symlink to NVMe)
- datasets arrow cache writes to root fs → set TMPDIR/HF_DATASETS_CACHE/HF_HOME to NVMe

---

## Lightning OPD Pipeline Status

```
Step 1: Generate SFT Data ✅ (teacher_sft_filtered.jsonl, 79K)
Step 2: SFT Training ✅ (4B Full FT v2, AIME 50%)
Step 3: Collect Student Rollouts 🔄 (p5-9, DAPO-Math-17k, 8 engines)
Step 4: Precompute Teacher Logprobs (TODO)
Step 5: Lightning OPD Training (TODO)
Step 6: Convert & Eval (TODO)
```

### Next Steps (Priority Order)

1. **Wait for rollout collection** (p5-9, ~3-4h)
2. **Precompute teacher logprobs**:
   - Start Qwen3-8B sglang server (TP=1 or TP=2)
   - Run `prepare_lightning_opd.py --compute-teacher-logprobs`
   - Output: parquet with `teacher_log_probs` in metadata
3. **Run Lightning OPD**:
   - Use Lightning-OPD repo's training config
   - 8 GPU all for actor (no teacher server needed)
   - `--advantage-estimator on_policy_distillation`
   - lr=2e-6, global_batch=256, num_rollout=3000
   - Expected: 60%+ AIME-2024
4. **Also try**: Lightning OPD on 8B SFT model (already have rollouts collecting)

---

## Eval Results Summary

| Model | AIME-2024 | AIME-2025 | Notes |
|-------|:---------:|:---------:|-------|
| Qwen3-8B (teacher) | 73.3% | 70.0% | max_tokens=32768 |
| 8B OPD Instant | 60.0% | — | +10pt over SFT |
| 8B OPD Cumulative | 53.3% | — | +3pt |
| 8B-Base LoRA SFT | 50.0% | 40.0% | max_tokens=30000 |
| **4B-Base Full FT v2** | **50.0%** | **30.0%** | **3 epochs, filtered data** |
| 4B-Base Full FT v1 | ~0% | — | 1 epoch, unfiltered |
| 4B-Base LoRA v5 | 0% | — | + lm_head, degenerate |
| 4B-Base LoRA v1/v3 | 0% | — | no lm_head |

---

## File Locations

| File | Purpose |
|------|---------|
| `scripts/run-sft-full-4b-multinode.sh` | Full FT training script (final) |
| `scripts/collect-rollouts-8b-sft.sh` | Rollout collection script |
| `sft_full_4b_teacher_v2_curves.png` | Training curves |
| p5-4: `/root/.cache/huggingface/sft-qwen3-4b-full-teacher-v2-ckpt759/` | 4B Full FT v2 checkpoint |
| p5-3: `/root/.cache/huggingface/sft-qwen3-4b-full-teacher-v2/v9-20260515-233350/checkpoint-759` | Original checkpoint location |
| p5-9: `/root/.cache/huggingface/rollouts-8b-sft/` | 8B rollout output (in progress) |
| p5-4/p5-3: `/workspace/data/teacher_sft_filtered.jsonl` | Filtered teacher data (79K) |
| `Lightning-OPD/` | Lightning OPD repo with pipeline code |
