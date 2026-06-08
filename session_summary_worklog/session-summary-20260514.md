# Session Summary — 2026-05-14

## Project: K-Step OPD (On-Policy Distillation)

### Goal
Reproduce Lightning OPD pipeline: Qwen3-4B-Base → SFT (teacher-consistent) → OPD with Qwen3-8B teacher.

---

## Key Findings This Session

### 1. LoRA v1 Eval (from previous session) — AIME 0%
- Merged LoRA checkpoint on p5-5, started SGLang server, ran AIME-2024 eval
- Result: **0/30 (0%)** — model never generates `</think>` or `\boxed{}`
- Symptoms: Thai characters at start (`ไว้`, `ฟัง`), degeneration at end
- Confirmed tokenizer/config are correct (rope_theta, vocab_size, chat_template all match base)
- Training data verified: 100% samples have `</think>` + `\boxed{}` within 16K tokens

### 2. Previous "Full FT" Training Was Invalid (Critical Bug)
- Discovered p5-4's `sft_math_100k_v2.jsonl` in `k-step-opd` container had only **10 lines** (stale file)
- `k-step-opd-sft` container had the correct 100K file (same host path, both containers mount same dir)
- **Root cause**: ms-swift `--load_from_cache_file true` cached a tokenized dataset from a previous failed run that only had 10 samples
- All "full FT" checkpoints (v0 through v3, steps 500/1000/1500/3000) trained on **only 10 samples**
- Loss went to 0 (memorized 10 samples), eval 0% (no generalization)
- Verified: host file was always 4.3GB/100K lines, md5 consistent across containers

### 3. Full FT Attempts (All Failed Due to Infrastructure)
- **DDP bs=4**: OOM (75GB/card activations too large for 16K packing)
- **DDP bs=2**: OOM (still too large)
- **DDP bs=1**: Would work but 95s/step = 79 hours
- **DeepSpeed ZeRO-3 bs=1**: 95s/step, too slow
- **DeepSpeed ZeRO-1 bs=1 + Liger**: 83s/step, 28GB/card — works but slow
- **FSDP**: ms-swift 4.1.3 + transformers 5.x incompatible (`SwiftMixin.create_optimizer()` signature mismatch)
- **p5-4 disk full**: root fs 969G/969G (other users' home dirs 691G), couldn't even `docker exec`

### 4. LoRA v3 (Optimized Config) — Still 0%
- Config: r=128, α=128, lr=2e-4, cosine, bs=8, accum=2, 800 steps, liger
- Loss: 1.076 → 0.832 (normal descent), grad_norm stable at 0.025
- AIME-2024: **0/30** — identical symptoms to v1
- **Training set prompts also fail** — model didn't learn format even on seen data
- Confirmed with PEFT direct inference (no merge): same result

### 5. Root Cause Analysis: 4B vs 8B
| Field | 4B-Base | 8B-Base |
|-------|---------|---------|
| hidden_size | 2560 | 4096 |
| **tie_word_embeddings** | **True** | **False** |
| Tokenizer | Identical | Identical |
| Chat template | Identical | Identical |

`tie_word_embeddings=True` means lm_head and embed_tokens are the **same weight matrix**. LoRA targets linear layers but NOT embed_tokens/lm_head → output token distribution never changes → model can't learn to generate `</think>` (token 151668) at the right time.

### 6. Teacher-Consistent Data Generation — Success
8-replica TP=1 async generation with Qwen3-8B:

| Machine | Range | Status | Count |
|---------|-------|--------|-------|
| p5-4 | 0-40K | ✅ Complete | 40000 |
| p5-9 | 40K-80K | ✅ Complete | ~39774 |
| p5-4 | 80K-100K | 🔄 ~65% | ~12958 |

**Data quality** (p5-4 shard_0, 100-sample check):
- Has `</think>`: 79%
- Has `\boxed{}`: 80%
- Answer matches QwQ-32B (exact): 60%
- Answer matches QwQ-32B (sympy equivalent): 63%
- Wrong answer: 22%
- Truncated (no boxed): 15%

**Generation parameters**:
- SGLang TP=1 × 8 engines per node
- Async client: 24 concurrent requests per engine
- temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384
- Prompt: `Question: {original_prompt}\nPlease reason step by step, and put your final answer within \boxed{}.`
- Saved user message: original prompt only (no instruction)
- Stop token: `<|im_end|>`

### 7. SageMaker Quota Check
Checked accounts 654654486179 (Alpha), 211125461623 (Beta), 654654440640 (Gamma):
- **Training job quotas**: All 0 for p5/p4/g6e instances
- **Endpoint quotas**: Alpha has ml.g6e.24xlarge × 5, Gamma × 2
- Reserved capacity (training plans): 256 per instance type but requires purchasing a plan
- Not usable for on-demand training

---

## Currently Running

| Machine | Task | Status | ETA |
|---------|------|--------|-----|
| p5-5 + p5-3 | LlamaFactory LoRA v5 multi-node (2-step smoke test) | 🔄 Just launched | ~5 min |
| p5-4 | Teacher gen 80K-100K (8-replica async) | 🔄 ~65% | ~2h |

---

## Infrastructure State

### Machines
| Machine | IP (private) | GPUs | Container | Status |
|---------|-------------|------|-----------|--------|
| p5-3 | 172.31.12.111 | 8×H100 | `k-step-opd` (slime) + `k-step-opd-sft` (pytorch 2.6) | Multi-node worker |
| p5-4 | 172.31.0.227 | 8×H100 | `k-step-opd` (slime) + `k-step-opd-sft` (pytorch 2.6) | Teacher gen |
| p5-5 | 172.31.6.60 | 8×H100 | `k-step-opd` (slime) + `k-step-opd-sft` (pytorch 2.6) | Multi-node master |
| p5-9 | (Mumbai) | 8×H100 | `slime-training` (slime old) | Teacher gen done |

### Shared Storage
- `/mnt/wutianyi-efs` — EFS shared between p5-5 and p5-3 (confirmed accessible)
- Not currently mounted in containers

### Key Software Versions (k-step-opd-sft container)
- PyTorch 2.6.0, CUDA 12.6
- NCCL 2.21.5 (both p5-5 and p5-3 — verified matching)
- ms-swift 4.1.3
- LlamaFactory 0.9.4 (transformers 4.57.1, peft 0.17.1)
- liger-kernel installed
- flash-attn installed

### NCCL Multi-node Test Results
```
1MB × 20 all_reduce: 4.27s, 4.7 MB/s
128MB × 20 all_reduce: 5.79s, 442 MB/s
1024MB × 20 all_reduce: 12.75s, 1607 MB/s
```
Network is fine. TCP over ethernet ~1.6 GB/s.

---

## Failed Approaches (Don't Repeat)

### ms-swift Multi-node + Packing
- **Deadlocks** because each node packs independently → different dataset sizes → DistributedSampler mismatch
- ms-swift 4.1.3 has no `--packing_cache` parameter
- Official FAQ says: "packing_cache needs to be set to a shared disk path for multi-node training"
- Would need ms-swift upgrade or shared filesystem mount in containers

### ms-swift FSDP
- `TypeError: SwiftMixin.create_optimizer() takes 1 positional argument but 2 were given`
- ms-swift 4.1.3's Trainer override incompatible with transformers 5.x FSDP optimizer path
- Fixed in ms-swift main branch but not in 4.1.3

### Full FT on p5-4
- Root filesystem full (969G, other users' home dirs)
- Even with TMPDIR on NVMe, Docker overlay writes still hit root fs
- DDP bs=2+ OOM with 16K packing on 80GB H100

---

## Speed Benchmarks (Single Node p5-5, 8×H100)

| Framework | Config | Step Time | Memory/GPU | Notes |
|-----------|--------|-----------|------------|-------|
| ms-swift LoRA (r=128, bs=8, accum=2) | no lm_head | 40s | 36-65 GiB | Fastest LoRA |
| ms-swift LoRA (r=128, bs=8, accum=2) | + lm_head | 41s | 64 GiB | lm_head adds minimal overhead |
| LlamaFactory LoRA (r=128, bs=8, accum=2) | + lm_head | 55s | 77 GiB | Slower, more memory |
| ms-swift Full FT ZeRO-1 + Liger (bs=1, accum=32) | — | 83s | 28-32 GiB | Very slow |
| ms-swift Full FT ZeRO-3 (bs=1, accum=32) | — | 95s | 67 GiB | Slowest |

---

## Next Steps (Priority Order)

1. **Verify LlamaFactory multi-node works** (2-step smoke test running now)
   - If works: change max_steps to 800, run full training (~6h)
   - If fails: fall back to single-node ms-swift v5 (41s/step, 9h)

2. **If LoRA + lm_head still 0% on AIME** → abandon LoRA, do full FT
   - Use LlamaFactory full FT (Lightning OPD's proven setup)
   - Need multi-node (4B full FT needs ~32 GPUs for reasonable speed)
   - Or single-node with ZeRO-1 + Liger (83s/step, 69h for 3000 steps)

3. **Complete teacher data generation** (80K-100K range on p5-4)
   - Then merge: p5-4 (0-40K) + p5-9 (40K-80K) + p5-4 (80K-100K)
   - Filter: keep only samples with `</think>` + `\boxed{}` (~80% = ~80K usable)

4. **Teacher-consistent SFT + OPD**
   - SFT: Qwen3-4B-Base + 80K teacher-consistent data (full FT)
   - OPD: same Qwen3-8B as teacher
   - Expected: match Lightning OPD results (~60% AIME-2024)

---

## File Locations

| File | Purpose |
|------|---------|
| `scripts/run-sft-lora-4b-v5.sh` | ms-swift LoRA v5 (lm_head) single node |
| `scripts/run-lf-lora-v5-multinode.sh` | LlamaFactory multi-node launcher |
| `configs/llamafactory/qwen3_4b_lora_v5.yaml` | LlamaFactory LoRA v5 config (max_steps=2 smoke test) |
| `configs/llamafactory/dataset_info.json` | LlamaFactory dataset config |
| `scripts/generate-teacher-8replica.sh` | 8-replica async teacher data gen |
| `scripts/run-sft-full-4b.sh` | Full FT script (DeepSpeed ZeRO-1) |
| `scripts/test_dist.py` | NCCL multi-node connectivity test |
| `sft_lora_v3_curves.png` | LoRA v3 training curves |
| `sft_full_4b_curves.png` | Full FT training curves (invalid — 10 samples) |

---

## Eval Results Summary

| Model | AIME-2024 | Notes |
|-------|:---------:|-------|
| Qwen3-8B (teacher) | 73.3% | max_tokens=32768 |
| Qwen3-4B (post-trained) | 73.3% | Already strong |
| OPD Instant (8B SFT→OPD) | 60.0% | +10pt over SFT baseline |
| OPD Cumulative (8B SFT→OPD) | 53.3% | +3pt |
| SFT-100K 8B-Base (LoRA) | 50.0% | max_tokens=30000 |
| **4B-Base LoRA v1** | **0%** | Never generates `</think>` |
| **4B-Base LoRA v3** | **0%** | Same symptoms |
| **4B-Base "Full FT"** | **0%** | Invalid (10 samples) |
| Qwen3-4B-Base | 0% | No instruction following |
