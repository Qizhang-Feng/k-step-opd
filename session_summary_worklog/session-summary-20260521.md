# Session Summary — 2026-05-21

## Project: K-Step OPD (On-Policy Distillation)

### Goal
Establish strong 4B/8B SFT baselines for OPD experiments.

---

## Key Results This Session

### 8B Full FT 179K — AIME 63.3% ✅

**Best result so far.** Greenland job completed, downloaded to p5-4.

| Benchmark | pass@1 |
|-----------|:------:|
| AIME-2024 | **63.3%** |
| AIME-2025 | **50.0%** |

Config: Qwen3-8B-Base, 179K unfiltered teacher data, 1 epoch, lr=8e-5, global batch=256, ZeRO-1.

---

### 4B Full FT — Degeneration Problem

All 4B experiments with 179K data fail on AIME despite low loss:

| Experiment | Data | Steps | Loss | AIME-2024 |
|---|---|---|---|---|
| **v2 (baseline)** | 79K filtered | 759 | 0.201 | **50%** |
| 179K × 1ep | 179K unfiltered | 487 | 0.264 | 0% |
| 179K × 2ep | 179K unfiltered | 974 | 0.213 | 3.3% |
| 179K × 3ep | 179K unfiltered | 1461 | 0.194 | 3.3% |
| 152K filtered × 3ep | 152K filtered | 1461 | ~0.195 | TBD (running) |

**Degeneration pattern**: Model reasons correctly for ~10-30K chars, then degenerates into garbage (`0 0 0`, `\.printStackTrace`, `\ \ \`). Never generates `</think>` or `\boxed{}` on hard problems.

**Works on**: training data prompts, simple problems (x²-5x+6=0 → correct answer)
**Fails on**: AIME problems, even moderately hard problems

---

## Root Cause Analysis

### What's different between v2 (works) and 152K (fails)?

| | v2 (50%) | 152K 3ep (fails) |
|---|---|---|
| Data | 79K filtered | 152K filtered |
| Data quality | 100% complete | 100% complete |
| Generation params | Same | Same |
| Response length | Mean 38K chars | Mean 39K chars |
| lr/schedule | 8e-5 cosine 0.1 | Same |
| Global batch | 256 | 256 |
| Epochs | 3 | 3 |
| Total steps | 759 | 1461 |
| Total samples seen | 237K | 456K |

**Only meaningful differences**: data size (79K vs 152K) and total steps (759 vs 1461).

### Why 8B works but 4B doesn't?

Key model config difference:
- 4B: `tie_word_embeddings=true` (lm_head shares weights with embed_tokens)
- 8B: `tie_word_embeddings=false`

8B with same 179K unfiltered data → 63.3% AIME. 4B with same data → 3.3%.

### Hypotheses (to investigate)

1. **Cosine LR schedule**: v2 has 759 steps, 152K has 1461 steps. At step 759, v2's lr is near 0 (model "settled"), but 152K's lr is still ~50% peak. More training with higher lr may destabilize the termination behavior.

2. **tie_word_embeddings interaction**: With more data/steps, the shared lm_head/embed_tokens may develop conflicting gradients that hurt termination learning.

3. **v2 may not be reproducible**: 30 AIME problems with temperature=0.6 has high variance. 50% might have been lucky.

4. **Data distribution**: 79K prompts vs 152K prompts — different prompt distribution may matter.

---

## Infrastructure

### New machine: p5-1 (us-west-2/Oregon)
- Added to SSH config (`~/.ssh/dl-machine-ohio.pem` → `~/.ssh/dl-machine-oregon.pem`)
- Currently busy (slime OPD training)

### p5-3 freed up
- zhesu's `agentic_rl_0508` training completed
- Now available for eval

### Greenland
- Fixed `datasets.features.Json` import error (upgraded datasets library in Docker image)
- New image: `k-step-opd-sft:greenland-v2`

---

## Currently Running

| Task | Machine | Status | ETA |
|------|---------|--------|-----|
| 4B Full FT 152K filtered × 3ep | p5-4 | 🔄 step 1324/1461 (91%) | ~2h |
| AIME eval (152K ckpt-1200) | p5-3 | 🔄 running | ~30min |

---

## Eval Results Summary

| Model | AIME-2024 | AIME-2025 | Notes |
|-------|:---------:|:---------:|-------|
| Qwen3-8B (teacher) | 73.3% | 70.0% | max_tokens=32768 |
| **8B Full FT 179K** | **63.3%** | **50.0%** | Best student so far |
| 8B LoRA SFT 100K | 50.0% | 40.0% | Previous baseline |
| 4B Full FT v2 (79K×3ep) | 50.0% | 30.0% | Only working 4B |
| 4B Full FT 179K × 3ep | 3.3% | 6.7% | Degeneration |
| 4B Full FT 179K × 2ep | 3.3% | 3.3% | Degeneration |
| 4B Full FT 179K × 1ep | 0% | — | Degeneration |

---

## Next Steps

1. **Wait for 152K 3ep final eval** — if still fails, consult ChatGPT with `4b-sft-analysis.md`
2. **Investigate v2 reproducibility** — retrain with exact v2 config (79K, 16 GPU, 759 steps) to confirm 50% is real
3. **8B OPD** — 8B SFT is strong (63.3%), ready for Lightning OPD pipeline:
   - Collect 8B student rollouts on DAPO-Math-17k
   - Precompute teacher logprobs
   - Run Lightning OPD training
4. **4B OPD** — Use v2 checkpoint (50%) as student for OPD

---

## File Locations

| File | Purpose |
|------|---------|
| `4b-sft-analysis.md` | Full analysis for ChatGPT consultation |
| `sft_all_experiments_curves.png` | Loss curves for all experiments |
| p5-4: `sft-qwen3-4b-full-152k-3ep/v0-20260521-235133/` | 152K 3ep training |
| p5-4: `sft-qwen3-8b-full-179k-ckpt487/` | 8B final checkpoint |
| p5-4: `sft-qwen3-4b-full-teacher-v2-ckpt759/` | v2 checkpoint (50% AIME) |
| S3: `delphi-greenland-res-alpha/qzf/outputs/ffe02619.../` | 8B Greenland checkpoint |
