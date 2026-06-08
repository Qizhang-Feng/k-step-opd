# Session Summary — 2026-05-23

## Project: K-Step OPD (On-Policy Distillation)
**Status**: SFT investigation closed; **ready to start OPD experiments**.

---

## TL;DR

1. **Diagnosed why 4B SFT on big data (152K / 179K) fails**: it's not budget, not LR-area, not capacity per se — it's **混合 sampling style** in the teacher data.
2. **New 4B SFT baselines**:
   - v2 multi ckpt-700 = **60% AIME-2024** (best, 16 GPU multi-node, 79K-A only)
   - 73K-new ckpt-702 = **53% AIME-2024** (Greenland single-node, 73K-B only)
   - v2 single ckpt-700 = **47%** (8 GPU single-node, replicates v2 within noise)
3. **Loss is uncorrelated with AIME** — confirmed again (152K loss 0.196, 179K loss 0.194 → both 3% AIME; v2 loss 0.201 → 60%).
4. **Per-token KL dumping plumbed into slime** — ready for OPD analysis on token-level reverse KL.
5. **Currently running**: Greenland `4b-152k-700steps` (mix data, same budget as v2) — final ablation to confirm cause.

---

## The big finding: mixed-style teacher data is toxic for 4B

### What we did
Two batches of teacher data, both Qwen3-8B on OpenThoughts3 math prompts:

| Batch | Script | temperature | top_p | Notes |
|---|---|:---:|:---:|---|
| **Style A** (79K filtered) | `scripts/generate-teacher-8replica.sh` | **0.6** | **0.95** | Qwen3 official recommended |
| **Style B** (73K filtered) | `scripts/generate-teacher-extra100k.sh` | **0.7** | **0.9** | Lightning-OPD default |

`teacher_sft_179k_thinkfilter.jsonl` (152K filtered) = `teacher_sft_filtered.jsonl` (79K-A) + `teacher_extra_100k_filtered.jsonl` (73K-B). Numerically: 79,341 + 73,127 = 152,468 ≈ 152,479. Exact composition.

### Prompt overlap analysis (`scripts/check_prompt_overlap.py`)

| Dataset | Total samples | Unique prompts | Avg samples/prompt |
|---|---|---|:---:|
| 79K-A | 79,341 | 20,126 | 3.94 |
| 73K-B | 73,127 | 22,539 | 3.24 |
| **152K-AB** | **152,479** | **24,844** | **6.13** |

- 79K-A ∩ 73K-B = **17,821 prompts** (88% of 79K-A prompts also in 73K-B)
- 152K = 79K-A ∪ 73K-B (no new prompts, just same 25K with more responses)

OpenThoughts3 source itself has duplicate prompts (max 15 responses per prompt; 76% of prompts have ≥2 responses). Our dedup (`prompt[:200]+response[:200]` hash) only deduplicates exact (prompt, response) pairs, **not prompt-level**.

In 152K, some prompts have 18 different responses — many with unique-word ratio 0.07-0.18 (severe repetition; Qwen3 thinking-mode "endless repetition" failure mode at temp=0.7).

### Failure mechanism
- **Single style** (79K-A or 73K-B alone) → 4B learns: each prompt has ~3-4 consistent responses, single distribution.
- **Mixed (A+B)** → 4B fails: same prompt has both temp=0.6 (concentrated) and temp=0.7 (diffuse) traces. 4B capacity insufficient to fit both modes simultaneously → mode collapse on rollout → degeneration into `,,,,,,` `\ \ \ \` `printStackTrace`.
- **8B has enough capacity** → 179K-AB ×1ep → 63% AIME (works on mixed).

### Evidence table

| 实验 | Data | Style | Steps | AIME-2024 |
|---|---|---|:---:|:---:|
| 4B v2 multi ckpt-700 | 79K | A only | 700 | **60%** ⭐ |
| 4B v2 multi ckpt-759 | 79K | A only | 759 | 50% |
| 4B v2 single ckpt-700 | 79K | A only | 700 | 47% |
| 4B 73K-new ckpt-702 | 73K | B only | 702 | **53%** ⭐ |
| 4B 73K-new ckpt-200 | 73K | B only | 200 | 27% |
| 4B 73K-new ckpt-400 | 73K | B only | 400 | 40% |
| 4B 152K ckpt-600 | 152K | A+B mixed | 600 | **0%** |
| 4B 152K ckpt-800 | 152K | A+B mixed | 800 | 3.3% |
| 4B 152K ckpt-1400 | 152K | A+B mixed | 1400 | 3.3% |
| 4B 179K-AB ×1/2/3ep | 179K | A+B mixed | 487/974/1461 | 0/3.3/3.3% |
| **8B 179K-AB ×1ep** | 179K | A+B mixed | 487 | **63.3%** |

### Final loss vs AIME (loss is meaningless)

| 实验 | Final loss | AIME |
|---|:---:|:---:|
| v2 multi ckpt-759 | 0.201 | 50% |
| v2 single ckpt-759 | 0.201 | 47% |
| 73K-B ckpt-702 | 0.209 | 53% |
| 152K-AB ×3ep | **0.196** | 3% |
| 179K-AB ×3ep | **0.194** | 3% |

Mixed-data SFT achieves **lower** loss than single-style v2, but degenerates on rollouts. Loss being lower means model fits the (mode-collapsing) mixture better, but rollout policy is broken.

---

## All checkpoints / experiments

### 4B SFT baselines (sorted by AIME)

| Candidate | AIME-2024 | Where |
|---|:---:|---|
| **v2 multi ckpt-700** | **60%** ⭐ | p5-3: `/opt/dlami/nvme/qzf/models/sft-qwen3-4b-full-teacher-v2/v9-20260515-233350/checkpoint-700` |
| 73K-new ckpt-702 | 53% | p5-3: `/opt/dlami/nvme/qzf/models/sft-qwen3-4b-full-73k-new/checkpoint-702` |
| v2 multi ckpt-759 | 50% | p5-3: same dir, checkpoint-759 |
| v2 single ckpt-700 | 47% | p5-3: `/opt/dlami/nvme/qzf/models/sft-qwen3-4b-full-v2-single/v0-20260523-000055/checkpoint-700` |
| v2 multi ckpt-600 | 40% | p5-3: same v2 multi dir, checkpoint-600 |
| 73K-new ckpt-600 | 47% | p5-3: same 73k-new dir, checkpoint-600 |
| 73K-new ckpt-400 | 40% | same 73k-new dir, checkpoint-400 |
| 73K-new ckpt-200 | 27% | same 73k-new dir, checkpoint-200 |
| 152K ckpt-{600,800,1000,1200,1400,1461} | 0-3% | p5-4: `/opt/dlami/nvme/qzf/models/sft-qwen3-4b-full-152k-3ep/v0-20260521-235133/checkpoint-*`; some on p5-3 too |

### 8B SFT baseline

| Candidate | AIME-2024 | AIME-2025 | Where |
|---|:---:|:---:|---|
| **8B 179K-AB ×1ep ckpt-487** | **63.3%** | 50.0% | p5-4: `/root/.cache/huggingface/sft-qwen3-8b-full-179k-ckpt487`; S3: `s3://delphi-greenland-res-alpha/outputs/ffe02619-dddf-498e-92ec-10fbb7efce89#0/sft-checkpoint/v1-20260518-232752/` |

---

## Code changes

### Per-token reverse KL dump in slime (NEW)

Goal: dump token_ids + student logp + teacher logp + reverse_kl + advantage every N rollouts for offline analysis.

| File | Change |
|---|---|
| `slime/slime/utils/arguments.py` | Added `--opd-dump-kl-path TEMPLATE`, `--opd-dump-kl-interval N`, `--opd-dump-kl-max-samples K` |
| `slime/slime/backends/megatron_utils/actor.py` | Stuff `rollout_id` into `rollout_data` before calling `compute_advantages_and_returns` |
| `slime/slime/backends/megatron_utils/loss.py` | New `_dump_opd_kl()` writes jsonl per-rank; called from `apply_opd_kl_to_advantages` if `--opd-dump-kl-path` set |
| `scripts/train-opd.sh` | Pass `OPD_DUMP_KL_PATH/INTERVAL/MAX_SAMPLES` env vars to CLI flags |
| `configs/opd-4b-v2-ckpt700-instant.env` | NEW. Instant OPD config, student=v2 ckpt-700, dumps KL every 10 rollouts × 8 samples/rank |

#### Dump format (jsonl, 1 line per sample)
```json
{
  "rollout_id": 5,
  "rank": 0,
  "sample_idx": 3,
  "prompt_length": 245,
  "response_length": 4096,
  "reward": 0.0,
  "prompt_token_ids": [151644, 8948, ...],
  "response_token_ids": [151667, 32555, ..., 151668, ...],
  "student_log_probs": [-0.123, -0.045, ...],
  "teacher_log_probs": [-0.098, -0.052, ...],
  "reverse_kl": [-0.025, 0.007, ...],
  "advantage": [-0.025, 0.007, ...]
}
```

Allows offline analysis of: which token positions have largest KL, whether `</think>` / `\boxed{}` tokens have outlier KL, KL distribution evolution over training.

### New training scripts (this session)

- `scripts/run-sft-full-4b-v2-single.sh` — v2 reproduction on 8 GPU single node (79K, 759 steps)
- `scripts/run-sft-full-4b-73k-new.sh` — 4B Greenland on 73K-only (3 epochs, ~702 steps)
- `scripts/run-sft-full-8b-73k-new.sh` — 8B Greenland on 73K-only (capacity check; not submitted)
- `scripts/run-sft-full-4b-152k-700steps.sh` — 4B Greenland on 152K mix capped at 700 steps (running)

### New audit / analysis scripts

| Script | Purpose |
|---|---|
| `scripts/audit_sft_data.py` | Char-level: response length distribution, garbage pattern detection |
| `scripts/audit_sft_tokens.py` | Token-level: tokenization length, `</think>` position before/after max_length |
| `scripts/compare_teacher_data.py` | Style A vs B sample comparison |
| `scripts/check_chat_template.py` | Qwen3 chat template behavior verification |
| `scripts/check_template_round_trip.py` | Raw response vs templated; does `</think>\n\n` survive |
| `scripts/check_prompt_overlap.py` | Prompt set intersection across 79K/73K/152K |
| `scripts/check_ot3_dups.py` | OpenThoughts3 source duplicate-prompt distribution |
| `scripts/analyze_prompt_response_dups.py` | Same-prompt multi-response analysis (the "18 responses per prompt" finding) |
| `scripts/eval-152k-ckpts.sh` | Multi-ckpt eval orchestration for 152K |
| `scripts/eval-73k-new-ckpts.sh` | Multi-ckpt eval orchestration for 73K-new |
| `scripts/eval-v2-ckpts.sh` | Multi-ckpt eval orchestration for v2 |
| `scripts/plot_v2_single_curves.py` | v2 single vs multi training curves |
| `scripts/plot_all_sft_curves.py` | All-experiments single-panel comparison (rewritten) |

### Greenland infrastructure

- `greenland/job_sft_full_4b_73k_new.json` ✅ ran successfully (job ID `ba889b94-68df-43db-b86f-89db26b1cfdc`)
- `greenland/job_sft_full_8b_73k_new.json` (not submitted)
- `greenland/job_sft_full_4b_152k_700steps.json` 🔄 running

S3 uploads (Greenland bucket `delphi-greenland-res-alpha`):
- ✅ `qzf/data/teacher_extra_100k_filtered.jsonl` (2.8 GB, 73K-B data)
- ✅ `qzf/data/teacher_sft_179k_thinkfilter.jsonl` (6.0 GB, 152K mix)
- ✅ `qzf/code/k-step-opd.tar.gz` (latest with all 73k-new + 152k-700steps scripts)

---

## OPD pipeline status (carrying forward from before)

### Existing OPD baselines (before this session)
- **8B Instant OPD on SFT 100K** → AIME-2024 **60%** (+10pt over SFT 50%) ✅ work
- 8B Cumulative OPD v1 (kl=0.05, K=8) → 53.3%
- 8B Cumulative OPD v2 (kl=1.0, K=2, normalized) → unrun/incomplete

### What we have ready for next OPD round
- **4B SFT baselines**: v2 ckpt-700 (60%), 73K-B ckpt-702 (53%) — already on p5-3 NVMe
- **8B SFT baseline**: 179K-AB ckpt-487 (63.3%) — on p5-4 + S3
- **slime with KL dump** — code patched, ready to deploy
- **OPD config**: `configs/opd-4b-v2-ckpt700-instant.env` (uses 4B v2 ckpt-700)
- **Existing OPD config**: `configs/opd-cumulative.env` (8B SFT-100K), `configs/phase2-sft100k-opd.env`

### What's missing
- Convert 4B v2 ckpt-700 to torch_dist for slime training (slime uses Megatron actor)
- Convert 73K-B ckpt-702 to torch_dist (alternative student)
- Convert 8B SFT ckpt-487 to torch_dist (8B OPD path)
- Lightning-OPD offline pipeline: collect rollouts + precompute teacher logprobs (alternative to live-teacher OPD)

---

## Cluster / infrastructure state

### Active machines

| Machine | Region | Status | Use |
|---|---|---|---|
| **p5-3** | us-east-2 | active | All 4B SFT ckpts, slime+sglang containers; main eval/OPD candidate |
| **p5-4** | us-east-2 | active | 152K training (done), teacher data, 8B 179K ckpt; AWS creds fresh |
| p5-2 | us-east-2 | boshih's | not ours |
| p5-5 | us-east-2 | wutianyi's | mostly not ours |
| p5-1 | us-west-2 | czhangzi's container has 3.1 TB writable layer; root fs full | **don't touch** |
| p5-8 | ap-south-1 | GPUs free, root fs 97% full, no slime image; needs docker data-root migration to NVMe | possible setup target |
| p5-10 | ap-south-1 | GPUs free, NVMe 1.2T free, root fs 880G; chenluy's container idle but not impacting | best alternative if p5-3/4 busy |
| p5-11 | ap-south-1 | unknown | not checked recently |

### Containers on p5-3
- `k-step-opd` — slime container (slimerl/slime:latest), Up 18h, has SGLang + Megatron
- `k-step-opd-sft` — pytorch 2.6 + ms-swift container, Up 8 days, used for SFT

### Containers on p5-4
- `k-step-opd-sft` — Up 7 days, used for SFT + ms-swift 4.2.0

### Greenland account info
- Account 654654486179 (`IibsAdminAccess-DO-NOT-DELETE`)
- Auth: `ada credentials update --provider=conduit --account=654654486179 --role=IibsAdminAccess-DO-NOT-DELETE --once --profile=default`
- Bucket: `delphi-greenland-res-alpha` (region us-east-2)
- ECR: `654654486179.dkr.ecr.us-east-2.amazonaws.com/k-step-opd-sft:greenland-v2`

---

## Plots generated this session

| File | Content |
|---|---|
| `sft_all_experiments_curves.png` | Single panel; all 12 SFT loss curves color-coded by data composition |
| `sft_v2_single_vs_multi.png` | v2 single-node vs multi-node loss/grad/lr (3 panels) |

---

## Key insights / lessons

1. **Loss does not predict rollout quality** for SFT-trained reasoning models. Always evaluate on rollout (AIME) every N steps, not just train loss.

2. **Sampling parameters in teacher data generation matter**. Mix of temp=0.6 and temp=0.7 traces is toxic for 4B. Always use the **same** sampling params across teacher data batches.

3. **OpenThoughts3 has heavy prompt duplication** (max 15 responses per prompt; 76% of prompts have ≥2). Naive `(prompt, response)` hash dedup does NOT dedup at prompt level.

4. **Qwen3-4B-Base + tied embeddings**: ChatGPT's earlier hypothesis of capacity issue + tied embeddings as amplifier is consistent with our findings, but the **specific trigger** is mixed-style teacher data, not just data size.

5. **`enable_thinking=True` in Qwen3 chat template** does NOT add `<think>` prefix; it lets the model decide. `enable_thinking=False` injects empty `<think></think>`. Both our generation pipelines (manual `<|im_start|>` text) and Lightning-OPD (vLLM `llm.chat()`) end up at the same point: model sees `<|im_start|>assistant\n` as the generation prompt.

6. **v2 single-node IS reproducible**, but with ~10pt variance vs multi-node on 30-question AIME (60% vs 47% at ckpt-700). Bigger AIME sample (e.g. all 60 problems from 2024+2025) would reduce noise.

7. **Lightning-OPD aligned recipe** (lr=8e-5, cosine, packing, ZeRO-1, full FT, ~3000 steps) is good for 8B. For 4B, we found 700-800 steps is sweet spot if data is single-style; longer training causes degradation in mixed-data regimes.

---

## Next session priorities

### High priority (start of session)
1. **Check 152K-700steps Greenland result** (job submitted late this session)
   - If pass@1 ≥ 30% → "long training is the trigger"; data mix is salvageable with early stopping
   - If pass@1 < 10% → "mix itself is toxic"; need to fix data curation
2. **Decide which 4B SFT to use as OPD student**:
   - v2 multi ckpt-700 (60%, multi-node trained) — highest performance
   - 73K-B ckpt-702 (53%, single-style, Greenland reproducible) — cleanest setup

### Medium priority
3. **Set up OPD training**:
   - Convert chosen 4B SFT ckpt to torch_dist (`tools/convert_hf_to_torch_dist.py`)
   - Run `configs/opd-4b-v2-ckpt700-instant.env` with KL dump enabled
   - Eval ckpts every 25 steps on AIME
4. **Analyze first KL dump** — distribution of reverse KL per token; identify which positions / token types have outlier KL
5. **8B OPD path** (parallel option):
   - Convert 8B SFT ckpt-487 to torch_dist
   - Run instant OPD with KL dump
   - Compare 4B vs 8B OPD trajectories

### Lower priority / nice to have
6. **Re-curate clean SFT data**: regenerate all 100K prompts at temp=0.6/top_p=0.95 with one script, skip dedup-by-pair, do strict dedup-by-prompt → train 4B once on this clean dataset for clean comparison.
7. **Submit 8B 73K-only Greenland job** as capacity control vs 4B 73K (predict ~60-65% AIME).
8. **Lightning-OPD offline path** for 4B: collect rollouts on dapo-math-17k, precompute teacher logprobs, run with `--advantage-estimator on_policy_distillation` (no live teacher server).

---

## File / location reference

### Most useful checkpoints (already on disk)

```
p5-3 (us-east-2):
  /opt/dlami/nvme/qzf/models/
    sft-qwen3-4b-full-teacher-v2/v9-20260515-233350/checkpoint-{600,700,759}/  ← v2 multi
    sft-qwen3-4b-full-v2-single/v0-20260523-000055/checkpoint-{100..759}/      ← v2 single (8 ckpts)
    sft-qwen3-4b-full-73k-new/checkpoint-{100..702}/                           ← 73K-B (8 ckpts)
    sft-qwen3-4b-full-152k-ckpt{600,800,1000,1200,1400}/                       ← 152K (early downloads)
    Qwen3-4B-Base/, Qwen3-8B/, Qwen3-4B/                                       ← base + teacher

p5-4 (us-east-2):
  /opt/dlami/nvme/qzf/data/
    teacher_sft_filtered.jsonl                  ← 79K-A (~3 GB)
    teacher_extra_100k_filtered.jsonl           ← 73K-B (~3 GB)
    teacher_sft_179k_thinkfilter.jsonl          ← 152K mix (~6 GB)
    teacher_sft_179k_merged.jsonl               ← 179K unfiltered (~7.5 GB)
  /opt/dlami/nvme/qzf/models/
    sft-qwen3-4b-full-152k-3ep/v0-20260521-235133/checkpoint-{800..1461}/      ← 152K final ckpts
    sft-qwen3-8b-full-179k-ckpt487/                                            ← 8B SFT (best 8B baseline)

S3 (delphi-greenland-res-alpha, us-east-2):
  qzf/code/k-step-opd.tar.gz                                                   ← latest code tarball
  qzf/code/bootstrap_sft.sh
  qzf/data/teacher_sft_179k_merged.jsonl
  qzf/data/teacher_extra_100k_filtered.jsonl
  qzf/data/teacher_sft_179k_thinkfilter.jsonl
  qzf/models/Qwen3-4B-Base/, Qwen3-8B-Base/, Qwen3-8B/
  outputs/ba889b94-68df-43db-b86f-89db26b1cfdc#0/sft-checkpoint/...            ← 4B 73K-new (Greenland)
  outputs/ffe02619-dddf-498e-92ec-10fbb7efce89#0/sft-checkpoint/...            ← 8B 179K (Greenland)
```

### Code changes (committed / on disk)
```
worklog.md                                          ← appended 2026-05-23 entry
session-summary-20260523.md                         ← this file
sft_all_experiments_curves.png                      ← all 12 SFT loss curves
sft_v2_single_vs_multi.png                          ← v2 single vs multi 3-panel

slime/slime/utils/arguments.py                      ← +3 KL dump flags
slime/slime/backends/megatron_utils/loss.py         ← +_dump_opd_kl(), call site
slime/slime/backends/megatron_utils/actor.py        ← rollout_id stuffed into rollout_data
scripts/train-opd.sh                                ← KL dump env vars
configs/opd-4b-v2-ckpt700-instant.env               ← NEW

scripts/run-sft-full-4b-v2-single.sh                ← NEW
scripts/run-sft-full-4b-73k-new.sh                  ← NEW (Greenland)
scripts/run-sft-full-8b-73k-new.sh                  ← NEW (not submitted)
scripts/run-sft-full-4b-152k-700steps.sh            ← NEW (Greenland, running)

scripts/audit_sft_data.py                           ← NEW
scripts/audit_sft_tokens.py                         ← NEW
scripts/compare_teacher_data.py                     ← NEW
scripts/check_chat_template.py                      ← NEW
scripts/check_template_round_trip.py                ← NEW
scripts/check_prompt_overlap.py                     ← NEW
scripts/check_ot3_dups.py                           ← NEW
scripts/analyze_prompt_response_dups.py             ← NEW
scripts/eval-152k-ckpts.sh                          ← NEW
scripts/eval-73k-new-ckpts.sh                       ← NEW
scripts/eval-v2-ckpts.sh                            ← NEW
scripts/plot_v2_single_curves.py                    ← NEW
scripts/plot_all_sft_curves.py                      ← rewritten

greenland/job_sft_full_4b_73k_new.json              ← NEW (succeeded)
greenland/job_sft_full_8b_73k_new.json              ← NEW (not submitted)
greenland/job_sft_full_4b_152k_700steps.json        ← NEW (running)
```

---

## Open questions for next session

1. Will 152K @ 700 steps work? (Decisive test for "long training" vs "mix itself")
2. Should we use v2 ckpt-700 (60%, multi-node trained) or 73K-B ckpt-702 (53%, single-style, reproducible) for OPD?
3. Live-teacher OPD (current `train-opd.sh`) or Lightning-OPD offline (precomputed logprobs)?
4. Token-level KL dump volume: with default `interval=10, max_samples=8`, expect ~400 MB total over 300 rollouts. OK?
5. Per-token KL analysis tooling: write `scripts/analyze_kl_dumps.py` to plot distributions, find outlier positions, etc.
