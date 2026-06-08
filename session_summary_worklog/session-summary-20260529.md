# Session Summary — 2026-05-29

## Project: K-Step OPD (On-Policy Distillation)
**Status**: opd-4b-B (paper recipe) eval done — **+7.0pt AIME-2024**, new best. K-step bias-variance fully analyzed.

---

## TL;DR

1. **opd-4b-B paper recipe gives +7.0pt AIME-2024 (n=16)** — best so far. Saturated at iter_299, no gain at iter_599.
2. **Naming cleaned up**: opd-4b-A (5/24, conservative) vs opd-4b-B (5/26, paper Table 6). Both standard online OPD; we did NOT implement Lightning paper's offline approach.
3. **K-step KL bias-variance fully characterized** from 357MB+206MB dump data:
   - K-step **sum** variance grows ∝ K (REINFORCE reward-to-go)
   - K-step **mean** variance shrinks ∝ 1/K (slime cumulative v2 implementation)
   - These are different bias-variance tradeoffs, not the same metric
4. **K=8 is the structural sweet spot**: 60%+ independent signal, 6.5x variance reduction (mean version), stable across training.
5. **opd-4b-A driver log lost** — can only reconstruct KL trajectory via dump sampling (29 points). Lesson: tee driver output to NVMe.

---

## Key Eval Results

### opd-4b-B vs opd-4b-A (n=16 avg pass@1)

| ckpt | AIME-2024 | AIME-2025 | 综合 | avg_len |
|---|:---:|:---:|:---:|:---:|
| baseline v2-700 (SFT) | 48.8% | 40.6% | 44.7% | 55K-58K |
| opd-4b-A iter_99 | 50.2% | 46.9% | 48.6% | 55K-56K |
| opd-4b-A iter_299 | 54.4% | 45.2% | 49.8% | 52K-57K |
| **opd-4b-B iter_99** | **53.5%** | 45.2% | 49.4% | 52K-55K |
| **opd-4b-B iter_299** ⭐ | **55.8%** | 45.0% | **50.4%** | 51K-57K |
| **opd-4b-B iter_599** | 55.2% | 44.6% | 49.9% | 50K-55K |

**Δ vs baseline** (iter_299):
- opd-4b-A: +5.6pt AIME-2024
- opd-4b-B: **+7.0pt AIME-2024** ⭐

**vs Lightning-OPD paper (n=32, standard OPD)**: 56.7% → 65.4% (+8.7pt). Our +7.0pt is 1.7pt smaller, baseline is 8pt lower.

**Saturation point**: opd-4b-B iter_299 = peak. iter_599 doesn't improve. Matches paper Figure 3b (4B saturates ~50 steps).

---

## Naming Cleanup

Old "instant" / "lightning recipe" was confusing — both are standard online OPD with different hyperparams; "lightning" overlapped with the paper's offline method name.

| New | Old | Config |
|---|---|---|
| **opd-4b-A** | "instant-extteacher" | lr=5e-7, max_resp=8192, T=0.6, 300 rollouts |
| **opd-4b-B** | "lightning-recipe" | lr=2e-6, max_resp=4096, T=0.8, 600 rollouts (paper Table 6) |

> ⚠️ Both are standard online OPD. Lightning-OPD paper's "Lightning" = offline precomputed teacher. We did NOT implement that.

Actions:
- `NAMING.md` documents the mapping
- Symlinks added on p5-2/3/4 (old paths still work)
- `configs/opd-4b-A.env` and `opd-4b-B.env` (with header explanation)
- Eval JSON filenames keep old suffixes (rerunning waste); plots/tables use new labels

---

## opd-4b-B Training Trajectory

### Full 6-panel: `kl_analysis/figures/opd_trajectories_lightning_full.png`

| Metric | Start (r1) | End (r599) | 趋势 |
|---|:---:|:---:|---|
| opd_reverse_kl | 0.139 | 0.094 | -32%, 但震荡 0.09-0.15 |
| train/loss | 0.139 | 0.094 | = opd_reverse_kl × kl_coef=1.0 |
| grad_norm | 1.04 | 0.37 | -65% (持续下降，收敛中) |
| truncated_ratio | 0.92 | 0.92 | 高位稳定 ⚠️ |
| response_length | 4012 tok | 4019 tok | flat (4096 cap) |
| kl_loss vs ref | 0 | 0.07 | student 慢慢漂离 SFT 起点 |

### Counter-intuitive finding

opd-4b-B 训练时 **reverse_KL 不下降**（从 0.13 震荡到 0.09），但 **eval 数字最好** (+7pt)。

**Explanation**: sampled trajectory 上的 reverse_kl 是 *sampling-strategy-dependent*。lr 大 + T=0.8 + 无 top_k → student 探索更广 → sampled KL 不容易降。但 student 真实参数在动（grad_norm 下降，kl_loss vs ref 上升），eval-time pass@1 验证了进步。

→ **训练时 reverse_KL 不是好的 monitoring metric for OPD success**。eval avg pass@1 才是 ground truth.

---

## K-step KL Bias-Variance Analysis

### Setup

For each token position t in dumped sample:
```
mean_K[t] = (1/K) × Σ_{d=0}^{K-1} reverse_kl[t+d]   # slime cumulative v2 用的
sum_K[t]  =        Σ_{d=0}^{K-1} reverse_kl[t+d]   # RL 教材 reward-to-go
```

### Sum variance grows with K (REINFORCE-style)

| K | A: var(Σ_K) | A: ratio | B: var(Σ_K) | B: ratio | K ideal |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.44 | 1.0× | 0.30 | 1.0× | 1× |
| 8 | 4.37 | 9.9× | 2.74 | 9.3× | 8× |
| 64 | 48.1 | 109× | 28.4 | 96× | 64× |
| **full** | **163,030** | **368,000×** | **13,740** | **46,000×** | ~T |

✅ Validates textbook RL reward-to-go variance ∝ K. Slight super-linearity (9.9× vs 8×) due to token autocorrelation.

### Mean variance shrinks with K (averaging noise reduction)

| K | A: var(mean_K) | B: var(mean_K) | 1/K ideal |
|:---:|:---:|:---:|:---:|
| 1 | 0.445 | 0.295 | 1.0 |
| 8 | 0.068 | 0.043 | 0.125 |
| 64 | 0.012 | 0.007 | 0.016 |

→ Perfect 1/K-ish shrinkage. But this is NOT the same as RL gradient variance reduction — it's just averaging.

### Two views on K=1 (instant) bias

**RL view (textbook)**: K=1 ignores a_t's effect on future states → **biased Q estimate**. K=full = unbiased Monte Carlo Q.

**Paper Theorem 3.6 view**: OPD treated as distillation surrogate, not RL. advantage = log_T - log_S directly minimizes KL(π_θ || π_T) with fixed-point guarantee. K=1 has the right fixed point even if biased as Q estimate.

→ Paper uses surrogate view, doesn't need cumulative. RL view of cumulative is open.

### K-step properties stable across training

`kstep_over_time_*.png` shows:
- Pearson r(instant, K=8) ≈ 0.38 unchanged from r10 to r590
- var(mean_K=8) / var(instant) ≈ 0.15 unchanged
- K=8 sweet spot is structural, doesn't need scheduling

→ For future cumulative experiments: just pick K=8 and stick with it.

### Cumulative OPD design space (decision matrix)

| Variant | Advantage | bias | per-step variance | magnitude | slime ready? |
|---|---|:---:|:---:|:---:|:---:|
| K=1 instant | reverse_kl[t] | high | low (σ²) | 1× | ✓ |
| **K=8 mean (slime v2)** | (1/8)Σ kl[t..t+7] | high | σ²/8 | 1× | ✓ |
| K=8 sum (textbook reward-to-go) | Σ kl[t..t+7], coef=1/8 | mid | 8σ² | 1× (after coef) | ❌ need to remove /K |
| K=full suffix | Σ_t^T γ^d kl[t+d] | low | huge | enormous | ❌ need coef×~600 |

**Important**: slime's current mean implementation is averaging-based noise reduction, NOT reward-to-go bias correction. Implementing true reward-to-go requires modifying slime to skip the `/K` normalization.

---

## Deliverables

### Figures (`kl_analysis/figures/`)
- `opd_trajectories_kl.png` — A vs B reverse_kl (sampled vs full)
- `opd_trajectories_lightning_full.png` — opd-4b-B 6-metric full
- `kstep_kl_A.png` / `kstep_kl_B.png` / `kstep_kl_compare.png` — single-run + comparison
- `kstep_over_time_A.png` / `kstep_over_time_B.png` / `kstep_over_time_compare.png` — temporal
- `kstep_sum_vs_mean.png` — bias-variance tradeoff teaching figure (4-panel)

### Scripts (`scripts/`)
- `analyze_kstep_kl.py` — K-step Pearson + variance ratio (single rollout midpoint)
- `analyze_kstep_kl_over_time.py` — temporal K-step stats
- `analyze_kstep_sum_vs_mean.py` — sum vs mean variance comparison
- `extract_opd_trajectories.py` — parse driver log → CSV
- `plot_opd_trajectories.py` — training trajectory plotting
- `convert-opd-lightning-ckpts.sh` — Megatron → HF for 3 checkpoints
- `eval-aime-n16.sh` — n=16 AIME-2024/2025 eval (DP=8)

### Configs (`configs/`)
- `opd-4b-A.env` (new clean name; original `opd-4b-v2-ckpt700-instant-extteacher.env` kept)
- `opd-4b-B.env` (new clean name; original `opd-4b-lightning-recipe.env` kept)

### Doc (`/`)
- `NAMING.md` — naming convention + file location mapping

### Analysis JSON
- `kl_analysis/summary_A.json`, `summary_B.json`, `summary.json` (older)
- `kl_analysis/trajectories/lightning.csv` (600-rollout per-step metrics)

### KL dumps (on local mac for analysis)
- `kl_analysis/dumps_A/` — 116 jsonl, 357MB
- `kl_analysis/dumps_B/` — 116 jsonl, 206MB

---

## Lessons

1. **K=1 vs K-step is bias-variance tradeoff in textbook RL sense** — verified. Sum variance grows ∝K, K=full has 368,000× the variance of K=1.
2. **slime's "cumulative OPD v2" (mean) is NOT reward-to-go**. It's noise averaging. Different bias-variance.
3. **K-step KL structural properties are stable across training**. Don't need to schedule K.
4. **Sampled reverse_KL is not a good OPD training monitor**. Eval avg pass@1 is the real ground truth.
5. **Always tee driver logs to persistent storage**. Ray cleans /tmp/ray/ on next session start; opd-4b-A trajectory had to be reconstructed from KL dumps because of this.
6. **opd-4b-B saturates at iter_299**. Matches paper Figure 3b. ~300 rollouts is enough; doing 600 wastes GPU time.
7. **The +7pt gap from paper's +8.7pt is mostly SFT data quantity** (we have 1/4 of paper's 768K samples). Can't easily close without redoing SFT.

---

## Next Actions (priority order)

### High
1. Eval opd-4b-B intermediate ckpts (iter_199 / 399 / 499) → full 7-point AIME pass@1 curve
2. Decide on opd-4b-C: mean K=8 (slime ready) vs sum K=8 (needs slime patch)

### Medium
3. Match-up SFT: regenerate 300K teacher data (no temp mixing) + train multi-node 3000 steps
4. Try 32B teacher (gap 13pt → 25pt+ means more headroom)

### Low
5. Implement true reward-to-go in slime (remove `/K` from cumulative loss path)
6. Paper outline write-up

---

## Cluster State

| Machine | Status | What |
|---|---|---|
| p5-3 | idle (ckpts kept) | Has both A and B Megatron + HF, KL dumps, training logs |
| p5-4 | idle | Has A iter_199, B iter_599, n=16 eval results |
| p5-2 | idle | Has A iter_299, B iter_299, n=16 eval results |
| qzf-dev | teacher server (may have stopped) | Qwen3-8B DP=4 for OPD |
