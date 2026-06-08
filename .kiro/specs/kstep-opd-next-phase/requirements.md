# Requirements Document

## Introduction

This spec defines the **concrete, executable next phase** of the K-Step Reward-to-Go OPD research project. It is deliberately scoped to what must happen given the experimental results already obtained; it does **not** restate the overarching `research-plan.md` (Phases 0-5).

The project studies whether extending single-step (instant, K=1) On-Policy Distillation into a K-step discounted reward-to-go signal improves long-horizon math reasoning. Confirmed results to date:

- Student baseline (Qwen3-4B Full-FT SFT v2-ckpt700) = 48.8% AIME-2024 (n=16). The bottleneck is SFT data volume (~1/4 of literature).
- Instant OPD (run `opd-4b-B`, paper recipe) reaches iter299 = 55.8% AIME-2024 (+7pt) and saturates by ~300 rollouts.
- The Phase 2.5 "mean" half has finished training; R4 (mean-K=4) iter99 = 54.2% AIME-2024; R1 (mean-K=8) shows instant reverse-KL collapsing to ~0.003 (a mode-collapse danger signal), eval pending.
- KL analysis confirms mean-K accumulated KL ≈ instant KL, so mean-K is **local denoising**, not reward-to-go signal amplification.
- The "sum-K" runs (true textbook reward-to-go) are staged but not yet run: updated `loss.py` (`--opd-agg {sum,mean}`, `--opd-dualclip-c`, token_ids dump fix), hardened reward func, and configs R2/R3/R3b are deployed to machine p5-4.

The literature has refuted naive uniform fixed-K/fixed-γ return-to-go (Revisiting OPD, arXiv 2603.25562: O(T^4) sequence-level variance, γ=1 drift). The project's defensible novelty is therefore narrowed to (a) top-K local support matching to clean the single-step signal, then (b) conditional/gated/adaptive lookahead that couples future KL only when the teacher is trustworthy.

This next phase must: complete Phase 2.5 (mean evaluation, sum-K runs, the 2×2 factorial, and seed replication), apply the decision logic that routes to Phase 3 / pivot / negative-result, and establish the entry criteria and first concrete steps for Phase 3.

**This is a research project whose explicit goal is a publishable paper.** Every Phase 2.5 / Phase 3 outcome — including a clean negative result — must be written up to publication standard. The defensible contribution is the combination of **(a) top-K local support matching to clean the single-step signal** plus **(b) conditional/gated lookahead that couples future KL only when the teacher is trustworthy**. This contribution is positioned against the two anchoring papers: Revisiting OPD (arXiv 2603.25562), which refutes naive fixed-γ return-to-go, and FIPO (arXiv 2603.19835), which supports future-discounted credit assignment when the base signal is clean. Reproducing the known "fixed-k sweep beats naive k=1" result is not a contribution; a negative outcome must instead be framed as strengthening/extending 2603.25562.

## Glossary

- **OPD**: On-Policy Distillation. The student generates rollouts; a teacher scores them via per-token KL used as an RL penalty (slime PPO/GRPO advantage = base reward + OPD KL penalty).
- **Instant OPD (K=1)**: Single-step OPD where each token's penalty is the local reverse-KL `r_t = log π_S(y_t) − log π_T(y_t)`. The confirmed baseline is run `opd-4b-B`, iter299 = 55.8% AIME-2024.
- **Mean-K**: K-step aggregation that divides the truncated discounted KL sum by the actual horizon (`÷actual_k`). Magnitude stays aligned with instant; behaves as local denoising.
- **Sum-K**: K-step aggregation without `÷K` (textbook reward-to-go, full-magnitude future credit assignment). Variance grows with K.
- **Dual-clip mask**: FIPO-style (arXiv 2603.19835) mechanism that removes tokens whose importance-sampling ratio `exp(student_logp − rollout_logp)` exceeds threshold `c` from the future cumulative sum (`--opd-dualclip-c`).
- **Run_R0..R6**: The Phase 2.5 experiment matrix. R0 = instant baseline (opd-4b-B, reused). R1 = mean-K8 no mask. R2 = sum-K8 no mask. R3 = sum-K8 + mask c=10 (main candidate). R3b = mean-K8 + mask. R4 = mean-K4. R5 = sum-K4 + mask. R6 = sum-K8 + mask c=5.
- **2×2 Factorial**: The clean comparison {sum, mean} × {mask on, off} formed by R1/R2/R3/R3b. The main judgement is R3 vs R3b (sum vs mean, mask held constant).
- **OPD_Trainer**: The slime-based training orchestration (SGLang rollout + Megatron training) that executes an OPD run of 300 rollouts.
- **Evaluator**: The evaluation pipeline (convert torch_dist → HF, then SGLang DP=8) that produces AIME scores.
- **avg_pass_at_1**: Noise-reduced AIME metric averaged over 16 samples per problem. The mandated reporting metric (NOT `pass@1[s0]`).
- **KL_Dump**: The per-rollout diagnostic file capturing per-token reverse-KL, token_ids, and positions.
- **Decision_Gate**: The decision logic that interprets Phase 2.5 results and routes the project to Phase 3, a P0 pivot, or a negative-result write-up.
- **Phase3_Entry**: The set of criteria and first implementation steps that begin Phase 3 (top-K local support, then adaptive/gated lookahead).
- **Top-K local support matching**: The Revisiting-OPD single-step fix — truncated reverse-KL over the teacher's top-K support with renormalization, top-p rollout, and special-token masking.
- **Strong single-step baseline**: K=1 + top-K local support matching. The true Go/No-Go opponent (NOT naive sampled-token K=1).
- **Noise band**: The combined uncertainty of n=16 eval (±2-3pt) plus seed standard deviation. An effect must exceed this band to count as real.
- **opd-4b-B recipe**: lr=2e-6, temperature=0.8, max_response_len=4096, 300 rollouts. The fixed recipe reused for comparability.
- **Teacher**: Qwen3-8B (73.3% AIME-2024), served externally (DP=4), shared across runs.
- **Paper_Artifact**: The manuscript-ready bundle comprising results tables, figures, statistical tests, reproducibility metadata, and the contribution/novelty claim.
- **Statistical_Test**: Paired significance testing across prompts (paired bootstrap over problems and/or McNemar's test on exact-match AIME correctness), used in addition to seed mean±std.
- **Strong_Baseline_Suite**: The set of external/literature baselines a publishable result must compare against. At minimum: the instant K=1 OPD baseline (opd-4b-B); the Revisiting-OPD top-K local support single-step baseline; and ideally one further literature arm such as a uniform fixed-k arm shown to be inferior, to demonstrate reproduction of the known negative phenomenon.

## Requirements

### Requirement 1: Complete Mean-K Evaluation

**User Story:** As a researcher, I want the mean-K runs (R1, R4) fully evaluated under the standard protocol, so that I can determine whether pure denoising provides measurable eval value over the instant baseline.

#### Acceptance Criteria

1. WHEN an R1 or R4 training checkpoint at iter99, iter199, or iter299 is available, THE Evaluator SHALL produce an AIME-2024 avg_pass_at_1 score over 16 samples per problem.
2. WHEN an R1 or R4 checkpoint is evaluated, THE Evaluator SHALL also produce an AIME-2025 avg_pass_at_1 score over 16 samples per problem.
3. THE Evaluator SHALL report each mean-K score alongside the student baseline (48.8% AIME-2024) and the instant OPD baseline (55.8% AIME-2024) in a single comparison table.
4. WHILE the R1 instant reverse-KL is at or below 0.003, THE Evaluator SHALL flag the R1 result as a suspected mode-collapse case in the comparison table.
5. IF an R1 or R4 evaluation produces a score whose difference from the instant baseline is within the noise band, THEN THE Evaluator SHALL record the comparison as "no significant difference".

### Requirement 2: Deploy and Execute Sum-K Runs

**User Story:** As a researcher, I want the staged sum-K runs executed on the third machine, so that I can measure true reward-to-go credit assignment against mean-K denoising.

#### Acceptance Criteria

1. WHEN the updated `loss.py` is deployed to machine p5-4, THE OPD_Trainer SHALL pass the `--opd-agg mean` regression test producing token-level values numerically equal to the existing cumulative v2 implementation.
2. THE OPD_Trainer SHALL execute run R3 (sum-K8, dual-clip mask c=10, kl_coef=0.125) for 300 rollouts as the main sum-K candidate.
3. THE OPD_Trainer SHALL execute run R3b (mean-K8, mask c=10) for 300 rollouts as the factorial counterpart to R3.
4. THE OPD_Trainer SHALL execute run R2 (sum-K8, no mask) for 300 rollouts as the bare reward-to-go reference.
5. IF run R2 grad_norm or advantage variance indicates training divergence before iter99, THEN THE OPD_Trainer SHALL stop run R2 early and record the divergence as evidence that sum-K requires the dual-clip mask.
6. WHEN run R3 completes its first training attempt without producing eval improvement over R3b, THE OPD_Trainer SHALL execute a kl_coef mini-sweep over {0.0625, 0.125, 0.25} before any conclusion about sum-K is recorded.
7. WHERE training throughput must be increased, THE OPD_Trainer SHALL use the fast configuration (recompute disabled, max_tokens_per_gpu=16384) and SHALL NOT exceed max_tokens_per_gpu=16384 on the 4B actor.

### Requirement 3: Reuse the Recipe for Comparability

**User Story:** As a researcher, I want every Phase 2.5 run to use the same fixed recipe, so that observed differences are attributable to aggregation choice rather than confounding hyperparameters.

#### Acceptance Criteria

1. THE OPD_Trainer SHALL apply the opd-4b-B recipe (lr=2e-6, temperature=0.8, max_response_len=4096, 300 rollouts) to every Phase 2.5 run.
2. WHERE a run uses sum-K aggregation with horizon K, THE OPD_Trainer SHALL set kl_coef to the instant kl_coef divided by K as the magnitude-alignment starting point.
3. WHEN a sum-K run uses K=8, THE OPD_Trainer SHALL set kl_coef to 0.125.
4. THE OPD_Trainer SHALL use the shared external Qwen3-8B teacher (DP=4) for every Phase 2.5 run.
5. WHILE a run is training, THE OPD_Trainer SHALL record the OPD penalty magnitude (`opd_kl_coef × cumulative_kl`) so that alignment with the instant baseline penalty magnitude can be verified.

### Requirement 4: KL Diagnostics with Token Identity

**User Story:** As a researcher, I want KL dumps that include token identity and position, so that I can perform position-wise and special-token failure-mode diagnostics.

#### Acceptance Criteria

1. WHEN a Phase 2.5 run writes a KL_Dump, THE KL_Dump SHALL include the per-token reverse-KL, the token_ids, and the token position for each recorded token.
2. IF a KL_Dump is written with an empty token_ids field, THEN THE OPD_Trainer SHALL treat the dump as invalid and report the diagnostic configuration as broken.
3. THE OPD_Trainer SHALL record advantage variance and grad_norm trajectory for every Phase 2.5 run.
4. WHEN a run uses a dual-clip mask, THE OPD_Trainer SHALL record the fraction of tokens removed by the mask.

### Requirement 5: Standardized Evaluation Protocol

**User Story:** As a researcher, I want a single fixed evaluation protocol, so that all run comparisons are valid and noise-controlled.

#### Acceptance Criteria

1. THE Evaluator SHALL evaluate every checkpoint with max_tokens=30000, temperature=0.6, and top_p=0.95.
2. THE Evaluator SHALL produce scores by converting the torch_dist checkpoint to HF format and serving via SGLang with DP=8.
3. THE Evaluator SHALL report avg_pass_at_1 over 16 samples per problem as the comparison metric for every run.
4. THE Evaluator SHALL exclude pass@1[s0] from cross-run comparison tables.
5. THE Evaluator SHALL evaluate checkpoints at iter99, iter199, and iter299 for every run so that the score-versus-rollout curve can be plotted.

### Requirement 6: Seed Replication to Clear the Noise Band

**User Story:** As a researcher, I want the core comparison runs replicated across seeds, so that effects of 1-3 absolute points are distinguishable from noise.

#### Acceptance Criteria

1. THE OPD_Trainer SHALL execute runs R1, R3, and R3b at 2 distinct random seeds each.
2. THE Evaluator SHALL report mean and standard deviation across seeds for runs R1, R3, and R3b.
3. WHERE a run is a sensitivity arm (R2, R4, R5, R6), THE OPD_Trainer SHALL execute a single seed first and SHALL add a second seed only after a signal exceeding the noise band is observed.
4. IF the difference between two compared runs does not exceed the noise band (n=16 eval uncertainty plus seed standard deviation), THEN THE Decision_Gate SHALL classify the comparison as "no significant difference".

### Requirement 7: 2×2 Factorial Analysis

**User Story:** As a researcher, I want a clean 2×2 factorial analysis of aggregation and masking, so that I can isolate credit assignment from denoising and from mask effects.

#### Acceptance Criteria

1. THE Decision_Gate SHALL compute the R3-versus-R3b comparison (sum vs mean, mask held on) as the primary credit-assignment judgement.
2. THE Decision_Gate SHALL compute the R3-versus-R2 comparison (mask on vs off, sum held) to quantify the mask's effect on sum-K variance.
3. THE Decision_Gate SHALL compute the R1-versus-R3b comparison (mask on vs off, mean held) to quantify the mask's effect on mean-K.
4. THE Decision_Gate SHALL compute the R1-versus-instant-baseline comparison to quantify the eval value of mean-K denoising.
5. THE Decision_Gate SHALL present the four comparison results in a 2×2 table with seed-aware mean and standard deviation per cell.

### Requirement 8: Phase 2.5 Decision Routing

**User Story:** As a researcher, I want explicit decision logic that maps Phase 2.5 outcomes to next steps, so that the project advances, pivots, or concludes honestly based on evidence rather than intuition.

#### Acceptance Criteria

1. WHEN R3 exceeds R3b by more than the noise band, THE Decision_Gate SHALL conclude that the reward-to-go story holds and SHALL route the project to Phase3_Entry.
2. WHEN sum-K runs fail to exceed their mean-K counterparts AND mean-K does not exceed the instant baseline by more than the noise band, THE Decision_Gate SHALL classify the outcome as a negative result and SHALL route the project to the P0 top-K local support pivot.
3. WHEN mean-K exceeds the instant baseline by more than the noise band AND sum-K does not exceed its mean-K counterpart, THE Decision_Gate SHALL reframe the contribution as "local denoising helps" rather than reward-to-go credit assignment.
4. THE Decision_Gate SHALL record the selected route together with the supporting comparison numbers from the 2×2 factorial.
5. IF a route is selected while run R3 has not yet completed its kl_coef mini-sweep after a failed first attempt, THEN THE Decision_Gate SHALL defer the negative-result classification until the mini-sweep is complete.

### Requirement 9: Phase 3 Entry — Top-K Local Support Foundation

**User Story:** As a researcher, I want the first concrete Phase 3 step to implement the strong single-step baseline, so that any future lookahead method is measured against a clean, trustworthy single-step signal.

#### Acceptance Criteria

1. WHILE Phase3_Entry is active, THE OPD_Trainer SHALL obtain the teacher's top-K logprobs per prefix via the SGLang `top_logprobs_num` rollout parameter.
2. THE OPD_Trainer SHALL compute the single-step signal as a truncated reverse-KL over the teacher top-K support with support-set renormalization on both distributions.
3. THE OPD_Trainer SHALL apply top-p rollout sampling at p=0.9 when collecting rollouts for top-K local support matching.
4. THE OPD_Trainer SHALL apply special-token masking to the top-K local support signal.
5. WHEN the top-K local support baseline is first evaluated, THE Evaluator SHALL confirm the ordering sampled-token K=1 < K=1+mask < K=1+top-K local support to reproduce the Revisiting-OPD trend.
6. WHERE the support size K must be chosen, THE OPD_Trainer SHALL use a support size between 20 and 50.

### Requirement 10: Phase 3 Go/No-Go Against the Strong Baseline

**User Story:** As a researcher, I want the true Go/No-Go gate defined against the strong single-step baseline, so that conditional/adaptive lookahead is only adopted if it provides genuine incremental value.

#### Acceptance Criteria

1. THE Decision_Gate SHALL compare conditional/adaptive lookahead against the strong single-step baseline (K=1 + top-K local support), NOT against naive sampled-token K=1.
2. WHEN adaptive lookahead exceeds the strong single-step baseline by at least 1 absolute point on a long-horizon reasoning benchmark under matched compute, AND response length, repetition rate, and advantage variance remain controlled, THE Decision_Gate SHALL classify the result as a Go.
3. IF adaptive lookahead fails to exceed the strong single-step baseline by at least 1 absolute point, THEN THE Decision_Gate SHALL route the project to either a stronger gating signal or a negative-result write-up.
4. THE Decision_Gate SHALL require that any Go decision is supported by seed-aware mean and standard deviation across at least 2 seeds.

### Requirement 11: Publication-Grade Contribution Framing

**User Story:** As researchers aiming to publish, I want a clearly stated, defensible novelty claim, so that reviewers immediately see what is new versus prior art.

#### Acceptance Criteria

1. THE Paper_Artifact SHALL state a single-sentence contribution claim.
2. THE Paper_Artifact SHALL position the contribution claim explicitly against Revisiting OPD (arXiv 2603.25562) and FIPO (arXiv 2603.19835).
3. THE Paper_Artifact SHALL frame the contribution as the combination of top-K local support matching plus conditional/gated lookahead, rather than as "fixed-k sweep beats naive k=1".
4. WHEN the Phase 2.5 outcome is a negative result, THE Paper_Artifact SHALL frame the negative result as a contribution that strengthens and extends arXiv 2603.25562 (future coupling is not worthwhile even under a clean single-step signal).

### Requirement 12: Statistical Rigor for Publication

**User Story:** As a researcher, I want significance testing beyond seed averages, so that every claimed gain is defensible.

#### Acceptance Criteria

1. WHEN two runs are compared for a published claim, THE Statistical_Test SHALL apply prompt-level paired testing (paired bootstrap over problems and/or McNemar's test on exact-match correctness) in addition to seed mean±std.
2. THE Paper_Artifact SHALL report mean ± standard deviation over the run's seeds for every headline number.
3. IF a gain does not pass both the noise band (Requirement 6) and the Statistical_Test, THEN THE Decision_Gate SHALL NOT report the gain as a positive result.

### Requirement 13: Strong Baseline Comparison

**User Story:** As a researcher, I want comparisons against literature-grade baselines, so that the result is competitive and not just self-referential.

#### Acceptance Criteria

1. THE Paper_Artifact SHALL report the method against the Strong_Baseline_Suite.
2. THE Paper_Artifact SHALL include the instant K=1 OPD baseline (opd-4b-B) and the top-K local support single-step baseline in the comparison.
3. THE Paper_Artifact SHALL report a uniform fixed-k arm to demonstrate reproduction of the known bias-variance phenomenon from arXiv 2603.25562.
4. THE Evaluator SHALL evaluate every baseline in the Strong_Baseline_Suite under the identical standardized protocol defined in Requirement 5.

### Requirement 14: Reproducibility and Artifact Release

**User Story:** As a researcher, I want the experiments reproducible, so that the paper's claims can be independently verified and the artifact released.

#### Acceptance Criteria

1. THE Paper_Artifact SHALL record, for every reported run, the exact recipe (lr, temperature, max_response_len, rollout count), the aggregation/mask configuration (opd-agg, opd-dualclip-c, K, kl_coef), the seeds, and the code revision.
2. THE Paper_Artifact SHALL retain the iter99, iter199, and iter299 score-versus-rollout curves and the KL diagnostics (including token_ids and position) as supporting evidence.
3. THE Paper_Artifact SHALL preserve the converted HF checkpoints, OR the torch_dist checkpoints together with the conversion command, for every headline run.
