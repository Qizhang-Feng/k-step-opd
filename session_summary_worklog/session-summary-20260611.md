# Session Summary — 2026-06-09/11 (Day ~40-42)

## Project: K-Step OPD (On-Policy Distillation)
**Status**: Phase 2.5 完整 eval 完成 + KL dump 系统性诊断 (lift/autocorr/spike clustering) + R5 跑完 eval + IS ratio dormant 诊断 + OPD vs multi-step PPO 本质冲突 + Phase 3 Hybrid-K OPD candidate (RB future) 设计完。新发现：3 个数学等价 run 实测 8pt spread → single-seed noise band ≈ ±4pt，**R1 +3.34pt finding 降级到 "等 multi-seed 验证"**。

---

## TL;DR

1. **R1 是 Phase 2.5 winner**：mean K=8 + kl_coef=1.0 + 不带 mask flag。AIME-24 iter299 = **59.17%** (n=16), AIME-25 = **48.33%**, pass@any-24 = **83.33%** > baseline 76.67% → **不是 mode collapse**。比 instant baseline (opd-4b-B = 55.83) +3.34pt。
2. **R3 / R5 / R3b 数学等价 R1 但实测分散 8pt**：mask 不触发 (IS max=1.80 < c=10) → advantage 计算逐 token 应该一致，但 eval 落 50.83 / 55.21 / 57.92 / 59.17 四点。**4B+8B AIME single-seed eval 真实 noise band ≈ ±4pt**，比之前以为的 ±2-3 大。
3. **R5 IS ratio 精确测量**（29 dump rollouts × 全训练）：median=1.0000, max=1.80, frac(IS>c=10)=**0.0000%**, mean(soft_w)=**1.000000**。soft mask 在我们 setup 下数学上严格等于 R1。
4. **OPD ≠ multi-step PPO 是结构性冲突**：reward (reverse_kl) 跟 student weight 耦合，违反 PPO unbiased assumption。dualclip-c=10 是 FIPO 在 multi-step PPO 下的合理阈值，single-step OPD 永不触发。
5. **KL dump phenomenology** (worklog Day 40-41 第一篇 finding)：spike clustering 真实存在 (lift(d=1)=1.5, lift(d=8)=1.22, lift(d=32)=1.14)，spike amplitude 接近独立 (ρ\|x\|≈0.08)。K-window catch rate K=8→81% / K=16→92%。修正 5/29 "K=8 sweet spot" — 是 plot artifact，SNR 单调上升。
6. **训练健康度三个 concern 都有数据**：(a) R1/R3 前 25 步 instant_kl 11x 暴降；(b) R1/R3b/R4 各有 1-2 个 grad_norm spike (~10x median，PPO clip 接住)；(c) R1 末段 kl_ref ≈ 0.092 vs B 的 0.07，student 离 SFT init 远。
7. **Phase 3 candidate**：Hybrid-K OPD with Rao-Blackwellized future term — j=0 sampled (PG 必须)，j ≥ 1 用 distribution-level $d(h_{t+j})$ 替代 sampled $c_{t+j}$。文献空白点 (sampled, RB) 和 (RB, RB) 组合。条件触发：等 R1 second seed 验证。

---

## 1. Phase 2.5 完整 eval 表 (n=16)

| run | 配置 | iter99 | iter199 | **iter299 AIME-24** | iter299 AIME-25 | pass@any-24 |
|---|---|---:|---:|---:|---:|---:|
| baseline (SFT v2-700) | — | — | — | 48.75 | 40.62 | 76.67 |
| opd-4b-A (instant K=1) | — | — | — | 54.37 | 45.21 | 76.67 |
| **opd-4b-B (instant K=1)** | paper recipe | 53.54 | — | **55.83** | 45.00 | 80.00 |
| **R4** (mean K=4 no mask) | kl=1.0 | 54.17 | 51.04 | 55.00 | 43.96 | 76.67 |
| **R3b** (mean K=8 + hard mask) | kl=1.0 | 54.58 | 50.62 | **50.83** ↓ | 45.00 | 73.33 |
| **R5** (mean K=8 + soft mask) | kl=1.0 | 52.71 | 53.75 | **55.21** | 43.33 | 80.00 |
| **R3** (sum K=8 + hard mask) | kl=0.125 | 50.83 | **57.92** | **57.92** | 45.62 | 80.00 |
| **R1** (mean K=8 no mask) | kl=1.0 | 53.12 | 57.50 | **59.17** ⭐ | **48.33** ⭐ | **83.33** ⭐ |

R1 winning: +3.34pt over instant baseline B-iter299, pass@any 同时涨 → diversity 没退。

eval files: `kl_analysis/phase25/eval/`，summary 脚本 `scripts/extract_eval_summary.py`。

---

## 2. 关键 anomaly：3 个数学等价 run 实测 8pt spread

R1 / R5 / R3b 的 advantage 计算在 IS ≤ c=10 条件下逐 token 数值一致：
- R1: `masked_kl = reverse_kl` (else 分支)
- R5: `soft_w = clamp(c/clamp(IS,1), 1) = 1` 全 → `masked_kl = reverse_kl × 1`
- R3b: `keep_mask = (IS<=c).cast = 1` 全 → `masked_kl = reverse_kl × 1`

但实测 iter299 AIME-24：R1=59.17, R5=55.21, R3b=50.83 — **8.34pt spread**。

差异变量（无法分离）：
- 平台不同（R1 on p5-2, R5/R3b on p5-3）
- patched vs unpatched loss.py（多了 dump 路径、agg/mask 分支节点）
- 训练时间不同（容器 / Ray cluster 状态）
- random seed 不同
- autograd graph 节点差异（hard cast vs soft clamp）

→ 任何一个都可能贡献几 pt eval 漂移。**R1 +3.34pt finding 必须 multi-seed 验证才能 paper-grade**。

---

## 3. R5 IS ratio 完整诊断 (29 dump rollouts × 全训练)

| 量 | 全训练 aggregate |
|---|---:|
| Median IS | 1.0000 |
| Mean IS | 1.0000 |
| p99 IS | 1.12 |
| Max IS over 整训练 | **1.80** |
| frac(IS > 1.5) | 0.0000% |
| frac(IS > 2) | 0.0000% |
| frac(IS > c=10) | 0.0000% |
| Mean soft_w | **1.000000** (7-dec) |
| Worst soft_w_min ever | 1.0000 |
| Mean signal_loss = 1−mean(soft_w) | 0.000000 |

数据 / 图：
- `kl_analysis/phase25/r5_is_ratio_summary.json`
- `kl_analysis/phase25/r5_is_ratio_diagnostic.png`

R5 dump 有 `rollout_log_probs` 字段 (patch_p53_softmask_logging.py 装上)，所以能离线算 IS。R3b/R3 dump 没存这个，无法重做对比。

---

## 4. OPD ≠ multi-step PPO（结构性冲突）

**为什么 slime / TML / MiniLLM 都是 single-step PPO**：

| 冲突 | 说明 |
|---|---|
| 数学不 well-defined | OPD reward = reverse_kl[t] = log π_S(y_t) − log π_T(y_t) **显式依赖 student weights**。multi-step IS correction 假设 reward 跟 θ 独立，这个 invariant OPD 不满足。 |
| Reverse KL mode-seeking 需 fresh rollout | KL(S\|\|T) = E_{y~S}[...]。多步 update 后 S 变了，旧 sample 不再 from current S。MiniLLM §2.2 reward hacking 段在讨论这个。 |
| Teacher forward 不能省 | OPD 主成本是 teacher logp（teacher ≥ student）。multi-step PPO 设计目标是省 rollout，但 OPD 必须每次 update 重算 teacher logp（理由 1）→ 不省反增。 |
| PPO clip 跟 mode-seeking 对冲 | clip 限制 token-level 概率剧烈改变；OPD mode-seeking 正是要把概率集中。multi-step 真触发 clip 时压抑 OPD signal。 |
| 工程惯性 | MiniLLM/slime/VeRL 把 OPD 当 RLVR 换 reward 的特例，复用 PPO trainer 但 single-step。"PPO" 在 OPD context 里基本只是个外壳。 |

**含义**：dualclip-c=10 阈值是 FIPO 在 multi-step PPO setting 下的合理选择（IS 真能跑到 5-50）。我们 single-step 下 IS=1+ε（fp16 SGLang vs bf16 Megatron 的浮点残差，**不是真实 policy drift**）。c=10 是 dead code。

---

## 5. KL dump phenomenology (Day 40-41 主要 finding)

### Spike clustering 真实存在

| 量 | 实测 (R3b/R5/R4 平均) |
|---|---:|
| frac(reverse_kl exact 0) | 16-23% |
| frac(\|x\| > 0.05) (active) | 31-34% |
| spike spacing 中位数 | **2 token** |
| mean(\|x\|) | 0.16-0.22 (随训练 -20%) |

### Lift = 真正的 spike clustering 指标

ρ\|x\| (Pearson autocorr) 受 0-0 baseline 稀释，看似 ≈ 0.08 但不反映 spike 时间结构。**Lift** 是 conditional spike probability：

`lift(d) = P(active[t+d]=1 | active[t]=1) / P(active=1)`

| run | lift(1) | lift(4) | lift(8) | lift(16) | lift(32) |
|---|---:|---:|---:|---:|---:|
| B | 1.45 | 1.24 | 1.17 | 1.15 | 1.13 |
| R4 | 1.52 | 1.29 | 1.23 | 1.18 | 1.15 |
| R3b | 1.53 | 1.30 | 1.23 | 1.18 | 1.14 |
| R5 | 1.53 | 1.29 | 1.23 | 1.20 | 1.15 |

→ active token 在 reasoning chunk 内 cluster，d=1 比 random 高 50%，d=8 仍 22%。**慢衰减无拐点**。

### K-window catch rate

```
P(K-window contains ≥1 active token):
K=1   K=2   K=4   K=8   K=16  K=32  K=64
0.32  0.48  0.65  0.81  0.92  0.97  0.99
```

K=8 让 81% token 拿到非零 advantage，K=1 只 32%。**mean-K 提供 update density**。

### 修正 5/29 "K=8 sweet spot"

5/29 SNR 定义不同（K-step KL 跟 instant KL 的相关性）。重做 mean/std SNR 在所有 5 个 run 一致单调上升，K=32/K=1 ≈ √32 = 5.66，**没 sweet spot**。K=8 那次报告是 plot artifact + 不同 metric 定义。

### Stationarity

训练 early/mid/late 三阶段：lift / autocorr / K-catch rate **基本不变**，只 mean(\|x\|) 缓慢下降（reverse_kl amplitude 收敛）。temporal structure 是 stationary 的。

数据 / 图：
- `kl_analysis/phase25/kstep_lift.png` (lift curves，最关键)
- `kl_analysis/phase25/kstep_window_coverage.png`
- `kl_analysis/phase25/kstep_v2_summary.json`
- `scripts/analyze_kstep_lift.py` / `analyze_kstep_v2.py`

---

## 6. 训练健康度三个 concern 的数据

### Concern 1: R1/R3 前期 reverse_kl 暴降

| run | rollouts 0-9 | rollouts 20-29 | 暴降倍数 |
|---|---:|---:|---:|
| R1 | 0.137 | 0.024 | **5.7×** |
| R3 | 0.136 | ~0.025 | 5.4× |
| B/R3b/R4 | 0.13 | 0.11-0.13 | <1.2× |

R1/R3 在 30 步内把 instant_kl 推下 5×（最后到 0.003 / 0.0001），其它 run 几乎不动。

### Concern 2: grad_norm spike

| run | median grad | spike rollout | spike grad |
|---|---:|---:|---:|
| B | 0.394 | — | 无 (0 spikes) |
| R1 | 0.072 | 186, 201 | **11.59 / 4.35** |
| R3 | 0.071 | 217 | 0.41 |
| R3b | 0.072 | 213 | **10.15** |
| R4 | 0.121 | 20 | **11.15** |

每个 cumulative run 都有 1-2 个 ~150x median 的孤立 spike，PPO clip 接住没崩，但每 spike 是一次 stale gradient direction。

### Concern 3: low instant_kl + high kl_ref

R1 末段 instant_kl=0.003, kl_ref=0.092. **不是矛盾**：student 在 sampled token 上贴 teacher (mode-seeking) + 整体分布离 SFT init 远 (重新分布)。这是 reverse KL OPD 的设计目标 (MiniLLM §2.1)。

diversity 实测 OK：pass@any-24 = 83% > baseline 76% → 不是 collapse。

---

## 7. 文献立场综合 (4 篇)

| 文献 | 立场 | 跟 R1 +3.3pt 关系 |
|---|---|---|
| **MiniLLM** (2306.08543) | future $R_t$ 有用但需 single-step + length-norm + teacher-mix；naive $R_t$ 会爆 | **不冲突** — R1 mean K (≈ length-norm 近似)，正好在他们安全区 |
| **TML blog** (2026, thinkingmachines.ai) | discount > 0 没看见 improve（脚注，无量化数据）；他们 SFT 400K prompts → 60% AIME，gap 13pt | weakly conflict — 我们 SFT 79K prompts → 48.8%，gap 24pt，可能不同 SFT 饱和度有不同 OPD recipe |
| **Revisiting OPD** (2603.25562) | fixed-γ return-to-go variance O(T⁴)，γ=1 toy drift | weakly — 他们用 raw sum 我们用 ÷K |
| **Rethinking OPD** (2604.13016) | 不讨论 future coupling；focus on thinking pattern compatibility + new knowledge condition + token overlap dynamics | **正交** — 我们 setup 满足两个 success condition |

**FIPO** (2603.19835): RLVR 侧的 future-KL，乘性 reweight $\tilde{A}_t = \hat{A}_t \cdot f_t$ where $f_t = \text{clip}(\exp(\text{FutureKL}_t), 1-\epsilon_l, 1+\epsilon_h)$。需要 nonzero base advantage，纯 OPD reward=0 时失效。

---

## 8. ChatGPT 讨论 → Hybrid-K OPD candidate (Phase 3)

跟 ChatGPT 讨论的 idea: **Low-variance K-step OPD with Rao-Blackwellized future**。核心 insight：

### 数学基础

joint reverse KL chain rule decomposition：
```
KL(S_{1:T}|h || T_{1:T}|h) = sum_{i=1}^T E_{y_<i ~ S} [d(h_i)]
where d(h_i) = KL(S(·|h_i) || T(·|h_i))
```

两种 unbiased estimator：
- Sampled $c_t = \log \pi_S(y_t|h_t) - \log \pi_T(y_t|h_t)$ (current slime)
- Distribution $d(h_t) = \sum_v \pi_S(v) [\log \pi_S(v) - \log \pi_T(v)]$ (Rao-Blackwell)

`E_{y_t}[c_t] = d(h_t)` → distribution 是 Rao-Blackwellization of sampled，variance ≤。

### Hybrid 设计

| 项 | 选 sampled | 选 distribution | 选哪个 |
|---|---|---|---|
| j=0 (current) | 依赖 y_t，PG signal 保留 | 仅依赖 h_t，对 PG 是 baseline → 梯度=0 | **必须 sampled** |
| j ≥ 1 (future) | 高 variance | 通过 prefix path 间接依赖 y_t，RB 减 variance | **distribution 优** |

### 三种 form

**Form A** (Pure PG hybrid):
`A_t = -c_t - sum_{j=1}^{K-1} γ^j d(h_{t+j})`

**Form B** (推荐，MiniLLM Single + RB Long):
`L = sum_t d(h_t) + E_τ sum_t ratio_t · sum_{j=1}^{K-1} γ^j d(h_{t+j})`

Current 通过 direct grad（exact），future 通过 PG with RB advantage。

**Form C** (top-K approximation): 用 top-20 d_K(h) 替代 full d(h)，bandwidth 从 V=152K 降到 K=20。Rethinking OPD §4.1 实测 top-K 占 97-99% mass。

### 跟现有方法 estimator 矩阵

| 方法 | j=0 | j ≥ 1 |
|---|---|---|
| MiniLLM | exact-grad d(h_t) | sampled c |
| slime current (R1) | sampled c | sampled c |
| Rethinking OPD top-K | top-K d_K(h_t) | (no future) |
| **Form A (空白)** | sampled c | distribution d |
| **Form B (空白)** | exact-grad d(h_t) | distribution d via PG |

→ (sampled, RB) 和 (RB, RB) 是文献空白。

### Novelty 评估

- 数学 novelty low (Rao-Blackwell 是标准 trick)
- 文献空白程度 medium ((sampled, RB) 和 (RB, RB) 这两个组合都没人完整写过)
- 实证 contribution 取决于 +pt 大小：>2pt paper-grade，<1pt ablation appendix
- 跟 Phase 2.5 lift 数据 alignment 强 — spike clustering 给 RB future motivation

### MVP 实验

复用 R1 setup (4B + 8B + dapo-math + lr=2e-6 + K=8)：

| variant | j=0 | j ≥ 1 |
|---|---|---|
| A0 = R1 baseline | sampled c | sampled c |
| B1 = Form A | sampled c | top-20 d_K(h_{t+j}) |
| B2 = Form B | exact-grad d_K(h_t) | top-20 d_K(h_{t+j}) via PG |

每 variant **2 seeds**（Risk 0：8pt spread 已经 demonstrate 必须）。

判读：
- B1 vs A0：纯 RB future 效应
- B2 vs B1：MiniLLM Single 边际
- 全部 < noise band → R1 已 saturate K-step OPD signal

### Cost

SGLang `top_logprobs_num=20` 已验证 (worklog 5/11)，rollout +20-30%。

### Risk

- Risk 0 (highest)：8pt single-seed spread → 任何改进必须 ≥ 2 seeds 验证
- R1 已 saturate mean-K denoising (worklog Day 40-41)，RB 边际可能小
- top-K bias 在 OOD prefix 上未必 97-99% mass
- Form B 实现需小心 backward graph

---

## 9. dualclip mask 在 OPD 的失误诊断

我们抄 FIPO mask metric (IS ratio = `exp(student_now - student_old)`) 但累积的是 reverse_kl (student vs teacher)，**两个量在 single-step on-policy 下基本独立**：

| | FIPO | 我们 |
|---|---|---|
| 累积量 | Σ Δlogp = Σ (new − old) | Σ reverse_kl = Σ (student − teacher) |
| Mask metric | IS = exp(new − old) | IS = exp(student_now − student_old) |
| metric 与累积量同源？ | ✓ 同一个数的指数 | ✗ 完全不同 |

→ R3 / R3b / R5 都用 dualclip mask 但都 dormant，**不是 mask 调坏了，是 metric 选错了**。

**OPD-native mask metric** (paper-grade direction):
- **reverse_kl 阈值**: 直接对 OPD 信号自身的 outlier
- **Teacher confidence**: mask = 1[H(π_T) ≤ τ]，剔除 teacher 不自信位置 (Revisiting OPD Fig.4)
- **Token entropy / decision token gating**: 对接 EOPD (2603.07079)

---

## 10. 当前训练状态 / 资源

| 节点 | 状态 |
|---|---|
| p5-3 | 完整空闲（R5 跑完，SGLang zombie 已清）。所有 8 GPU 可用 |
| p5-2 | R1 ckpt 还在，未验证当前训练状态（其他实验未跑） |
| 其它 | 未查 |

---

## 11. 下一步 priority

### 高优先级

1. **R1 second seed 复现** (R1') — Phase 2.5 收尾必须，也是 Phase 3 触发条件。
   - 配置：在 p5-3 用 patched loss.py + 不传 dualclip flag + 不同 seed
   - 三 hypothesis 区分：
     - R1' ≈ 59 → R1 winning 真实
     - R1' ≈ 55 (R5 level) → R1 是 single-seed luck
     - R1' ≈ 51 (R3b level) → 所有 spread 都是 noise

2. **R3b / R5 second seed (optional)**: 看是不是 spread 缩窄到 R1' 水平

3. **写 R1 + ChatGPT hybrid 方向到 worklog Phase 3 candidate** (已写 Day 41 + Day 42)

### 中优先级

4. **OPD-native mask metric 实验**: reverse_kl 阈值 τ ∈ {0.3, 0.5, 1.0} 跑一组 single-seed sweep
5. **K=16 / K=32 sweep**: 验证 SNR/lift 单调预测；预期 K 越大越好直到 boundary truncation 主导
6. **MiniLLM Single trick 加进 R1**: paper-grade ablation

### 低优先级

7. **Top-K dump format 改进**: 加 student/teacher top-20 logp 到 dump，能算 overlap_ratio + entropy_gap (Rethinking OPD 诊断)
8. **重做 SFT (补数据量到 ~300K)**: 目标 SFT baseline 56-57%，跟 Lightning OPD 对齐

---

## 12. 文件清单 (Day 40-42)

### 脚本
| 文件 | 用途 |
|---|---|
| `scripts/extract_eval_summary.py` | 从 eval JSON 拉 avg_pass_at_1 / pass@any |
| `scripts/diversity_check.py` | mode-collapse signature 表 |
| `scripts/analyze_phase25_eval_vs_kl.py` | eval ↔ training metric 相关性 |
| `scripts/analyze_kstep_autocorr.py` | raw autocorr + SNR(K) |
| `scripts/analyze_kstep_v2.py` | sparsity-aware autocorr + K-window catch |
| `scripts/analyze_kstep_lift.py` | lift = P(active\|active)/P(active) |
| `scripts/analyze_kstep_per_phase.py` | early/mid/late 分阶段 |
| `scripts/analyze_r5_is_ratio.py` | R5 IS ratio 全分布诊断 |
| `scripts/plot_phase25_kl.py` | 6-panel vstack training trajectory |
| `scripts/r5_extract.py` | R5 ray log → CSV |

### 数据 / 图
| 文件 | 内容 |
|---|---|
| `kl_analysis/phase25/eval/aime20{24,25}_*.json` | 全 24 个 n=16 eval (R1/R3/R3b/R4/R5 + baseline + opd-A/B) |
| `kl_analysis/phase25/dump_summary.json` | per-rollout KL dump 数值 |
| `kl_analysis/phase25/kstep_v2_summary.json` | K-window catch + sparsity |
| `kl_analysis/phase25/kstep_lift_summary.json` | lift(d) 表 |
| `kl_analysis/phase25/kstep_per_phase.json` | 三阶段统计 |
| `kl_analysis/phase25/r5_is_ratio_summary.json` | R5 IS ratio 全训练分布 |
| `kl_analysis/phase25/phase25_trajectories_vstack.png` | 6-panel 训练曲线 |
| `kl_analysis/phase25/phase25_dump_diagnostic.png` | KL dump 4-panel 诊断 |
| `kl_analysis/phase25/kstep_lift.png` | lift(d) 曲线 (最关键) |
| `kl_analysis/phase25/kstep_window_coverage.png` | K-window catch rate |
| `kl_analysis/phase25/r5_is_ratio_diagnostic.png` | R5 IS / soft_w / signal_loss |

### Worklog
- Day 40-41 entry: Phase 2.5 完整 eval + KL phenomenology
- Day 41 entry (续): Phase 3 candidate Hybrid-K OPD
- Day 42 entry: R5 eval + IS dormant + OPD vs multi-step PPO

### Commits (origin/main)
- `51901f0` Phase 2.5 complete: R1 winner (59.17 AIME-24) + KL temporal analysis
- `9bd333e` Phase 3 candidate: Hybrid-K OPD with Rao-Blackwellized future term
- `50a02c7` Day 42: R5 eval complete + IS ratio dormant + OPD vs multi-step PPO

---

## 13. 关键认知修正 (跨 session)

| 之前以为 | 现在知道 |
|---|---|
| K=8 是 SNR sweet spot (5/29 finding) | 错。SNR 单调上升，K=32 比 K=8 还高 |
| reverse_kl token-level 接近独立 | 错。spike clustering 真实 (lift) ，amplitude 接近独立 |
| R3 "winner" (57.92) | 升级。R1 (59.17) 是真 winner，但 R1 数学 ≡ R5 ≡ R3b spread 8pt → single-seed |
| FIPO dual-clip mask 在我们 setup 起作用 | 错。c=10 dead code，IS 永远 < 2 |
| K-step OPD = future credit assignment | 部分对。是 update density × √K 噪声平均 + spike clustering 利用，不是简单 future credit |
| Mode collapse 由 instant_kl ≈ 0 标志 | 错。R1 instant_kl=0.003 + pass@any=83% > baseline = deeper convergence，不是 collapse |
| OPD 可以套 multi-step PPO | 错。reward 跟 weight 耦合，结构性冲突 |
| R1 +3.34pt 是 robust finding | 降级。8pt spread 表明 4B+8B single-seed noise band ≈ ±4pt，必须 multi-seed |

---

## 14. 给下一个 session 的 context

**正在跑的实验**: 无（R5 已完，p5-3 空闲）

**最重要的待跑**: R1 second seed (R1' on p5-3, patched loss.py, no dualclip flag)，约 12-24h

**Phase 2.5 主要产出**:
1. R1 mean K=8 是当前最强 4B+8B+8B-teacher OPD recipe (单 seed)
2. KL dump phenomenology 是干净的 paper-grade contribution
3. dualclip mask metric 选错了 → OPD-native mask 是 follow-up 方向
4. single-seed eval noise band ≈ ±4pt 是 hard finding (negative)

**Phase 3 candidate (设计完未起)**: Hybrid-K OPD with Rao-Blackwellized future term。条件触发于 R1' 验证。

**ChatGPT 讨论的核心 idea**:
- j=0 必须 sampled c_t（PG baseline trap）
- j ≥ 1 用 distribution d(h_{t+j})（RB unbiased + lower variance）
- Form B = MiniLLM Single + RB Long 是 (RB, RB) cell，文献空白
- top-K (K=20) 当 d_K 是 compute-feasible 实现，bandwidth +20-30%
- MTP heads 不算严格 chain-rule joint KL，作 ablation 不作主方法

**Paper story 候选**:
> "On-Policy Distillation Meets Temporal Credit Assignment: A Phenomenology and a Hybrid Estimator"
>
> §3 Phenomenology (lift / autocorr / spacing / K-window)
> §4 Theory (4-way estimator matrix, bias-variance-cost)
> §5 Method (Hybrid-K OPD = MiniLLM Single + RB Long with top-K KL)
> §6 Experiments (AIME 4B+8B with R1 baseline, 2 seeds × 3 variants)
