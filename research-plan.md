# K-Step Reward-to-Go OPD 研究计划

## 1. 项目概述

### 1.1 研究问题

在 LLM on-policy distillation (OPD) 中，当前主流做法是 token-level single-step KL reward：每个 token 的 advantage 仅由当前位置的 `r_t = log π_S(y_t|h_t) - log π_T(y_t|h_t)` 决定。这种做法 variance 低、实现简单，但存在 **myopic credit assignment** 问题——一个 early token 可能把 student 送上错误分支，但 single-step reward 无法把后续 token 的坏信号归因回来。

本项目研究：将 single-step reward 扩展为 **k-step discounted reward-to-go** `G_t^{(k,γ)} = Σ_{j=0}^{k-1} γ^j r_{t+j}`，能否在 math reasoning 等长链推理任务上带来可测量的改进？如果 fixed-k 不够，adaptive/gated lookahead 是否更好？

### 1.2 论文定位

**不要**定位为 "plain k-step OPD" 或 "does bigger k help"。

**应该**定位为：**"Adaptive Lookahead OPD under Calibrated Teacher Trust"** —— future reward only when the teacher can still be trusted。

理由：
- 文献已经知道 future coupling 是 bias-variance tradeoff 的关键 knob（Revisiting OPD）
- 纯 fixed-k sweep 可能不够发论文，除非 empirical gains 异常清晰
- 最强正面证据（KETCHUP）来自 RL-based KD 而非标准 OPD
- 最强负面证据（MiniLLM, Revisiting OPD）指出 future coupling 增加 variance
- 真正的 gap 是：**在 local signal quality 已经修复的前提下，future coupling 何时有帮助？**

### 1.3 核心假设

> **Moderate lookahead helps only once the student is already visiting prefixes where teacher guidance is reliable.**
> K-step OPD should be conditional on support overlap, teacher confidence, or rollout correctness, not applied uniformly.

**⚠️ 边界声明（2026-05-31 更新，依据 arXiv 2603.25562 Revisiting OPD）**：

Naive **uniform fixed-k / fixed-γ return-to-go 已被证否**，不能作为本项目的 contribution：
- 理论：token-level OPD worst-case variance O(T²)，sequence-level（γ=1）O(T⁴)。long-horizon reasoning 下 T 极大，naive future coupling 的 variance 爆炸是结构性的（与我们 5/29 KL dump 实测 sum variance ∝K、full 368,000× 互相印证）。
- 实验：他们的 toy study 显示 γ↑ → gradient variance↑、optimization 越不稳，γ=1 时 policy 直接 drift 不收敛。

因此本项目的立足点**必须落在他们没覆盖的那一半**：
1. **Conditional / gated lookahead** —— 只在 teacher 可信、variance 可控时才耦合 future（adaptive，不是 uniform）。
2. **先把"单步信号"做干净**（top-K local support，见 §5.2bis / P6 升级），再在干净信号上做选择性 future coupling。

> uniform fixed-k/fixed-γ 在本项目里只作为 **复现性 baseline + variance 现象复现**，不是主结果。任何"主结果 = fixed-k sweep 赢了 k=1"的写法都会被 reviewer 判为重复 2603.25562。

### 1.4 Go/No-Go 判据

**⚠️ 基线对手已升级（2026-05-31）**：不再是原始 sampled-token k=1，而是 **k=1 + teacher top-K local support matching**（2603.25562 的 fix，Qwen2.5-7B 上 36.4→41.5）。理由：拿原始 sampled-token 当对照，赢了也不算数——reviewer 会说没比真正的 SOTA 单步。bar 必须抬到他们的水平。

在 matched compute 下，如果 **conditional/adaptive lookahead** 不能比 **"做好的单步"（k=1 + top-K local support）** 提升 **1-2 个绝对点** on a long-horizon reasoning benchmark，同时 variance、response length、repetition 可控：
- → Pivot 到更强的 gating signal（teacher confidence / support overlap / rollout correctness）
- → 或者把 contribution 重心移到 "top-K local support + 我们的诊断/分析"，lookahead 作为 negative/ablation result
- → 如果 adaptive variants 也全面打不过做好的单步，写成 negative result paper（"future coupling 在干净单步信号下也不值得" —— 这本身是对 2603.25562 的有价值补强）

> 注意：uniform fixed-k sweep（Phase 2）赢 naive k=1 **不构成 Go**，那只是复现已知现象。Go 的门槛是 adaptive 赢"做好的单步"。

### 1.5 代码基础

**主 repo**：THUDM/slime（SGLang rollout + Megatron training）
- 官方已有 OPD example：`examples/on_policy_distillation/run-qwen3-8B-opd.sh`
- 改动切口：`slime/backends/megatron_utils/loss.py` → `apply_opd_kl_to_advantages()`
- 当前实现：`reverse_kl = student_log_probs[i] - teacher_log_probs[i]`，直接作为 single-step penalty
- K-step 改动：对 `reverse_kl` 做 truncated discounted cumsum，几乎不增加计算开销

**备选 repo**（第二阶段 strong baseline）：
- hhh675597/revisiting_opd：verl-based，有 top-K local support matching，OPD taxonomy 最清晰
- THUNLP/OPD：有 sampled-token / top-K / weighting / eval pipeline，资源需求较重

---

## 2. Phase 0: 环境搭建 & Pipeline Smoke Test

**时间**：2-3 天
**目标**：确认 Slime + SGLang OPD pipeline 端到端能跑通，不 crash，数值合理

### 2.1 环境搭建

1. Clone THUDM/slime，按 README 安装依赖
2. 确认 SGLang 版本兼容性（slime 官方 example 指定的版本）
3. 确认 Megatron backend 能正常初始化
4. GPU 资源：smoke test 阶段 2-4 张 A100/H100 即可

### 2.2 模型准备

| 角色 | 模型 | 理由 |
|------|------|------|
| Student | Qwen2.5-3B-Instruct 或 Qwen3-4B | 小模型，快速迭代 |
| Teacher | Qwen3-8B 或 Qwen2.5-7B-Instruct | Same tokenizer family，避免 tokenizer mismatch artifact |

**关键约束**：teacher 和 student 必须 same tokenizer + same special tokens。这是 OPD 的硬性要求，Revisiting OPD 明确指出 cross-tokenizer 会直接扭曲 supervision signal。

### 2.3 数据准备

- 训练集：从 dapo-math-17k 中抽 1k-3k prompts
- 评估集：MATH-500 子集
- 转成 slime 要求的数据格式

### 2.4 Smoke Test 运行

启动 teacher SGLang server：
```bash
CUDA_VISIBLE_DEVICES=X python3 -m sglang.launch_server \
  --model-path /path/to/teacher \
  --host 0.0.0.0 --port $TEACHER_PORT \
  --tp 1 --chunked-prefill-size 4096 \
  --mem-fraction-static 0.6
```

修改官方脚本，缩小配置：
```
--n-samples-per-prompt 2
--rollout-max-response-len 2048
--global-batch-size 16
--max-steps 20  # 只跑几步确认不 crash
```

### 2.5 Smoke Test 验证清单

- [ ] SGLang teacher server 正常启动并响应
- [ ] Student rollout 正常生成
- [ ] `teacher_log_probs` shape 和 `student_log_probs` 对齐
- [ ] `reverse_kl = student_log_probs - teacher_log_probs` 数值合理（非 NaN/Inf，量级 [-10, 10]）
- [ ] Loss 在几步内有变化（非常数）
- [ ] Checkpoint 正常保存和加载
- [ ] Eval script 能正常运行

全部通过才进 Phase 1。

---

## 3. Phase 1: Single-Step OPD Baseline 复现

**时间**：5-7 天
**目标**：拿到一个可靠的、有完整 diagnostics 的 k=1 baseline，同时建立 diagnostics pipeline

### 3.1 训练配置

**Near-pure OPD 设置**（让 OPD KL penalty 成为主要学习信号）：
```
--advantage-estimator grpo
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
--use-kl-loss
--kl-loss-coef 0.00    # 不加额外 KL loss
--entropy-coef 0.00    # 不加 entropy bonus
```

Reward function 返回 0（pure distillation mode）。这样学习信号主要来自 OPD KL penalty。

**Caveat**：Slime 的 OPD 实现是 `base RL advantage + OPD KL penalty`，不是完全纯粹的 `A_t = teacher_logp - student_logp`。但设 reward=0 + kl_loss_coef=0 后，GRPO advantage 基本为 0（因为所有 rollout reward 相同），学习信号主要来自 OPD KL penalty。需要在实验中验证这一点。

### 3.2 训练规模

**Pilot 规模**（先跑通）：
- Student: Qwen2.5-3B-Instruct
- Teacher: Qwen3-8B
- Data: 3k-5k math prompts
- `n_samples_per_prompt`: 4
- `max_response_len`: 2048-4096
- 训练 200-500 steps
- 2-3 random seeds

**如果 pilot 成功，scale up**：
- Student: Qwen2.5-7B-Instruct
- Teacher: Qwen3-14B 或 Qwen3-32B
- Data: 10k-17k prompts
- `n_samples_per_prompt`: 4-8
- `max_response_len`: 4096-8192

### 3.3 Diagnostics Pipeline（核心基础设施）

这是整个项目最重要的基础设施。每 N 步记录以下指标，用 wandb 或 tensorboard：

**A. 任务质量指标**：
1. Eval pass@1 on MATH-500
2. Eval pass@k (k=4 or 8)
3. Avg@k

**B. 分布匹配指标**：
4. Mean reverse_kl per token position（画 position-wise 曲线）
5. `teacher_logprob - student_logprob` 分布直方图
6. 按 rollout 正确/错误分组的 token reward 分布

**C. 稳定性指标**：
7. Average response length（训练过程趋势）
8. Truncation rate（被 max_len 截断的比例）
9. Repetition rate（n-gram 重复检测，例如 4-gram 重复率）
10. Advantage variance（per batch）

**D. 信用分配指标**（为 Phase 2 做准备）：
11. `r_t` 的 temporal autocorrelation：`corr(r_t, r_{t+h})` for h = 1, 2, 4, 8, 16
12. Early-token reward 与 final correctness 的 correlation
13. Teacher perplexity along trajectory（position-wise）
14. Reward variance by position

**E. Failure-mode 诊断**（2026-05-31 新增，复用 2603.25562 已验证的诊断，为 adaptive gating 提供信号来源）：
15. **Teacher-student log-prob gap vs position 分布**（他们 Fig 4）：验证 "后段 gap 变宽、更 noisy" → gating 时后段 token 应降低 future coupling 权重。**这是我们 5/29 想做但因没存 token_ids 没做成的分析。**
16. **Sampled-token reward 正负比例**（他们 Fig 2）：量化 "one-token signal imbalance"（大多数 token 负 reward，正信号集中在少数 token）
17. **Repetition-loop / 退化 prefix 上 teacher 是否仍给高概率**（他们 Fig 3）：验证 "teacher 不惩罚坏行为" → 这类 prefix 是 future coupling 最危险的地方，gating 必须能识别并切断

> ⚠️ **KL dump 必须修复（5/29 踩坑）**：下次所有 OPD dump **务必存 token_ids + position**，否则上述 E 类诊断和尾部 KL 分析全做不了。5/29 的 dump 只存了 logprob、token_ids 是空 list，导致无法做 position-wise / special-token 分析。

### 3.4 关键分析

**Autocorrelation 分析**是 Phase 1 最重要的产出之一：
- 如果 `corr(r_t, r_{t+h})` 在 h > 1 时接近 0 → future tokens 几乎没有额外信息，k-step 可能不会帮助（提前预警）
- 如果存在显著正 autocorrelation → k-step 有信息增益的理论基础
- 如果 autocorrelation 在 correct rollouts 和 incorrect rollouts 之间有显著差异 → adaptive gating 有依据

### 3.5 实验命名规范

```
{method}__k{k}__gamma{gamma}__seed{seed}__student-{model}__teacher-{model}
```
例如：`opd__k1__gamma1.0__seed42__qwen2.5-3b__qwen3-8b`

### 3.6 Phase 1 交付物

- [ ] 稳定的 k=1 OPD baseline，2-3 seeds 的 mean ± std
- [ ] 完整的 diagnostics dashboard
- [ ] `r_t` autocorrelation 分析报告
- [ ] Baseline accuracy 数字（用于后续 Phase 的对比基准）


---

## 4. Phase 2: K-Step Return 实现 & Fixed-K Sweep

**时间**：5-7 天
**目标**：实现 k-step reward-to-go，做 fixed-k sweep，回答 "bigger k helps?" 这个基础问题

### 4.1 核心代码改动

改动位置：`slime/backends/megatron_utils/loss.py` → `apply_opd_kl_to_advantages()`

**当前代码**（single-step）：
```python
reverse_kl = student_log_probs[i] - teacher_log_probs[i]
advantages[i] = adv - args.opd_kl_coef * reverse_kl
```

**改为**（k-step）：
```python
reverse_kl = student_log_probs[i] - teacher_log_probs[i]

# k-step discounted reward-to-go
k_step_kl = truncated_discounted_cumsum(
    reverse_kl,
    k=args.opd_lookahead_k,
    gamma=args.opd_gamma,
    mask=response_mask[i],
)

advantages[i] = adv - args.opd_kl_coef * k_step_kl
```

**`truncated_discounted_cumsum` 实现**：
```python
def truncated_discounted_cumsum(rewards, k, gamma, mask):
    """
    对每个位置 t，计算 G_t = sum_{j=0}^{min(k,T-t)-1} gamma^j * r_{t+j}
    使用 vectorized reverse scan，O(B*T) 复杂度，不需要额外 model forward。
    """
    B, T = rewards.shape  # 或者 rewards 是 1D per sample
    returns = torch.zeros_like(rewards)

    # 高效实现：reverse scan with sliding window
    # 方法1：简单循环（correctness first）
    for t in range(T - 1, -1, -1):
        horizon = min(k, T - t)
        g = 0.0
        for j in range(horizon - 1, -1, -1):
            if mask[t + j]:
                g = rewards[t + j] + gamma * g
            else:
                g = 0.0  # mask 外的 token 不参与
        returns[t] = g

    return returns

    # 方法2：vectorized（性能优化，Phase 2 后期）
    # 用 torch.cumsum + 减去超出 window 的 tail
```

### 4.2 新增命令行参数

```
--opd-lookahead-k     # default=1，k=1 时退化为原始 single-step OPD
--opd-gamma           # default=1.0，discount factor
```

### 4.3 回归验证（必须先做）

在做任何 k>1 实验之前，必须确认：
- `--opd-lookahead-k 1 --opd-gamma 1.0` 的结果和 Phase 1 baseline **完全一致**（数值级别）
- 用同一个 seed，跑 50 steps，对比 loss curve 和 advantage 数值
- 任何不一致都说明实现有 bug

### 4.4 Fixed-K Sweep 实验矩阵

| 实验 | k | γ | 其他配置 | 目的 |
|------|---|---|----------|------|
| baseline | 1 | 1.0 | 同 Phase 1 | 对照组 |
| k2-g1.0 | 2 | 1.0 | 同上 | 最小 lookahead |
| k4-g1.0 | 4 | 1.0 | 同上 | moderate lookahead |
| k8-g1.0 | 8 | 1.0 | 同上 | larger lookahead |
| k16-g1.0 | 16 | 1.0 | 同上 | stress test |
| k4-g0.95 | 4 | 0.95 | 同上 | discount 的效果 |
| k8-g0.95 | 8 | 0.95 | 同上 | discount + larger k |
| k4-g0.8 | 4 | 0.8 | 同上 | stronger discount |
| full-g1.0 | T | 1.0 | 同上 | full reward-to-go（upper bound on variance） |

每个配置跑 2-3 seeds。

### 4.5 必须记录的指标

Phase 1 的所有 diagnostics，加上：
- **Advantage variance vs k**：这是最关键的图。如果 variance 随 k 指数增长，说明 future signal 太 noisy
- **Accuracy vs k curve**：是否存在 sweet spot？
- **Length inflation vs k**：k 越大是否导致 response 越长？
- **Truncation rate vs k**
- **k-step return 的 position-wise 分布**：early positions 的 return 是否比 late positions 方差大得多？

### 4.6 Phase 2 交付物

- [ ] K-step OPD 实现，通过回归验证
- [ ] Fixed-k sweep 完整结果表：

| Method | k | γ | Eval acc | Avg length | Truncation | Repetition | Adv variance |
|--------|---|---|----------|------------|------------|------------|--------------|
| SFT/init | - | - | | | | | |
| single-step OPD | 1 | - | | | | | |
| k-step OPD | 2 | 1.0 | | | | | |
| k-step OPD | 4 | 1.0 | | | | | |
| k-step OPD | 8 | 1.0 | | | | | |
| k-step OPD | 4 | 0.95 | | | | | |

- [ ] Accuracy vs k 曲线图
- [ ] Advantage variance vs k 曲线图
- [ ] Go/No-Go 决策

### 4.7 Go/No-Go 决策点

**Go（进 Phase 3）**：某个 k > 1 在 matched compute 下比 k=1 提升 ≥ 1 绝对点，且 variance/length/repetition 可控

**Conditional Go（进 Phase 3 但换方向）**：k > 1 没有清晰提升，但 diagnostics 显示：
- Autocorrelation 在 correct rollouts 上显著高于 incorrect rollouts → adaptive gating 有希望
- Variance 是主要问题 → min-form / clipped aggregation 有希望
- 某些 position range 有提升但被其他 position 的 noise 抵消 → position-dependent k 有希望

**No-Go（pivot）**：k > 1 全面不如 k=1，diagnostics 也没有 actionable signal → 写 negative result，或把重心移到 **P0 top-K local support**（已升级为 main-method 基石，见 §5.1/§5.2）+ 我们的诊断分析

> ⚠️ 注意：Phase 2 的 fixed-k sweep 赢 naive k=1 **不等于项目成功**。它只是复现 2603.25562 已知的 bias-variance 现象、验证我们的实现和 diagnostics。真正的 Go/No-Go 在 Phase 3：adaptive lookahead 能否赢"做好的单步"（P0）。

---

## 4.8 Phase 2.5: Sum vs Mean Aggregation 决战（核心实验，2026-05-31 定）

**这是 K-step 这条线能不能立得住的分水岭实验。** 目标：在 4B real OPD 上，干净地分离 "credit assignment（reward-to-go）" 和 "magnitude 稀释/去噪" 两个因素，确定我们的增益（如果有）到底来自哪个。

### 背景：为什么是 sum vs mean

slime 现在的 cumulative v2 是 **mean-K**（truncated horizon 分支 `÷actual_k`）。问题（见 worklog 5/29 + 2603.25562 分析）：
- **mean-K 的 return 上界 = B_r**，和 instant 同阶 → variance 低（实测 = instant 的 0.15×），但未来信号 magnitude 被 ÷K 稀释，本质偏向"局部去噪"
- **sum-K 才是教科书 reward-to-go**（把未来后果 full-magnitude 归因回当前 token），但 variance 随 K 快速增长（5/29 实测 sum variance ∝K、略超线性，K=8 时 9.9×、full 时 368,000×；worst-case bound 是 O(K²)/O(T⁴)），不控制几乎必爆
- mean 和 sum **携带同一组未来信号，唯一差别是 ÷K**。同 K 下对比 sum vs mean，差异干净地归因到"magnitude 是否稀释"，不需要 EMA 之类的额外对照

### 三个已定的设计决策

1. **sum 的 kl_coef = (instant kl_coef)/K 起点**：让 sum-K 和 mean-K 的 penalty 量级对齐，差异才归因到"分配结构"而非"量级"。instant kl_coef=1.0 → sum-K=8 用 0.125。
2. **K 主测 8，顺带 4**：5/29 KL dump 已证 K=8 是 structural sweet spot（60%+ 独立信号、variance 缩 6.5×），不铺满 2/4/16 省算力。
3. **dual-clip mask 默认只开在 sum 上**（mean variance 已低不需要）；但 **mean+mask（R3b）不是可有可无的 sanity，而是核心对照**——见下方 2×2 析因。

### 需要的代码改动（loss.py，3 处）

| 改动 | 现状 | 要加 |
|---|---|---|
| `--opd-agg {sum,mean}` | truncated 分支硬编码 `÷actual_k` | sum 时不除 K |
| `--opd-dualclip-c` | 无 | 用 `rollout_log_probs`（已确认在 rollout_data 里）算 IS ratio `exp(student_logp - rollout_logp)`，超阈值 c 的 token 从 cumsum 剔除（mask=0），参考 FIPO Eq.5 |
| kl_coef 缩放约定 | 手动 | 文档化 sum 用 1/K，回归测试 `--opd-agg mean` 与现 v2 数值完全一致 |

> ⚠️ 回归测试是硬门槛：改完 `--opd-agg mean` 必须和现 cumulative v2 逐数值一致，否则说明引入 bug。

### 实验矩阵 Batch 1（主轴，每 run 300 rollouts）

复用 opd-4b-B setup（v2-700 student, Qwen3-8B teacher, lr=2e-6, T=0.8, max_resp=4096；5/29 证 iter_299 饱和，300 rollouts 足够）。**Batch 1 全部固定 gamma=1.0**（隔离 agg 效应，discount 留给 Batch 2）。

| run | agg | K | kl_coef | mask | 测什么 |
|---|---|---|---|---|---|
| R0 | instant | 1 | 1.0 | — | baseline（=opd-4b-B iter_299，已有，不重跑） |
| R1 | mean | 8 | 1.0 | off | slime v2 现状（去噪派） |
| R2 | sum | 8 | 0.125 | off | 裸 reward-to-go（预期 variance 偏高，可能发散） |
| **R3** | **sum** | **8** | **0.125** | **c=10** | **sum + dual-clip mask（main candidate）** |
| **R3b** | **mean** | **8** | **1.0** | **c=10** | **mean + mask（2×2 析因的关键角，不是 sanity）** |
| R4 | mean | 4 | 1.0 | off | K 敏感性 |
| R5 | sum | 4 | 0.25 | c=10 | K 敏感性 |
| R6 | sum | 8 | 0.125 | c=5 | mask 强度敏感性 |

共 **7 个新 run**（R0 已有）。

### 干净的 2×2 析因（核心）

R1/R2/R3/R3b 构成 {sum,mean}×{mask on,off} 的 2×2，每个 cell 只差一个因子，避免混淆变量：

| | mask off | mask on |
|---|---|---|
| **mean** | R1 | R3b |
| **sum** | R2 | R3 |

- **R3 vs R3b**（sum vs mean，都开 mask）= **干净的 credit-assignment 测试**。sum 赢 → full-magnitude reward-to-go 比稀释版强，**story 成立**。
- **R3 vs R2**（mask on vs off，都 sum）= mask 对 sum 的贡献（救 variance）。
- **R1 vs R2**（mean vs sum，都不 mask）= ÷K 在无 mask 时的影响（但 R2 可能发散，发散本身即"sum 必须配 mask"的证据）。
- **R1 vs R3b**（mask on vs off，都 mean）= mask 对 mean 的无害性。

### Seeds 策略（必须，否则重蹈 n=1 覆辙）

worklog 第一大教训是 noise（n=1 eval 把 ±5pt 信号淹没）。sum vs mean 预期效应可能仅 1-3pt，**单 seed 区分不出**。因此：
- **R1/R3/R3b（2×2 主对照）跑 2 seeds**，报 mean±std
- R2/R4/R5/R6（敏感性）单 seed 先看趋势，有信号再补 seed
- 判定门槛：效应必须 **clear noise band**（n=16 eval ±2-3pt + seed std）才算数，否则记为"无显著差异"

### 判读逻辑（决定 paper story）

- **R3 vs R3b**：sum+mask 赢 mean+mask（控制 mask 不变）→ credit assignment 比去噪强，**reward-to-go story 成立**（这是主判据，替代之前混淆的 R3 vs R1）
- **R3 vs R2**：mask 是否真救了 sum 的 variance（看 grad_norm / adv variance / 是否发散）
- **R1 vs R0**：mean-K 相比 instant 有没有用（纯去噪有没有价值）
- 全部 n=16 eval AIME24/25（用现成 `eval-aime-n16.sh`），eval iter_99/199/299 画曲线（5/29 证 iter_99 已吃掉 ~80% gain）

### ⚠️ kl_coef 敏感性风险

sum 的 `kl_coef=1/K` 只是 magnitude 对齐的**起点**，不是调好的值。kl_coef 是 OPD 最敏感的旋钮——若起点偏，R2/R3 可能因"系数错"而非"方法错"失败，造成假阴性。缓解：
- main candidate（R3）若首跑不 work，先做 mini coef-sweep（0.0625 / 0.125 / 0.25）再下结论，不要一次失败就判 sum 死刑
- 监控 opd penalty 量级：R3 的 `opd_kl_coef * cumulative_kl` 应和 R0 的 `1.0 * reverse_kl` 同量级，不同则系数没对齐

### Batch 2（条件触发，只在 Batch 1 有正信号时）

- 若 R3 最好 → 加 discount γ sweep（R3 + γ=0.95/0.99，FIPO soft-decay 味道）+ K=16 stress test
- 若 sum 全败、mean 也 ≈ instant → 记为 **negative result**，重心转 P0 top-K local support（§5.2）
- 若 mean 赢但 sum 不赢 → story 诚实改成 "local denoising helps"，不硬套 reward-to-go

### 必记诊断（每 run）

1. **adv variance**（sum vs mean 核心区别应体现在这）
2. **grad_norm 轨迹**（sum 不 mask 预期 spike；R2 若早期发散可早停省算力）
3. **dual-clip mask 触发率**（多少 token 被剔除）
4. **KL dump 存 token_ids + position**（修 5/29 踩坑，为 position 分析）

### 资源

- 每 run 300 rollouts × ~100s/step ≈ 8-9h（单机 8×H100，复用 opd-4b-B 配置）
- Batch 1 = 7 个新 run，其中 R1/R3/R3b 各 2 seeds → 共 10 个训练 run。多机并行 ~2 天 / 单机串行 ~4 天
- teacher：复用 qzf-dev 远程 Qwen3-8B（DP=4）或本地 TP=2
- teacher：复用 qzf-dev 远程 Qwen3-8B（DP=4）或本地 TP=2

---

## 5. Phase 3: Advanced Variants

**时间**：2-3 周
**目标**：实现和测试 adaptive/gated variants，找到最佳方案

### 5.1 Variant 优先级排序

**⚠️ 2026-05-31 重大调整（依据 2603.25562 + FIPO 两篇夹出的结论）**：top-K local support 从最低 P6 升到 **P0**。理由：两篇文献夹出的结论是"单纯加 future 不行，但 future 在信号干净/可信时可能有救"。top-K local support 正是把单步信号变干净的最强手段（2603.25562 实测 36.4→41.5），且和 K-step **正交**。本项目的 main method 候选从"纯 K-step"挪到 **"top-K local support（修单步）+ adaptive/gated lookahead（选择性耦合 future）"的组合**——这是文献空白，比单纯 K-step 更难被 scoop、故事更完整。

| 优先级 | Variant | 核心思路 | 实现复杂度 |
|--------|---------|----------|-----------|
| **P0** | **Top-K local support matching** | **truncated reverse-KL over teacher top-K support + renorm + top-p rollout + special-token mask（2603.25562）。先把单步做干净** | **高（需 teacher top-K logits，slime rollout 加 `top_logprobs_num`，5/11 已验证 sglang 支持 K=50）** |
| **P0+** | **组合 main method：top-K local support + adaptive lookahead** | **在干净的 top-K 单步信号上，做 conditional/gated future coupling（gating 信号来自 §3.3 E 类诊断）** | **高（P0 + P2/P3 gating）** |
| P1 | Advantage normalization & clipping | 对 k-step return 做 per-batch normalize + clip | 低（几行代码） |
| P1 | λ-return OPD | geometric mixture over n-step returns：`G_t^λ = (1-λ) Σ_n λ^{n-1} G_t^{(n)}` | 低（标准 GAE 公式） |
| P2 | Adaptive-k by teacher perplexity | `k_t = f(teacher_ppl_t)`，teacher 不确定时缩短 horizon | 中 |
| P3 | Adaptive-k by overlap ratio | `k_t = f(overlap_t)`，student-teacher 分布重叠低时缩短 horizon | 中（需要额外计算 overlap） |
| P4 | Min-form lookahead | `A_t = min(r_t, r_{t+1}, ..., r_{t+k-1})` 替代 sum-form | 低（几行代码） |
| P5 | Clipped-sum lookahead | `A_t = clip(Σ γ^j r_{t+j}, -c, c)` | 低 |
| P7 | FIPO-style tricks（借鉴 arXiv 2603.19835） | soft decay window + dual-clip mask + 乘性 reweight，三个正交 trick | 中（loss.py 内即可） |

> 注：原 P6 (top-K local support) 已升级为 P0。原 "uniform fixed-k sweep" 不再列为 variant —— 它在 Phase 2 作为 baseline/现象复现存在，不是 candidate method。

### 5.2 P0: Teacher Top-K Local Support Matching（新 main-method 基石）

来源：Revisiting OPD (2603.25562)。把 sampled-token 单步比较换成"在 teacher top-K support 上的 truncated reverse-KL"，让单步信号更干净、更 robust to tokenizer/outlier。这是后续 adaptive lookahead 的前提（在干净信号上耦合 future 才有意义）。

**核心 loss**（per prefix `c_t`）：
```
S(c_t) = TopK_teacher(c_t)                       # teacher 概率最高的 K 个 token
# 在 support 内 renormalize 两边
π̂(v) = π(v) / Σ_{u∈S} π(u);  q̂(v) = q(v) / Σ_{u∈S} q(u)
L_LSM(c_t) = Σ_{v∈S} π̂(v) · log(π̂(v) / q̂(v))   # truncated reverse-KL
```

**三个必须的 stabilization（缺一不可，他们 ablation 验证）**：
1. **Support-set renormalization**：truncated support 上必须重新归一化，否则训练崩
2. **Top-p rollout sampling**（p=0.9）：让 rollout 留在 teacher 可信区，否则 teacher 信号不可靠
3. **Special-token masking**：消除 tokenizer/special-token artifact（他们这一项单独 +4.3pt）

**实现要点**：
- slime rollout 需要取 teacher 的 top-K logprobs（`top_logprobs_num`，5/11 worklog 已验证 sglang 支持 K=50），不再只是 sampled-token logprob
- K 不要太小（ablation 显示太小 hurt），默认 K=20-50；对 K 不敏感只要够大
- 改动点仍在 `apply_opd_kl_to_advantages()` 附近 + rollout reward func 取 top-K

**回归基线**：先单独跑 P0（k=1 + top-K），确认能复现 2603.25562 的提升趋势（sampled-token < +mask < top-K），再叠加 adaptive lookahead（P0+）。

### 5.3 P1: Advantage Normalization & Clipping

这应该在 Phase 2 就加上，但作为 ablation 单独测试效果。

```python
# normalize
k_step_kl = (k_step_kl - k_step_kl.mean()) / (k_step_kl.std() + 1e-8)
# clip
k_step_kl = torch.clamp(k_step_kl, -clip_value, clip_value)
```

测试 clip_value ∈ {3.0, 5.0, 10.0, ∞}

### 5.4 P1: λ-Return OPD

标准 GAE/TD(λ) 思路，但用 OPD reward：

```python
def lambda_return_opd(rewards, gamma, lam, mask):
    """
    G_t^λ = r_t + γ * [(1-λ) * r_{t+1} + λ * G_{t+1}^λ]
    等价于 GAE 但 value baseline = 0
    """
    T = rewards.shape[-1]
    returns = torch.zeros_like(rewards)
    g = 0.0
    for t in range(T - 1, -1, -1):
        if mask[t]:
            g = rewards[t] + gamma * lam * g
        else:
            g = 0.0
        returns[t] = g
    return returns
```

Sweep：λ ∈ {0.0, 0.5, 0.8, 0.95, 1.0}（λ=0 退化为 k=1，λ=1 退化为 full return-to-go）

这个 variant 特别有吸引力，因为它用一个连续参数 λ 平滑地插值 bias-variance tradeoff，比 discrete k 更优雅。

### 5.5 P2: Adaptive-K by Teacher Perplexity

核心思路：teacher 在当前 prefix 上 perplexity 高 → teacher 自己也不确定 → 不应该信任 future teacher signal → 缩短 k。

```python
teacher_ppl_t = torch.exp(-teacher_log_probs[t])  # per-token perplexity
confidence_t = 1.0 / (1.0 + teacher_ppl_t / ppl_threshold)
effective_gamma_t = gamma * confidence_t  # position-dependent discount
```

或者更简单：
```python
# 当 teacher ppl 超过阈值时，截断 future reward
if teacher_ppl[t+j] > threshold:
    break  # 不再累加更远的 future reward
```

### 5.6 P4: Min-Form Lookahead

来自 process-reward 文献（PURE）的启发：sum-form credit assignment 容易 reward hacking，min-form 更稳定。

```python
def min_form_lookahead(rewards, k, mask):
    T = rewards.shape[-1]
    returns = torch.zeros_like(rewards)
    for t in range(T):
        horizon = min(k, T - t)
        valid_rewards = [rewards[t+j] for j in range(horizon) if mask[t+j]]
        if valid_rewards:
            returns[t] = min(valid_rewards)  # 或 torch.min
    return returns
```

直觉：如果 k-step window 内任何一个 token 的 reverse_kl 很大（student 和 teacher 差异大），就给当前 token 一个强信号。这比 sum-form 更 robust to outliers。

### 5.6b P7: FIPO-Style Tricks（借鉴 arXiv 2603.19835）

FIPO 是 RLVR 侧的 future-discounted credit assignment（不是 OPD），但它的三个工程 trick 和我们 K-step cumulative OPD 高度同构，可以直接迁移。三个 trick 正交，建议先单独 ablate 再组合。

**背景对照**：FIPO 的原子信号是 `Δlogp = logπ_θ - logπ_old`（policy self-shift），我们的是 `r_t = logπ_S - logπ_T`（teacher gap）。FIPO 已在 Qwen2.5-32B 上验证 future-discounted reweight +6pt AIME，是我们 cumulative OPD 最强的 motivation citation。下面把它的稳定化手段搬到 OPD reward-to-go 上。

#### Trick 1: Soft Decay Window（替代 hard horizon k）

FIPO 用指数衰减窗口而非硬截断，避免 boundary artifact：
```python
# gamma = 2^{-1/tau}，tau = half-life（FIPO 用 tau=32）
gamma = 2.0 ** (-1.0 / tau)
# full-suffix discounted sum，但 gamma 自带 soft cutoff，不需要硬 k
G_t = sum_{k>=t} gamma^{k-t} * r_k
```
这其实就是我们 λ-return（5.3）在 λ=1、γ=2^{-1/τ} 时的特例，但用 half-life τ 参数化更直观。我们 5/29 KL dump 分析显示 K=8 是 structural sweet spot → 对应 τ≈8-12。**Sweep τ ∈ {4, 8, 16, 32}**，和 hard-k 版对比哪种更稳。

#### Trick 2: Dual-Clip Mask（防 reward-to-go variance 爆炸）⭐ 最有价值

这是我们目前完全没做、但最该加的 trick。5/29 分析证实 sum-form reward-to-go 的 variance ∝K（full 时 368,000×），如果 opd-4b-C 做真正的 reward-to-go（去掉 ÷K 归一化），几乎一定会 variance 爆炸。FIPO 的解法：把 IS ratio 超阈值的 outlier token 从未来累积里 mask 掉。
```python
# M_k = 1 仅当该 token 的 IS ratio 在阈值内，否则 0（从 future sum 剔除）
M_k = (is_ratio_k <= c)          # c >= 10 (FIPO 用 dual-clip threshold)
G_t = sum_{k>=t} M_k * gamma^{k-t} * r_k
```
在我们的 OPD 场景里，IS ratio = `exp(logπ_θ - logπ_old)`（rollout 时的 behavior policy）。slime 训练里有 old logprob，可以算。**这个 mask 是让 sum-form reward-to-go 在我们这边变可行的关键前提。**

#### Trick 3: 乘性 Reweight（vs 我们的加性 reward-to-go）

FIPO 不直接把 future-KL 当 reward 相加，而是当 advantage 的乘性 modulation：
```python
f_t = clip(exp(FutureKL_t), 1 - eps_low, 1 + eps_high)   # FIPO 用 [0.8,1.2] / [1,1.2]
A_tilde_t = A_base_t * f_t
# 且对 negative-advantage + 大 IS ratio 的 token，reset f_t = 1
```
我们当前是加性：`A_t = A_base - λ Σ γ^d r_{t+d}`。乘性版的语义不同（放大/衰减已有 advantage 而非叠加新 reward），值得作为一个独立 arm 对比。**注意**：纯 OPD 下 reward=0、A_base≈0，乘性 reweight 会失效（0 乘任何数还是 0）。所以这个 trick 只在 **OPD + verifiable reward 混合**（A_base 非零）时有意义 → 顺便定义了一个新实验设定（见下）。

#### 衍生设定：OPD + RLVR 混合 advantage

Trick 3 天然引出一个组合：`A_t = (teacher-student OPD term) + (verifiable reward GRPO term)`，再用 FIPO 乘性 future-KL reweight。这把我们的 cumulative OPD 和 FIPO 的 RLVR 缝合，可能是 Phase 4 的一个 stretch arm（也回应 Lightning-OPD 把 OPD 和 RLVR 对立的设定）。

#### P7 实验子矩阵

| Arm | 配置 | 对照 |
|-----|------|------|
| soft-decay | λ=1 + γ=2^{-1/τ}, τ∈{4,8,16,32} | vs hard-k (Phase 2) |
| dual-clip-mask | sum-form reward-to-go (无 ÷K) + mask, c∈{5,10,20} | vs mean-form (slime v2) |
| mult-reweight | A_base·clip(exp(FutureKL)), 需 reward≠0 | vs 加性 (5.1) |
| opd+rlvr-fipo | 混合 advantage + FIPO reweight | Phase 4 stretch |

### 5.7 实验矩阵

| Variant | 关键参数 | Sweep |
|---------|----------|-------|
| **P0 top-K local support** ⭐ | **K (support size)** | **{20, 50}（+renorm +top-p +mask 固定开）** |
| **P0+ top-K + adaptive lookahead** ⭐ | **gating signal** | **teacher-ppl / overlap，配 best K** |
| λ-return | λ | {0.0, 0.5, 0.8, 0.95, 1.0} |
| Adaptive-k (ppl) | ppl_threshold | {5, 10, 20, 50} |
| Min-form | k | {2, 4, 8} |
| Clipped-sum | clip_value | {3.0, 5.0, 10.0} |
| Normalize + best-k | normalize={T/F} | 和 Phase 2 最佳 k 组合 |
| **FIPO soft-decay** | **τ (half-life)** | **{4, 8, 16, 32}** |
| **FIPO dual-clip-mask** | **c (IS ratio thresh)** | **{5, 10, 20}** |

每个 variant 跑 2 seeds。**主对照不是 naive k=1，而是 P0（k=1 + top-K local support）**——见 §1.4 升级后的 Go/No-Go。

### 5.8 Phase 3 交付物

- [ ] 所有 variants 的完整结果表
- [ ] 最佳 variant 的 accuracy-stability frontier 图（x=advantage variance, y=eval accuracy）
- [ ] **P0（做好的单步）vs P0+（单步+adaptive lookahead）的 head-to-head**（这是判定 lookahead 是否有增量价值的关键对照）
- [ ] 复现 2603.25562 的趋势：sampled-token k=1 < +mask < top-K local support
- [ ] 确定论文的 main method（最可能是 λ-return 或 adaptive-k）

---

## 6. Phase 4: Scale-Up & Cross-Domain Validation

**时间**：2-3 周
**目标**：在更大模型和更多 domain 上验证 Phase 3 的最佳方案

### 6.1 Scale-Up 实验

| 配置 | Student | Teacher | Data | Rollout len |
|------|---------|---------|------|-------------|
| Pilot（已完成） | Qwen2.5-3B | Qwen3-8B | 3-5k prompts | 2048-4096 |
| Main | Qwen2.5-7B | Qwen3-14B/32B | 10-17k prompts | 4096-8192 |
| Stretch | Qwen3-8B | Qwen3-32B | 17k+ prompts | 8192-16384 |

### 6.2 Cross-Domain 实验

Math reasoning 之外，至少加一个 domain 证明 generality：

**选项 A：Summarization**
- Data: XSum 或 TL;DR
- Metrics: ROUGE, pairwise LLM-judge preference
- 理由：KETCHUP 在 XSum 上有 k-step 正面结果

**选项 B：Translation**
- Data: Europarl EN-NL 或 WMT
- Metrics: BLEU, chrF, TER
- 理由：KETCHUP 在 Europarl 上也有正面结果

**选项 C：Code Generation**
- Data: code reasoning prompts
- Metrics: pass@k on coding benchmarks
- 理由：multi-token prediction 在 code 上效果最好，k-step OPD 可能也是

建议至少做 A 或 B 中的一个。

### 6.3 Strong Baseline 对比

Phase 4 需要和更强的 baseline 对比，不能只和 k=1 sampled-token OPD 比：

| Baseline | 来源 | 为什么要比 |
|----------|------|-----------|
| Top-K OPD | revisiting_opd / THUNLP OPD | 当前最强 local signal 修复方案 |
| Veto | Veto paper | Adaptive intermediate target，不用 future credit |
| StableOPD | Demystifying OPD | Reference divergence + rollout mixture |
| GRPO/REINFORCE++ (no OPD) | Slime 自带 | 纯 RL baseline，没有 teacher signal |

### 6.4 Phase 4 交付物

- [ ] Scale-up 结果（7B student）
- [ ] Cross-domain 结果（至少一个非 math domain）
- [ ] 和 strong baselines 的对比表
- [ ] 确认 main result 是否 robust

---

## 7. Phase 5: Paper Writing & Final Experiments

**时间**：2-3 周
**目标**：写论文，补充必要的 ablation 和 analysis

### 7.1 论文结构（草案）

1. **Introduction**：OPD 的 myopic credit assignment 问题，k-step 作为自然解决方案，但 naive k-step 有 variance 问题，需要 adaptive/gated approach
2. **Background**：OPD formulation，token-level vs sequence-level，reward-to-go decomposition
3. **Method**：k-step return construction，λ-return variant，adaptive gating mechanisms
4. **Experiments**：
   - Fixed-k sweep（Phase 2 结果）
   - Advanced variants comparison（Phase 3 结果）
   - Scale-up and cross-domain（Phase 4 结果）
   - Ablations and diagnostics
5. **Analysis**：
   - Bias-variance tradeoff 的实证分析
   - When does lookahead help?（autocorrelation, teacher confidence, support overlap）
   - Failure modes and mitigations
6. **Related Work**
7. **Conclusion**

### 7.2 必须补充的 Ablation

- [ ] k 和 γ 的独立效果（不只是 grid search，要有 interaction analysis）
- [ ] Sampled-token reward vs top-K local reward（如果 Phase 3 没做 P6）
- [ ] With/without reference-KL anchor
- [ ] With/without advantage normalization
- [ ] Greedy decoding vs pass@k evaluation
- [ ] 不同 rollout temperature 的效果

### 7.3 统计检验

- Exact-match reasoning benchmarks：McNemar's test 或 paired bootstrap over prompts
- ROUGE/BLEU/chrF：paired bootstrap resampling
- 所有 automatic metrics：3-5 random seeds，报告 mean ± std
- 优先做 prompt-level paired testing，而不是只看 seed-level averages

### 7.4 Human Eval（如果需要）

只在 final 2-3 candidates 上做：
- Pairwise preference：k=1 vs best-k vs best-adaptive
- 评估维度：correctness, coherence, conciseness
- 样本量：100-200 prompts

---

## 8. 资源估算

### 8.1 GPU 时间估算

| Phase | 配置 | 估算 GPU-hours (A100-80GB) |
|-------|------|---------------------------|
| Phase 0 | Smoke test | ~10 |
| Phase 1 | 3B student, 3 seeds, 500 steps | ~100-200 |
| Phase 2 | 3B student, 9 configs × 2 seeds | ~400-800 |
| Phase 3 | 3B student, ~15 configs × 2 seeds | ~600-1000 |
| Phase 4 | 7B student, 5-8 configs × 3 seeds | ~2000-4000 |
| Phase 5 | Ablations + final runs | ~500-1000 |
| **Total** | | **~3600-7000** |

### 8.2 关键成本驱动因素

- **Teacher scoring 是主要成本**：k-step 本身几乎不增加计算（只是一个 O(BT) reverse scan）
- **Rollout length 是第二大成本驱动**：从 2048 到 16384 会 ~8x 增加 rollout 时间
- **Seeds 数量**：3 seeds 是最低要求，5 seeds 更好但成本 ~1.7x

### 8.3 时间线总览

| Phase | 时间 | 累计 |
|-------|------|------|
| Phase 0: Smoke Test | 2-3 天 | 第 1 周 |
| Phase 1: Baseline | 5-7 天 | 第 2 周 |
| Phase 2: Fixed-K Sweep | 5-7 天 | 第 3-4 周 |
| Phase 3: Advanced Variants | 2-3 周 | 第 5-6 周 |
| Phase 4: Scale-Up | 2-3 周 | 第 7-9 周 |
| Phase 5: Paper | 2-3 周 | 第 10-12 周 |

**总计：约 10-12 周**，如果 Phase 2 就有清晰正面结果可以加速。

---

## 9. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解策略 |
|------|--------|------|----------|
| Variance 随 k 指数增长 | 高 | 高 | 先从 k=2 开始；advantage normalization + clipping；λ-return 平滑；**FIPO dual-clip mask 剔除高 IS-ratio outlier 的 future 累积（P7 Trick 2）** |
| Teacher 在 off-support prefix 上不可靠 | 高 | 高 | Phase 3 加 perplexity gating；只在 teacher confident 时用 future signal |
| Length inflation / repetition collapse | 中 | 高 | 每个实验都监控 length 和 repetition；加 reference-KL anchor |
| Fixed-k 全面不如 k=1 | 中 | 中 | 这本身是有价值的 negative result；pivot 到 adaptive variants |
| Slime OPD 不是 pure OPD | 低 | 中 | 设 reward=0 近似 pure OPD；如果不够纯，考虑 fork 一个 pure OPD mode |
| Tokenizer mismatch | 低 | 高 | 严格使用 same-family same-tokenizer pairs |
| SGLang/Slime 版本不兼容 | 中 | 中 | Phase 0 就验证；pin 版本号 |

---

## 10. 最终交付物清单

- [ ] 可复现的代码（基于 Slime fork，包含 k-step OPD 实现）
- [ ] 完整的实验结果表和图
- [ ] Diagnostics dashboard（wandb project）
- [ ] 论文初稿
- [ ] 补充材料（所有 ablation 结果、hyperparameter sensitivity）

---

## 附录 A: 关键代码位置（Slime）

| 文件 | 函数/位置 | 作用 |
|------|-----------|------|
| `slime/backends/megatron_utils/loss.py` | `apply_opd_kl_to_advantages()` | **主改动点**：single-step → k-step |
| `slime/rollout/on_policy_distillation/reward_func.py` | reward function | Teacher logprob scoring via SGLang |
| `slime/rollout/on_policy_distillation/post_process_rewards.py` | post-processing | 截取 teacher logprobs 到 response span |
| `examples/on_policy_distillation/run-qwen3-8B-opd.sh` | 训练脚本 | 官方 OPD example，改配置的起点 |

## 附录 B: 关键参考文献

| 简称 | 全称 | 与本项目的关系 |
|------|------|---------------|
| Revisiting OPD | Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes (arXiv 2603.25562, CASIA) | ⚠️ 我们 K-step 方向的**最强反例**：证明 naive fixed-γ return-to-go variance O(T⁴)、toy 上 γ=1 drift → uniform lookahead 不可发表。同时提供**互补 fix**（teacher top-K local support matching，已升为本项目 P0）+ 三个可直接复用的 failure-mode diagnostics（见 §3.3 E 类）。必引、必差异化。 |
| Rethinking OPD | Rethinking On-Policy Distillation of LLMs | Sampled-token / full-vocab / top-K taxonomy，overlap ratio diagnostics |
| MiniLLM | MiniLLM | 推导了 reverse-KL reward-to-go，但发现 single-step decomposition 更稳定 |
| KETCHUP | KETCHUP | 最强正面证据：k-step returns 在 RL-based KD 上有效 |
| Demystifying OPD | Demystifying OPD | Length inflation / truncation collapse failure modes |
| SCOPE | SCOPE | Signal quality gating，支持 adaptive k 的思路 |
| Veto | Veto | Adaptive intermediate target，k-step 的竞争方案 |
| PURE | PURE | Min-form credit assignment，替代 sum-form |
| FIPO | Future-KL Influenced Policy Optimization (arXiv 2603.19835) | RLVR 侧的 discounted future-KL reward-to-go，与本项目跨范式同构；提供 soft-decay window + dual-clip mask + 乘性 reweight 三个可迁移 trick（见 Phase 3 P7）。是 cumulative OPD 的 motivation citation，但它无 teacher，不抢占 "future-teacher-KL OPD" 的 novelty |
