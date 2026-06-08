# Session Summary — 2026-06-01/02 (Day 31-32)

## Project: K-Step OPD (On-Policy Distillation)
**Status**: 文献对标完成 + research-plan 按 Revisiting-OPD 重构 + Phase 2.5 (sum vs mean) 实验启动 (R4/R1 跑 mean 半边) + loss.py 加 sum-K/dual-clip/token_ids 修复 (过回归测试待部署)

---

## TL;DR

1. **OPD 文献 AIME 对标**：搜了 7 篇 OPD 论文。同量级 (Qwen3-4B+8B teacher) OPD 后 AIME24 健康水平 = **65-68%**；我们 55.8% 落后 ~10pt，瓶颈全在 SFT baseline (48.8 vs 文献 56-57)。
2. **FIPO (2603.19835)** = 跨范式同构工作（RLVR 侧的 discounted future-KL reward-to-go）。三个 trick (soft decay / dual-clip mask / 乘性 reweight) 整合进 research-plan P7。
3. **Revisiting OPD (2603.25562)** = 我们 K-step 方向的**最强反例**：证明 naive fixed-γ return-to-go variance O(T⁴)、toy 上 γ=1 drift。据此重构 research-plan：novelty 收窄到 conditional/gated；top-K local support 从 P6 升 P0；Go/No-Go 基线升级。
4. **Phase 2.5 (sum vs mean 决战) 设计 + 启动**：R4 (mean-K=4, p5-3) + R1 (mean-K=8, p5-2) 跑 mean 半边。2×2 析因 (sum/mean × mask on/off)，主判据 R3 vs R3b。
5. **训练加速 ~2.7x**：瓶颈实测是 actor_train (40%, 显存空闲却开 recompute)。fast config (关 recompute + max_tokens 16384) 把 actor_train 131s→30s。max_tokens_per_gpu 确认对 PPO 训练零数学影响。
6. **loss.py 改动过回归**：`--opd-agg {sum,mean}` + `--opd-dualclip-c` (FIPO mask) + token_ids dump 修复 (5/23 起的 bug)。回归测试抓到"slime v2 full-horizon 其实是 sum 不是 mean"。

---

## 1. OPD 文献 AIME 对标 (Day 31)

搜了 7 篇，提取 AIME 分数 (协议各异，标注后看净增益而非绝对值)：

| 论文 | arXiv | 关键 AIME24 |
|---|---|---|
| Lightning OPD (对标) | 2604.13010 | 4B SFT 56.7 → OPD 65.4 / Lightning 68.1; 8B 69.9 |
| ExOPD / G-OPD | 2602.12125 | 4B-NT OPD 55.0 → ExOPD(λ=1.25) 58.7 |
| EOPD | 2603.07079 | 4B Pass@8 OPD 26.7 → 36.7 (non-thinking, max=8192) |
| TML OPD blog | thinkingmachines | 8B SFT 60 → OPD 70; Qwen3 报告 OPD 74.4 |
| ATESD | 2605.11458 | 4B self-distill +2.05 Avg@12 |
| DED (off-policy) | 2508.09883 | 32B 800-sample 81.87 (蒸馏天花板) |
| FIPO | 2603.19835 | 32B DAPO 50 → 56 (RLVR future-KL, 见 §2) |

**结论**：4B+8B teacher 下 OPD 后健康 = 65-68%。我们 +7pt 相对增益和文献一致，绝对分落后 ~10pt 全在 SFT baseline (数据量 1/4)。

---

## 2. FIPO (2603.19835) — 跨范式同构

不是 OPD，是 GRPO/DAPO 路线的 RLVR，但核心 = discounted Future-KL reward-to-go credit assignment，和我们 K-step cumulative 同构。

三个可迁移 trick (整合进 research-plan Phase 3 P7)：
- **Soft decay window** `γ=2^{-1/τ}` (替代 hard horizon K)
- **Dual-clip mask** ⭐ (IS ratio 超阈剔除未来累积，防 variance 爆炸)
- **乘性 reweight** `A·clip(exp(FutureKL))` (纯 OPD reward=0 时失效，只适合 OPD+RLVR 混合)

FIPO 无 teacher (policy self-shift)，所以 "future-teacher-KL OPD" 仍是文献空白。

---

## 3. Revisiting OPD (2603.25562) — 最强反例 ⚠️

它定义的 discounted return-to-go `ĝ_γ` = 我们 cumulative OPD 同一个估计量。核心结论：
- 理论：token-level variance O(T²)，sequence-level O(T⁴)
- toy：γ↑ → gradient variance↑，γ=1 policy drift 不收敛
- **它的应对是退回 token-level (γ=0)，改善单步质量 (top-K local support)，而非修 future coupling**

三个 failure mode (和我们诊断互补)：imbalanced one-token signal / teacher 在 student prefix 上不可靠 (Fig 4: 后段 log-prob gap 更宽) / tokenizer mismatch。

### research-plan.md 据此重构 (4 处)
1. **§1.3 核心假设**：加边界声明——naive uniform fixed-k/γ 已被证否，立足点收窄到 conditional/gated + 先做干净单步
2. **§1.4 + §4.7 Go/No-Go**：基线对手从 naive k=1 升级为 k=1 + top-K local support
3. **§5.1/§5.2**：top-K local support 从 P6 升 **P0**，新增 P0+ 组合 main method (top-K 修单步 + adaptive lookahead)
4. **§3.3 Diagnostics**：加 E 类 failure-mode 诊断 (复用其 Fig 2/3/4) + 写死 KL dump 必存 token_ids

**新项目重心**：从"纯 K-step"挪到"top-K local support + adaptive/gated lookahead 组合"。两篇文献从相反方向 (Revisiting 反对 naive future / FIPO 支持 clean base 下的 future) 夹出这个定位。

---

## 4. Phase 2.5: sum vs mean 决战 (§4.8)

**核心问题**：slime cumulative v2 是 mean-K (÷K, 偏去噪)；sum-K 才是教科书 reward-to-go (variance ∝K)。分离 "credit assignment" vs "magnitude 稀释/去噪"。**不用 EMA** (sum vs mean 同 K 对比已干净分离)。

三个已定决策：
1. sum 的 kl_coef = instant/K 起点 (量级对齐) → sum-K8 用 0.125
2. K 主测 8 (5/29 sweet spot) 顺带 4
3. dual-clip mask 默认只开 sum；补 R3b=mean+mask (2×2 析因关键角)

干净 2×2 析因 (R1/R2/R3/R3b)：

| | mask off | mask on |
|---|---|---|
| mean | R1 | R3b |
| sum | R2 | R3 |

主判据 **R3 vs R3b** (都开 mask，只差 sum/mean) → credit assignment story 成立与否。R1/R3/R3b 跑 2 seeds (避免 noise，project 第一大教训)。

---

## 5. 训练加速 (Day 32)

### 瓶颈实测 (slime timer.py)
R4 单 rollout ~330s 拆解：train_wait(rollout) 80s/24% | ref_log_probs 38s | log_probs 37s | **actor_train 131s/40% (真瓶颈)** | update_weights 0.4s。

修正之前误判：瓶颈是 actor_train，不是 rollout。actor GPU 显存只用 30%，却开着 full recompute。

### max_tokens_per_gpu 对 RL/PPO 无数学影响 (确认)
读 slime arguments.py：它只控制 dynamic micro-batch 切分，不改 global_batch_size (parse 时 assert 固定)。梯度数学等价。纯吞吐参数。

### fast config 实测 ~2.7x
`MAX_TOKENS_PER_GPU=16384, LOG_PROBS_MAX_TOKENS_PER_GPU=24576, RECOMPUTE=0`：actor_train 131s→30s, log_probs 37s→7s, 单 rollout 330s→~120s。
- ⚠️ 32768 + 无 recompute **OOM** (4B actor)，16384 是显存安全甜点。

---

## 6. loss.py 改动 (过回归，待部署)

1. **`--opd-agg {mean,sum}`**：cumulative 聚合方式
2. **`--opd-dualclip-c`**：FIPO dual-clip mask (IS ratio = exp(log_probs - rollout_log_probs))
3. **token_ids dump 修复**：`unconcat_tokens` → `tokens` (修 5/23 起空 token_ids bug)
4. opd_reverse_kl 注释明确是 instant KL (跨 K/agg 可比)

### 回归测试发现
`scripts/test_opd_agg_regression.py` 对拍抓到：**slime v2 full-horizon (K=-1) 其实是 sum (从不除)，只 truncated (K>0) 才除 actual_k (mean)**。"v2=mean-K" 只对 truncated 成立。修正后 `--opd-agg mean` 严格复刻 v2，三项测试全过 (mean==old / sum==mean×K / dual-clip 剔除 outlier)。

R1/R4 用 truncated K=4/8 → 不受影响。

---

## 7. 关键认知修正

- **opd_reverse_kl log 的是 instant per-token KL**，不是 cumulative 后的值 (读 loss.py 确认)。所以 R4(0.10) vs R1(0.017) **跨 K 可比**，之前说"不可比"是错的。R1 降更多是 mean-K=8 推得更激进的真实信号，但 sampled KL 低 ≠ eval 好 (5/29 教训)。
- **slime OPD 是 PPO/GRPO 路线** (advantage = base + OPD KL penalty)，reward=0 时 advantage≈ -kl_coef×reverse_kl。

---

## 8. 当前运行状态

| 节点 | 实验 | 配置 | 速度 | 进度 |
|---|---|---|---|---|
| p5-3 | R4 (mean-K=4) | 8192, recompute on | ~330s/step | ~184+/300 |
| p5-2 | R1 (mean-K=8) | fast (16384, no recompute) | ~120s/step | ~52+/300 |
| qzf-dev | Qwen3-8B teacher | DP=4, 共用 (96 并发压测零错误) | — | serving |

---

## 9. 踩坑记录

1. **teacher 抖动 → slime pickle bug 杀 job**：`reward_func` 的 `raise_for_status()` 抛带 CIMultiDictProxy 的异常 (不可 pickle) → Ray 崩。修：`scripts/opd_reward_hardened.py` (retry + 转纯 RuntimeError)。
2. **teacher L40S OOM**：mem 0.85 + 256 并发 → 降 0.7 + max-running 24 + context 20480。
3. **p5-2 断 symlink**：student ckpt 是指向已删目录的断链 → 从 p5-3 rsync 真实文件。
4. **transformers 5.3.0 拒绝本地绝对路径** (`validate_repo_id` count("/")>1)：HF_HUB_OFFLINE 无效，降 hub 崩。解法：降 transformers 5.3.0→5.2.0 (megatron-bridge 要求 ≤5.2.0；sglang pin 5.3.0 但实测 5.2.0 可用)。
5. **共享 Ray cluster**：`--net=host` 下多容器共享 Ray，失败 job 的 192 个 prestart worker 持有旧 transformers 内存镜像 → patch 不生效。解法：ray stop --force 清干净重起。
6. **R1 OOM**：32768 + 无 recompute 太激进，降 16384。

---

## 10. 文件清单 (本次新增/修改)

| 文件 | 说明 |
|---|---|
| `research-plan.md` | 按 2603.25562 重构 (P7 FIPO trick + Phase 2.5 + top-K 升 P0) |
| `slime/.../loss.py` | sum-K + dual-clip mask + token_ids 修复 (本地，待部署) |
| `slime/.../arguments.py` | `--opd-agg` + `--opd-dualclip-c` |
| `scripts/test_opd_agg_regression.py` | sum/mean 回归对拍测试 |
| `scripts/train-opd-extteacher-fast.sh` | fast 训练脚本 |
| `scripts/opd_reward_hardened.py` | 加固 reward_func |
| `configs/opd-4b-R4-meanK4.env` / `opd-4b-R1-meanK8.env` | R4/R1 config |
| `session_summary_worklog/worklog.md` | Day 31-32 全程追加 |

---

## 11. 下一步

### 高优先级
1. 等 R4/R1 出 iter_99/199/299 → n=16 eval，对比 mean-K=4 vs K=8 vs baseline (opd-4b-B 48.8/55.8)
2. 部署改好的 loss.py 到第三台机 (p5-4，需同步 student+slime) → 起 R3 (sum-K8 + dualclip-c10 + kl_coef=0.125, main candidate) + R2/R3b/R5/R6
3. 验证修复后 KL dump 真有 token_ids → 做 position/special-token 分析

### 中优先级
4. 画 R4 vs R1 的 instant KL 下降曲线对比 (跨 K 收敛速度，现在可比)
5. 重做 SFT (补数据量 + 不混温度) 拉 baseline 到 ~56% — 文献对标显示这是够到 65% 的关键

### 判读分支 (Phase 2.5)
- R3 > R3b → reward-to-go story 成立
- sum 全败、mean ≈ instant → negative result，重心转 P0 top-K local support
- mean 赢但 sum 不赢 → 诚实改成 "denoising helps"
