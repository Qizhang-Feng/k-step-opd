# Session Summary — 2026-05-26

## Project: K-Step OPD (On-Policy Distillation)
**Status**: 4B Instant OPD 真实 +5.6pt (n=16). Lightning-OPD paper recipe 训练已起 on p5-3.

---

## TL;DR

1. **n=1 eval 把 OPD 信号埋了**: 之前 v2-700 vs OPD iter_299 = 60% vs 53.3% AIME-2024 (n=1) → 推断 OPD 失败。重 eval n=16 → **48.8% vs 54.4% (+5.6pt)** → OPD 是有效的，noise 太大看不出来。
2. **OPD 提升单调**: iter_099=50.2%, iter_199=50.4%, iter_299=54.4% (n=16 AIME-2024). avg_len 同步下降 (55K→52.5K)。
3. **Lightning-OPD paper recipe 训练已起** (p5-3): lr 5e-7→2e-6, max_response 8192→4096, T 0.6→0.8, 600 rollouts. 预计 ~17h。
4. **Greenland 152K-700-steps job** 完成 (5/24)。下载时一度抓错 v0 (8B 残留)。修正后 v1 4B ckpt 在 p5-4 上 eval n=16 中。
5. **Eval 基础设施重写**: `eval_math.py` 加 `avg_pass_at_1`，`scripts/eval-aime-n16.sh`，跨 p5 直连 ssh + rsync (no mac relay)。

---

## n=16 Eval Results

3 台机器并行 (~30 min)，每模型 30 题 × 16 sample (avg pass@1):

| 模型 | AIME-2024 | AIME-2025 | 综合 | avg_len |
|------|:---:|:---:|:---:|:---:|
| **baseline v2-700** | 48.8% | 40.6% | 44.7% | 55K-58K |
| OPD iter_099 | 50.2% | 46.9% | 48.6% | 55K-56K |
| OPD iter_199 | 50.4% | 42.1% | 46.3% | 53K-57K |
| **OPD iter_299** | **54.4%** | **45.2%** | **49.8%** | 52K-57K |
| **Δ (iter_299 vs baseline)** | **+5.6pt** | **+4.6pt** | **+5.1pt** | -2.6K |

参考 Lightning-OPD 4B paper (n=32, standard OPD): 56.7% → 65.4% AIME-2024 (+8.7pt)。

我们 baseline 比 paper 低 8pt（SFT 训练量 4× 少：194K vs 768K samples），OPD gain 比 paper 小 3pt（配置太保守）。

---

## Lightning-OPD Paper Recipe

### Standard OPD 4B settings (Table 6 + Appendix B)

```
training_steps:  150 (× global_batch=256 = 38400 samples)
global_batch:    256
max_response:    4096
lr:              2e-6 constant
weight_decay:    0.1
adam_beta2:      0.98
temperature:     0.8
top_p:           1.0
top_k:           none
advantage_clip:  [-10, 10]
advantage:       on_policy_distillation (advantage = log_T - log_S)
```

### slime 主线 ≠ paper recipe
- slime example: `lr=1e-6, max_response=16384, T=1.0` (8B+SFT → 32B teacher → Math500 76→94%)
- paper: `lr=2e-6, max_response=4096, T=0.8` (4B+SFT → 8B teacher → AIME-2024 56.7→65.4%)
- 两者都 work，差 4-8x 训练量、4x lr、2x rollout长度

### `--include-verifiable-reward` 的真相

- paper 把 OPD 跟 RLVR **对立**，不是组合
- `slime/rollout/on_policy_distillation.py`: `scalar_rewards = [0.0] * len(samples)` 写死
- `--include-verifiable-reward` flag 只用于 wandb logging (help string: "Whether to include the verifiable reward in **the log**")
- `sample.verifiable_rewards` 在 Lightning-OPD repo 里**从未被赋值**（dead code）

### 数学等价

slime 主线没有 `on_policy_distillation` advantage estimator choice，但：
```
我们: --advantage-estimator grpo --use-opd --opd-kl-coef 1.0, reward=0
   advantage = grpo_baseline_advantage(reward=0) - 1.0 × reverse_kl
            = 0 - (log_S - log_T)
            = log_T - log_S    ✓ 等价 paper formula
```
差别只在：paper 多了 advantage clip [-10,10] + distributed whitening。

---

## Lightning Recipe 训练（已起 on p5-3）

### Config: `configs/opd-4b-lightning-recipe.env`

| 参数 | 之前 (instant) | **Lightning recipe** | 倍数 |
|---|:---:|:---:|:---:|
| LR | 5e-7 | **2e-6** | 4× |
| MAX_RESPONSE_LEN | 8192 | **4096** | ÷2 |
| TEMPERATURE | 0.6 | **0.8** | +0.2 |
| TOP_P | 0.95 | **1.0** | disable |
| TOP_K | 20 | **-1** | disable |
| NUM_ROLLOUT | 300 | **600** | 2× |
| ROLLOUT_BATCH_SIZE | 16 | **64** | 4× |
| GLOBAL_BATCH_SIZE | 64 | **256** | 4× |
| OPD_KL_COEF | 1.0 | 1.0 | ✓ |
| MAX_TOKENS_PER_GPU | 8192 | 16384 | 2× |

### Step 1-3 监控

```
                 r1     r2     r3
opd_reverse_kl  0.139  0.142  0.144   (略升，lr 大了)
truncated_ratio 0.92   0.95   0.96    ⚠️ 96% 顶到 max_response
response_length 4012   4041   4083
grad_norm       1.04   1.04   1.04    ✓ 稳定
loss            0.139  0.142  0.144
```

⚠️ **truncation 96%**：student 在 dapo 题目上的 thinking 普遍 >4K，几乎全部顶到上限。这是 paper 的设定（"increasing beyond 4096 doesn't help"），但他们 train 短题 → eval 长题（AIME 32K）的 transfer 是关键风险。

预计：
- 100s/step × 600 = ~17h 训练时间
- 第一个 save 在 rollout 50 (~1.5h 后)，可早 eval 看效果
- KL 是否开始下降是"学到东西"的标志

---

## 152K-700-steps Greenland 实验回收

5/23 提交，5/24 完成 (`8bfd9b6e-...`)。**目的**：测试 152K (mix A+B) 短训练 (~1.18 epoch) 能否避开 152K × 3ep 崩溃。

### Greenland 输出多版本子目录踩坑

S3 下两个 versioned dirs:
- `v0-20260523-054547`: 同实例之前残留的 8B 训练，args.json `model: Qwen3-8B-Base`
- `v1-20260524-070537`: 正确的 4B-152K-700-steps，args.json `model: Qwen3-4B-Base` ✓

第一次抓错 v0 → config hidden_size=4096 → 一度怀疑 job 配错。**教训：list versioned dirs + 检查 args.json 的 model path 再 pull**。

### v1 训练摘要

| 指标 | 值 |
|---|---|
| Steps | 700 (cosine over 700) |
| Epoch over 152K | 1.44 |
| Final loss | 0.219 |
| Runtime | 10h 13min on 8×H100 |

| 实验 | Loss | AIME-2024 (n=16) |
|---|:---:|:---:|
| v2 79K-A × 3ep | 0.246 | 48.8% (baseline) |
| 152K-3ep (mix × 3ep) | 0.196 | 3% (n=1 之前) |
| **152K-700-steps (mix × 1.18ep)** | **0.219** | **eval 中 (10/30: 51.2%)** |

**预测**：最终 ~45-52%。早停减轻但不消除混合数据陷阱。不是显著的 SFT 升级，但比 152K-3ep 好。

---

## Eval 基础设施重写

### `eval_math.py` 修复

**Bug**: `pass1 = 1 if rewards[0] > 0 else 0` — 浪费 n_samples-1 个数据点。

**Fix**: 加 `avg_pass_at_1 = mean(rewards over n)` 字段，向后兼容保留 `pass_at_1` (= first sample only)。

### SGLang n=16 routing fix

**问题**: 单请求 `n=16` → 绑定一个 SGLang DP replica → 16 sample × 30K KV 超过单 replica 32K → 全部 400 Bad Request → server 崩溃。

**Fix**: 拆成 16 个独立 `n=1` 请求 ThreadPool 并发发出 → SGLang router 自动分配到 8 replica → 每 replica 实际 ~2 sample × 30K = 60K KV，正常。

### 跨节点 SSH 直连 (no mac relay)

p5-3 ~/.ssh/config 加:
```
Host p5-2-int
  HostName 172.31.3.242
  User ubuntu
  IdentityFile ~/.ssh/dl-machine-ohio.pem
Host p5-4-int
  HostName 172.31.0.227
```

7.6G HF model rsync ~17s (500 MB/s 内网)。

---

## KL Dump 离线分析（接昨天）

### 数据格式问题

`prompt_token_ids` / `response_token_ids` 都是空 list — 注入脚本只存了 logprobs。无法做"`</think>` token 是 KL outlier"分析。

### 用 reverse_kl 序列做位置/cumulative 分析（中点 r150）

| K | Pearson r(instant_kl, future_kl_K) | var(future_kl_K) / var(instant_kl) |
|:---:|:---:|:---:|
| 2 | 0.72 | 0.52 |
| 4 | 0.53 | 0.28 |
| 8 | 0.38 | 0.15 |
| 16 | 0.28 | 0.09 |
| 32 | 0.21 | 0.05 |
| full | 0.05 | 0.0015 |

**K=8 是 cumulative OPD sweet spot**：方差缩小 6.5x（noise reduction），但和 instant 还有 60%+ 独立信号。

**Position 分析**：头部 0-30% pos KL 偏高 (0.17-0.22)，后部偏低 (0.11-0.16)。没看到尾部 spike，但因为没 token IDs 无法验证 special token 假设。

### 文件

- `scripts/analyze_kl_dumps.py` — 主分析脚本
- `kl_analysis/dumps/` — 116 jsonl, 357MB
- `kl_analysis/figures/{instant_kl_over_training,instant_kl_distribution,future_kl_correlation,snr_by_K,kl_by_position}.png`
- `kl_analysis/summary.json`

---

## Slime / Lightning-OPD eval 设定 vs 我们

| 项 | Slime (Lightning-OPD) | 我们 |
|---|---|---|
| Reward function | `get_deepscaler_rule_based_reward` | **same** ✓ |
| Aggregation | mean over (n_problems × n_samples) | mean(per_problem(mean over n_samples)) — **等价** |
| Chat template | `apply_chat_template(..., add_generation_prompt=True)` | 手动 `<\|im_start\|>...assistant\n<think>\n` |
| `<think>` prefix | ❌ 不加 | ✅ 强制（不影响 v2 SFT 行为，10/10 都自动 emit） |
| n samples | 多次独立 n=1 + 不同 seed | 16 个独立 n=1 并发请求 |
| max_tokens | 32768 | 30000 (Qwen3-4B-Base context limit) |
| paper n_samples | **32** | 我们 16 |

我们 n=16 比 paper 少一半但已足够压住 noise 到 ±2pt 范围。

### Sanity check 结果（v2-700 SFT 模型, 1 题 × 8 sample）

| Format | starts with `<think>` | has `</think>` | has `\boxed{}` | avg_len |
|---|:---:|:---:|:---:|:---:|
| A: paper template ("Please reason..." + Qwen3 chat) | 8/8 | 8/8 | 8/8 | 10,280 |
| B: slime apply_chat_template (default) | 8/8 | 8/8 | 8/8 | 11,844 |
| C: 我们 (manual `<think>\n` prefix) | 0/8 (在 prompt) | 8/8 | 8/8 | 13,194 |

→ 三种格式都有完整 `</think>` + `\boxed{}`，不影响 termination 行为。我们的 prefix 只是把 `<think>` 从 generation 移到 prompt，等价。

---

## 当前运行任务

| 节点 | 任务 | 状态 |
|---|---|---|
| p5-3 | Lightning-recipe 4B OPD 训练 | step 3/600，~17h total |
| p5-4 | 152K-700-steps n=16 eval | 10/30 AIME-2024 (51.2%)，~30 min |
| p5-2 | idle | 可用于 cumulative ablation |
| qzf-dev | Qwen3-8B teacher (DP=4) | serving，用于 Lightning-recipe |

### 关键 checkpoint 位置

| 文件 | 位置 |
|---|---|
| 4B Instant OPD ckpts | p5-3:/opt/dlami/nvme/qzf/models/opd-4b-v2-ckpt700-instant/iter_* |
| 4B Instant HF (3 ckpts ×8G) | p5-3 + p5-2/p5-4（已同步） |
| 152K-700-steps SFT (4B 正确) | p5-3 + p5-4: sft-qwen3-4b-full-152k-700steps-ckpt700/ |
| n=16 eval results | 各 p5-X:/workspace/k-step-opd/eval_results_n16/ |
| Lightning-recipe ckpts (running) | p5-3:/root/.cache/huggingface/opd-4b-lightning-recipe/iter_* |

### 关键代码 / config

- `configs/opd-4b-lightning-recipe.env` ← NEW
- `scripts/train-opd-extteacher.sh` (synced p5-3 latest)
- `scripts/eval-aime-n16.sh` ← NEW
- `scripts/eval-aime-n16-p5-3.sh` ← NEW (串行 2 model)
- `scripts/collect-n16-results.sh` ← NEW
- `scripts/sanity_check_chat_template.sh` ← NEW
- `scripts/analyze_kl_dumps.py` ← NEW (从昨天)
- `eval_math.py` ← patched: `avg_pass_at_1` + n=1 splitting
- `Lightning-OPD/2604.13010v2.pdf` ← paper（10 pages + appendix）
- `sft_all_experiments_curves.png` ← 加上 152K-700-steps（deeppink）

---

## 下一步

### 高优先级（等当前任务）
1. 等 152K-700-steps eval 完整 30/30，确认 AIME 数字
2. 等 Lightning-recipe 训练 50 / 100 / 200 / 600 rollouts，分批 eval
3. 比较：Lightning-recipe vs Instant OPD 真实差距（n=16）

### 中优先级
4. 如果 Lightning-recipe 也只到 +5pt，**换更大 teacher**（Qwen3-32B）：gap 13pt → 25pt+
5. 如果 Lightning-recipe 给 +8pt 甚至 +10pt，**继续做 cumulative K=8 ablation**（KL dump 分析支持）

### 低优先级 / 长期
6. 重生成 SFT data：温度统一 0.7 + 不混合
7. 重做 SFT：300K prompts × 3000 steps（Lightning-OPD 4× 我们的训练量）
8. 写 paper outline

---

## 教训汇总

1. **n=1 pass@1 是 noise 机器**。30 题 ×1 标准误 ±9pt。任何 ±5pt 之内的差距都是 noise。Lightning-OPD paper 用 n=32 是对的。

2. **Greenland job 多版本子目录**。同一 instance 上之前残留的 ms-swift 输出（v0-...）会和当前 job（v1-...）共存。pull 之前 list versioned dirs + 看 args.json 里 model path。

3. **slime 配置 ≠ Lightning-OPD 完整 recipe**。两者都 work 但参数差很多。slime 主线 example 用 `lr=1e-6, max_response=16384, T=1.0`，paper 用 `lr=2e-6, max_response=4096, T=0.8`。差别影响很大。

4. **`--include-verifiable-reward` 是 logging 不是 advantage**。在 Lightning-OPD fork 的 slime 里也只是给 wandb 看 task accuracy，不进入梯度。

5. **SGLang n>1 单请求 ≠ 自动 DP 分配**。一个 n=16 请求会绑定一个 replica，KV 容量不够直接 crash。要 n=1 × 16 并发让 router 分散。

6. **`<think>` prefix 不是问题**。post-trained student / teacher-consistent SFT student 都会自动 emit。我们手动加 prefix 只是把 token 从 generation 移到 prompt，等价。

7. **OPD 在 small gap setting 下增益小但真实**。4B (60%) → 8B teacher (73%) gap 13pt → OPD 提 +5.6pt（gap 的 ~43%）；paper 4B 56% → Qwen3-8B 73% gap 17pt → OPD +8.7pt（gap 的 ~51%）。增益 ∝ gap，符合直觉。
