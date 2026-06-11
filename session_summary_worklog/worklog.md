# K-Step OPD 工作日志

## 2026-04-29 (Day 1)

### 完成事项

**1. 读完 ChatGPT deep research report**
- 主题：K-Step Reward-to-Go OPD for Language Models
- 核心结论：plain fixed-k OPD 可能不够，最佳方向是 adaptive lookahead + calibrated teacher trust
- 最强正面证据：KETCHUP（k-step returns 在 RL-based KD 上有效）
- 最强负面证据：MiniLLM / Revisiting OPD（future coupling 增加 variance）
- Go/No-Go 判据：k ∈ {2,4,8} 需比 k=1 提升 ≥1-2 绝对点，且 variance/length/repetition 可控

**2. 确定代码基础**
- 主 repo：THUDM/slime（SGLang rollout + Megatron training）
- 改动切口：`slime/backends/megatron_utils/loss.py` → `apply_opd_kl_to_advantages()`
- 当前实现：`reverse_kl = student_log_probs[i] - teacher_log_probs[i]`（single-step）
- K-step 改动：对 reverse_kl 做 truncated discounted cumsum
- 备选：revisiting_opd（verl-based，OPD taxonomy 最清晰），THUNLP/OPD（top-K baseline）

**3. 写完详细研究计划** → `research-plan.md`
- Phase 0: 环境搭建 & smoke test（2-3天）
- Phase 1: Single-step OPD baseline 复现（5-7天）
- Phase 2: K-step return 实现 & fixed-k sweep（5-7天）
- Phase 3: Advanced variants — λ-return, adaptive-k, min-form（2-3周）
- Phase 4: Scale-up & cross-domain validation（2-3周）
- Phase 5: Paper writing（2-3周）
- 总计约 10-12 周

**4. 扫描计算资源**
- p5-2 ~ p5-11 全部是 p5.48xlarge（8× H100 80GB），NVMe 28T
- 两个 region：us-east-2 (Ohio) 和 ap-south-1 (Mumbai)
- 写了 `check-gpu.sh` 快速查询脚本

当前空闲机器（截至 20:38 UTC）：

| 机器 | Region | GPU | NVMe 可用 |
|------|--------|-----|-----------|
| p5-2 | us-east-2 (Ohio) | 8×H100 空闲 | 9.1T |
| p5-5 | us-east-2 (Ohio) | 8×H100 空闲 | 7.6T |
| p5-6 | us-east-2 (Ohio) | 8×H100 空闲 | 8.5T |
| p5-8 | ap-south-1 (Mumbai) | 8×H100 空闲 | 7.5T |
| p5-11 | ap-south-1 (Mumbai) | 8×H100 空闲 | 17T |

不可用：p5-4（GPU busy），p5-7（offline），p5-9/10（GPU busy）

### 下一步

- [ ] 选定一台机器开始 Phase 0
- [ ] Clone THUDM/slime，安装环境
- [ ] 下载 student/teacher 模型（Qwen2.5-3B + Qwen3-8B）
- [ ] 准备 math 训练数据（dapo-math-17k 子集）
- [ ] 跑通官方 OPD smoke test

### 文件清单

| 文件 | 用途 |
|------|------|
| `deep-research-report.md` | ChatGPT 生成的文献综述和研究建议 |
| `research-plan.md` | 详细研究计划（Phase 0-5） |
| `check-gpu.sh` | GPU 集群状态查询脚本 |
| `worklog.md` | 本文件，工作日志 |


---

## 2026-04-29 (Day 1, 续)

### Phase 0 进展：环境搭建完成

**选定机器**：p5-6（us-east-2 Ohio, p5.48xlarge, 8×H100 80GB）

**完成步骤**：

1. ✅ 在 NVMe 上建工作目录：`/opt/dlami/nvme/qzf/{k-step-opd,models,data}`
2. ✅ rsync 同步 slime 代码到 p5-6（552 files）
3. ✅ Pull 官方 Docker image：`slimerl/slime:latest`（nightly-dev-20260428a）
4. ✅ 启动容器 `k-step-opd`，挂载 NVMe 目录

**容器配置**：
```
docker run --gpus all --ipc=host --net=host --privileged \
  -v /opt/dlami/nvme/qzf/k-step-opd:/workspace/k-step-opd \
  -v /opt/dlami/nvme/qzf/models:/root/.cache/huggingface \
  -v /opt/dlami/nvme/qzf/data:/workspace/data \
  --name k-step-opd \
  -d slimerl/slime:latest sleep infinity
```

**验证结果**：
- PyTorch 2.9.1+cu129, 8 GPUs 可见
- SGLang 0.5.9
- `import slime` 成功
- 代码挂载在 `/workspace/k-step-opd/slime/`

### 下一步

- [ ] 下载 student/teacher 模型到 `/opt/dlami/nvme/qzf/models/`
- [ ] 准备 math 训练数据
- [ ] 读 OPD example 脚本，理解配置
- [ ] 跑通官方 OPD smoke test


### Phase 0 Smoke Test 结果

**✅ Pipeline 跑通！**

- 第一次尝试 TP=1 OOM（单卡放不下 student + ref model + optimizer）
- 修改为 TP=2 后成功，和官方 example 分配一致
- GPU 分配：Actor GPU 0,1 (TP=2) | Rollout GPU 2,3,4,5 (4 engines) | Teacher GPU 7

**关键指标（step 8）**：
- train/loss: 0.096
- rollout/opd_reverse_kl: 0.096（OPD KL penalty 在起作用）
- rollout/advantages: -0.096（pure OPD，advantage = -opd_kl）
- rollout/rewards: 0.0（pure distillation mode）
- grad_norm: 2.31（正常范围）
- GPU 0 内存：44.6G/79.4G（有余量）

**注意事项**：
- truncation rate = 1.0（所有 response 被截断），因为 max_response_len=2048 对 thinking mode 太短
- 正式实验需要增加到 4096-8192


### Phase 1: Single-Step OPD Baseline 启动

**配置**：
- Student: Qwen3-4B, Teacher: Qwen3-8B
- Data: dapo-math-17k (17398 prompts)
- Eval: AIME-2024 (30 problems), every 50 steps
- 200 training steps, save every 50 steps
- max_response_len: 4096, n_samples_per_prompt: 4
- Pure OPD: opd_kl_coef=1.0, kl_loss_coef=0.0, reward=0

**预计时间**：~1.5-2 小时

**脚本**：`run-phase1-baseline.sh`


**更新**：加了 MATH-500 作为主 eval（500 题，多难度覆盖），AIME-2024 作为辅助 eval。
- MATH-500：HuggingFaceH4/MATH-500，500 题，字段 problem/solution/answer
- AIME-2024：Maxwell-Jia/AIME_2024，30 题竞赛级
- Eval 配置：`--eval-prompt-data math500 ... --eval-prompt-data aime ...`
- 训练已启动并正常运行


### Phase 1 问题记录

**问题 1**：`--eval-prompt-data` 写了两次，argparse 只保留最后一个，math500 丢失
- 修复：合并为一个参数 `--eval-prompt-data math500 path aime path`

**问题 2**：eval 在 step 50 crash，`TypeError: unsupported operand type(s) for +: 'int' and 'dict'`
- 原因：OPD 的 custom reward function 返回 SGLang response dict，eval 的 reward logging 期望标量
- 修复：暂时禁用 eval，训练完后单独跑 eval

**重启**：从头跑 200 步，无 eval。预计 ~3.5 小时完成。


### 节点部署状态

| 节点 | Region | 状态 | 备注 |
|------|--------|------|------|
| p5-2 | Ohio | ✅ 就绪 | 全部完成 |
| p5-4 | Ohio | ✅ 就绪 | 全部完成（重试后成功） |
| p5-5 | Ohio | ✅ 就绪 | 全部完成 |
| p5-6 | Ohio | ✅ 就绪 | Phase 1 训练完成，待 eval |
| p5-11 | Mumbai | ✅ 就绪 | 全部完成（docker 权限已修复） |

可用节点：p5-2, p5-4, p5-5, p5-6, p5-11（共 5 台）


### Phase 1 Baseline 训练完成 ✅

**完成时间**：2026-04-30 04:01 UTC（约 3.5 小时）

**训练趋势**（200 步）：
- opd_reverse_kl: 0.096 → 0.061（下降 36%）
- truncation rate: 1.0 → 0.84
- response_length: 2048 → 4010

**Checkpoint**：`/workspace/k-step-opd/checkpoints/phase1-baseline/iter_0000199`

**下一步**：
- [ ] 单独跑 MATH-500 + AIME eval
- [ ] 开始 Phase 2：实现 k-step return


### Eval 启动 (三模型对比)

**正在运行**：Teacher (Qwen3-8B) / Student init (Qwen3-4B) / Trained Student (Phase 1 ckpt) 的 MATH-500 + AIME-2024 eval

**配置**：
- N_SAMPLES=8（pass@1 + majority vote）
- MAX_RESPONSE_LEN=16384
- TEMPERATURE=0.6
- 16 parallel workers
- 使用 slime 的 deepscaler reward function 做答案匹配

**脚本**：`run-eval-all.sh` + `eval_math.py`

**流程**：
1. ✅ Checkpoint 转换完成（torch_dist → HF）
2. ✅ Teacher (Qwen3-8B) MATH-500 + AIME eval 完成
3. ✅ Student init (Qwen3-4B) MATH-500 + AIME eval 完成
4. ✅ Trained student MATH-500 + AIME eval 完成

**结果**：

| 模型 | MATH-500 pass@1 | MATH-500 pass@any | AIME pass@1 | AIME pass@any | Avg Len (MATH) |
|------|:---:|:---:|:---:|:---:|:---:|
| Teacher (Qwen3-8B) | 91.8% | 96.0% | 56.7% | 76.7% | 12,895 |
| Student init (Qwen3-4B) | 91.6% | 95.2% | 50.0% | 63.3% | 11,412 |
| Trained student (200 steps OPD) | 89.0% | 95.6% | 40.0% | 50.0% | 14,107 |

**分析**：
- Pure OPD（reward=0）200 步后 student pass@1 轻微下降（MATH -2.6pt, AIME -10pt）
- pass@any 基本持平，说明 diversity 没变
- Response 变长（+24%），OPD 让 student 生成更长的 thinking chain
- AIME 退步更明显，可能因为纯 KL penalty 没有正确性信号
- 这是 pure distillation 的 baseline，后续 k-step 实验在此基础上对比

**注意**：maj@8 显示 0 是 majority vote 实现的 bug（答案格式匹配问题），需要修复


---

## 2026-05-04 ~ 05-05 (Day 5-6)

### Phase 1 v2: Qwen3-1.7B-Base + Pure OPD

**训练**（p5-6, 200 steps）：
- opd_reverse_kl: 1.18 → 0.86（↓27%）
- truncation: 12.5% → 9.4%
- grad_norm: 225 → 35

**Eval（MATH-500）**：

| 模型 | pass@1 (first_boxed) | pass@any | avg_len |
|------|:---:|:---:|:---:|
| Student init (1.7B-Base) | 46.4% | 78.6% | 3,644 |
| Trained (200 steps OPD) | 46.6% | 74.4% | 47,895 |
| Teacher (Qwen3-8B) | 91.8% | 96.0% | 12,895 |

**结论**：OPD 没有提升 correctness，导致严重 repetition（response 13x 变长）。Base model 不适合直接做 pure OPD 起点。

---

### Phase 1 v3: Qwen3-1.7B (post-trained) + Pure OPD

**训练**（p5-6, 200 steps，同配置）

**Eval（MATH-500）**：

| 模型 | pass@1 | first_boxed@1 | pass@any | avg_len |
|------|:---:|:---:|:---:|:---:|
| Student init (1.7B post-trained) | 86.0% | 84.2% | 94.0% | 12,179 |
| Trained (200 steps OPD) | 82.8% | 80.6% | - | 22,152 |
| Teacher (Qwen3-8B) | 91.8% | 91.8% | 96.0% | 12,895 |

**结论**：OPD 训练后 pass@1 下降 3.2pt（86% → 82.8%），response 变长（12K → 22K）。Student 起点太高（86%），和 teacher（91.8%）gap 只有 6pt，pure OPD 没有提升空间。

---

### 关键发现

1. **Student-Teacher gap 是 OPD 成功的关键**：官方 76%→94%（gap 19pt），我们 86%→82.8%（gap 6pt）
2. **Pure OPD 导致 length inflation**：两个实验都出现 response 变长
3. **Base model 不适合直接 OPD**：没有 thinking mode / 停止行为，学歪了
4. **Post-trained model 太强**：1.7B 已经 86%，接近 8B teacher 的 91.8%

### 下一步

- 需要制造更大的 student-teacher gap
- 选项：更强 teacher（32B）/ 更弱 student / 不同 domain
- 或者加 task reward 而不是 pure OPD

### 基础设施改进

- 整理了脚本结构（scripts/ + configs/）
- 建了 S3 bucket（`s3://qzf-k-step-opd-us-east-2`）做跨节点同步
- 容器加了 `--init` flag 解决 zombie 进程问题
- Eval 脚本支持三种 grading（strict/loose/first_boxed）+ 中间结果保存


---

## 2026-05-07 ~ 05-11 (Day 7-12)

### Phase 2: SFT Cold-Start + OPD

#### SFT 训练完成

**框架**: ms-swift 4.1.3 (LoRA SFT)
**配置**: Qwen3-8B-Base + OpenThoughts3 math subset + LoRA rank 128, lr 1e-3, batch 128, linear schedule, 1 epoch
**数据**: `sft_math_100k_v2.jsonl` — 100K 条，token-level 过滤（≤16384 tokens, think tags 完整, boxed 不被截断）

| 数据量 | 训练步数 | 训练时间 | 机器 | Final Loss |
|:---:|:---:|:---:|:---:|:---:|
| 30K | 217 | 3h 6m | p5-4 | ~0.87 |
| 50K | 361 | ~5h | Greenland | - |
| 100K | 722 | 10h 49m | p5-5 | 0.873 |

#### Eval 结果

**设置**: SGLang TP=8, n=1, max_tokens=8192, temperature=0.6

| 模型 | MATH-500 pass@1 | AIME-2024 pass@1 | AIME-2025 pass@1 | avg_len | Teacher gap (MATH) |
|------|:---:|:---:|:---:|:---:|:---:|
| Qwen3-8B-Base (raw) | 54.4% | 16.7% | 13.3% | 2,349 | 37pt |
| SFT 30K | 76.8% | 待测 | 待测 | 22,589 | 15pt |
| SFT 50K | 73.6% | 待测 | 待测 | 22,725 | 18pt |
| SFT 100K | 79.6% | 30.0% | 20.0% | 22,336 | 12pt |
| **Teacher (Qwen3-8B)** | **91.8%** | **56.7%** | **23.3%** | 12,895 | - |

**注意**: 30K > 50K 可能是 50K 在 Greenland 训练参数略有不同或随机性。

#### OPD 训练

**配置**: SFT 100K checkpoint → OPD with Qwen3-8B teacher
- Actor TP=4 (GPU 0-3), Rollout 2 engines (GPU 4,5), Teacher TP=2 (GPU 6,7)
- dapo-math-17k, n_samples=4, global_batch=32, lr=5e-7, max_response_len=8192
- ~4 min/step

**p5-2 结果** (被 boshih 进程 kill 在 step 8):
- opd_reverse_kl: 0.22 → 0.21 (缓慢下降)
- truncated_ratio: 1.0 (100% 截断，8192 tokens 不够)
- repetition_frac: 0.09 → 0.22 (上升中)

**问题**:
1. p5-2 被 boshih 的 MegatronTrainRayActor 抢占 GPU，训练被 OOM killer 杀掉
2. 100% truncation（SFT 模型 response 太长）
3. Actor TP=2 OOM（8B model + ref + optimizer 每卡 77GB），改 TP=4 解决

**已迁移到 Greenland** 跑 OPD。

#### Greenland 部署

**镜像**:
- `654654486179.dkr.ecr.us-east-2.amazonaws.com/k-step-opd-sft:greenland-v1` — SFT 训练
- `654654486179.dkr.ecr.us-east-2.amazonaws.com/k-step-opd-slime:greenland-v1` — OPD 训练

**S3** (`delphi-greenland-res-alpha/qzf/`):
- `models/Qwen3-8B-Base/`, `models/Qwen3-8B/`
- `data/sft_math_30k_v2.jsonl`, `sft_math_50k_v2.jsonl`, `sft_math_100k_v2.jsonl`, `dapo-math-17k.jsonl`
- `checkpoints/sft-100k-torch_dist/`, `checkpoints/sft-100k-merged/`
- `code/k-step-opd.tar.gz`, `code/bootstrap_sft.sh`, `code/bootstrap_slime.sh`

**踩坑记录**:
1. ECR 需要给 `644924934147` 加 resource policy 才能让 Greenland 拉镜像
2. S3 bucket 需要给 `644924934147` 加 bucket policy
3. 镜像需要预装 boto3（slime 基础镜像没有）
4. ms-swift 4.2.0 有 datasets.features.Json 兼容问题，固定 4.1.3
5. torchao 版本冲突：ms-swift 需要 ≥0.16.0，sglang 需要 ==0.9.0

#### Eval 注意事项

- SFT 模型需要在 prompt 末尾加 `<think>\n` 前缀引导 thinking mode
- Base model 不需要（它不生成 think tags）
- SGLang server 容易 OOM：n=8 + workers=16 会 crash，n=1 + workers=8 稳定
- 容器长时间运行会积累 zombie 进程，需要定期 `docker restart`

### 下一步

- [ ] 等 Greenland OPD job 完成（~13h）
- [ ] 跑 30K/50K AIME eval（p5-5 上进行中）
- [ ] OPD 完成后 eval checkpoint
- [ ] 如果 OPD 有效 → 实现 k-step return
- [ ] 考虑用 Qwen3-32B 做 teacher（gap 更大）


---

## 2026-05-11 (Day 12)

### Cumulative OPD (Reward-to-Go) 实现

**动机**：当前 OPD 的 advantage 只用 instant per-token KL penalty。讨论后决定实现 cumulative 版本（reward-to-go），让每个 token 的 OPD penalty 考虑后续 token 的 KL 偏离。

**理论基础**：
- 当前 token 用 sampled KL（`log π_S(a_t) - log π_T(a_t)`）是正确的
- 当前 token 的 full KL 放 advantage 里没意义（detach 后梯度方向不变）
- 后续 token 的 KL 作为 future credit 是合理的（Rao-Blackwellization of sampled future KL）
- 主要风险：long horizon teacher signal degradation（Rethinking OPD 的发现）

**实现**：

修改了两个文件：
1. `slime/slime/utils/arguments.py` — 新增参数：
   - `--opd-cumulative`：启用 reward-to-go 模式
   - `--opd-gamma`：discount factor（默认 1.0）
   - `--opd-horizon`：最大 lookahead 步数（默认 -1 = full sequence）

2. `slime/slime/backends/megatron_utils/loss.py` — `apply_opd_kl_to_advantages()` 新增分支：
   - `gamma=1.0` + full horizon：`flip → cumsum → flip`（向量化 suffix sum）
   - `gamma<1.0` + full horizon：从末尾迭代 discounted suffix sum
   - truncated horizon：每个 token 只看未来 K 步的 weighted sum

**公式**：
```
A_t = A_t^base - λ * Σ_{d=0}^{K-1} γ^d * (log π_S(a_{t+d}) - log π_T(a_{t+d}))
```

**使用方式**：
```bash
--use-opd --opd-type sglang --opd-kl-coef 0.1 --opd-cumulative --opd-gamma 0.99 --opd-horizon 8
```

**注意**：cumulative 模式下 penalty magnitude 比 instant 大很多，`opd_kl_coef` 需要调小。

### 部署

已 patch 到 p5-2 和 p5-5：
```bash
rsync slime/ → p5-{2,5}:/opt/dlami/nvme/qzf/k-step-opd/slime/
docker exec k-step-opd cp ... /root/slime/slime/backends/megatron_utils/loss.py
docker exec k-step-opd cp ... /root/slime/slime/utils/arguments.py
```

### 后续发现：SGLang 原生支持 top-k logprobs

SGLang 的 `top_logprobs_num` 参数可以直接返回 top-K token 的 logprobs（已验证 K=50 可用）。
这意味着 future top-k KL 版本（ChatGPT 建议的方案）也可以在 sglang 模式下实现，不需要切 megatron 模式。

### 下一步

- [ ] 修改 `train-opd.sh` 支持 cumulative 参数的环境变量
- [ ] 设计实验：instant vs cumulative (K=4,8,16) vs cumulative+discount(γ=0.95,0.99)
- [ ] 跑 baseline instant OPD 对比
- [ ] 考虑实现 future top-k KL 版本（auxiliary distillation loss）


---

## 2026-05-12 (Day 13)

### Cumulative OPD 实现 & 部署

见上方 2026-05-11 条目。

### Eval 修正：max_tokens 问题

**发现**：之前所有 eval 用 `max_tokens=8192`，严重低估了模型能力。
- Qwen3-8B 官方 AIME 2024 = 79.4%，我们之前测的 = 56.7%
- 原因：Qwen3-8B 需要 32K+ tokens 的 thinking space

**修正后 Eval 结果**（max_tokens=32768/30000, n=1, temperature=0.6）：

| 模型 | AIME-2024 | AIME-2025 | Avg Len (chars) |
|------|:---:|:---:|:---:|
| Qwen3-8B (teacher) | **73.3%** | **70.0%** | ~13K |
| SFT-100K (student, 8B-Base) | **50.0%** | **40.0%** | ~84K |
| Gap | 23pt | 30pt | — |

**关键发现**：
- Teacher-Student gap 实际是 23-30pt（之前以为只有 12pt）
- SFT 模型 response 极长（84K chars ≈ 24K tokens），几乎每题都生成到 max_tokens
- SFT 模型 thinking 效率极低，不会停止

**SFT checkpoint 的 max_position_embeddings 问题**：
- Qwen3-8B-Base 原始值 = 32768
- Qwen3-8B (post-trained) = 40960（YaRN 扩展）
- SFT checkpoint 继承了 Base 的 32768，所以 max_tokens 不能超过 ~30000

### OPD 训练曲线对比

绘制了 Instant vs Cumulative OPD 对比图（`opd_comparison_curves.png`）：

| 指标 | Instant (kl_coef=1.0) | Cumulative (kl_coef=0.05, γ=0.99, K=8) |
|------|:---:|:---:|
| OPD Reverse KL | 0.216 → 0.079 (↓63%) | 0.210 → 0.036 (↓83%) |
| Grad Norm | 3.0 → 1.4 (不稳定) | 0.16 → 0.16 (稳定) |
| KL from ref | 0 → 0.209 | 0 → 0.241 |

Cumulative OPD 训练更稳定（grad norm 低 10x），KL 收敛更好。

### Lightning OPD 论文发现

**论文**：[Lightning OPD](https://arxiv.org/abs/2604.13010) (NVIDIA, 2026.04)

**核心洞察 — Teacher Consistency**：
- SFT 数据必须由 OPD teacher 自己生成
- 违反此条件会引入不可消除的 gradient bias
- 这解释了为什么我们的 OPD 效果差（SFT 用 OpenThoughts3 数据，OPD teacher 是 Qwen3-8B）

**Lightning OPD 结果**：
- Qwen3-4B-Base → SFT → OPD：20 GPU hours
- Qwen3-8B-Base → SFT → OPD：AIME 2024 69.9%，30 GPU hours
- Qwen3-30B-A3B → OPD：AIME 2024 71.0%，单 node 8×H100

**关键差异 vs 我们的实验**：

| | Lightning OPD | 我们 |
|---|---|---|
| SFT 数据来源 | Teacher 自己生成 | OpenThoughts3 (第三方) |
| SFT 方式 | Full fine-tuning | LoRA rank 128 |
| Teacher consistency | ✅ | ❌ |
| OPD 结果 | 69.9% AIME | 退步 |

### 下一步：按 Lightning OPD 重做

**正在进行**：
1. ✅ p5-4：用 Qwen3-8B 生成 100K SFT data（OpenThoughts3 prompts + teacher response）
   - SGLang TP=8, temperature=0.7, top_p=0.9, max_tokens=16384
   - 预计 10-15 小时
2. ✅ p5-5：Qwen3-4B-Base + SFT 100K（OpenThoughts3 数据，LoRA）
   - 用于对比实验
   - 在 `k-step-opd-sft` 容器（PyTorch 2.6）中运行

**计划**：
- [ ] Teacher data 生成完成后，用它做 SFT（teacher consistent）
- [ ] 然后用同一个 Qwen3-8B 做 OPD
- [ ] 对比：teacher-consistent SFT + OPD vs 当前 OpenThoughts3 SFT + OPD

### 基础设施笔记

- slime 容器（`slimerl/slime:latest`）不能跑 ms-swift（torchao 版本冲突）
- 解决方案：单独的 SFT 容器（`pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel` + ms-swift）
- p5-4 root filesystem 满（939G/969G），主要是其他用户 /home（820G）+ Docker overlay（95G）
- 清理了 logs + 旧 image 后有 35G free


---

## 2026-05-13 (Day 14, 续)

### OPD Eval 修正结果 ✅

用 `max_tokens=30000` 重新 eval 两个 OPD 模型：

| 模型 | AIME-2024 pass@1 | 变化 vs SFT baseline |
|------|:---:|:---:|
| Qwen3-8B (teacher) | 73.3% | — |
| **OPD Instant** (kl_coef=1.0, 200 steps) | **60.0%** | **+10pt** ✅ |
| **OPD Cumulative** (kl_coef=0.05, γ=0.99, K=8) | **53.3%** | **+3pt** |
| SFT-100K (baseline) | 50.0% | — |

**结论**：OPD 是有效的！Instant OPD 提升 10pt，接近 teacher 的 73%。之前以为 OPD 退步是因为 eval 的 max_tokens=8192 太短。

### Qwen3-4B Eval 结果

| 模型 | AIME-2024 |
|------|:---:|
| Qwen3-4B (post-trained) | 73.3% |
| Qwen3-4B-Base | 0% (无 instruction following) |

4B post-trained 和 8B 一样强（73.3%），不适合做 student。

### Qwen3-4B-Base SFT 100K 完成 ✅

- 机器：p5-5 (`k-step-opd-sft` 容器, PyTorch 2.6)
- 配置：LoRA rank 128, lr 1e-3, 722 steps, 9h 41m
- Final loss: 0.844
- Checkpoint: `/opt/dlami/nvme/qzf/models/sft-qwen3-4b-base-lora/`

### Teacher Data Generation 进行中

- 机器：qzf-dev (4×L40S)
- 任务：用 Qwen3-8B 生成 100K SFT data（OpenThoughts3 prompts）
- 进度：742/100000
- 速度：~3000 条/hr
- 预计完成：~33h

### 基础设施笔记

- p5 机器之间没有 SSH key，需要通过本地中转或传 key
- `max_position_embeddings` 问题：Base model = 32768, post-trained = 40960
  - SGLang 的 `max-total-tokens` 不能超过模型的 `max_position_embeddings`
  - eval 时 `max_tokens` 要留 prompt 空间（用 30000 而不是 32768）
- Docker overlay filesystem 问题：`docker exec cp` 写入的文件对 Ray worker 不可见
- ms-swift 4.1.3 需要 PyTorch ≥2.6（FSDPModule），torchao ≥0.16.0
  - slime 容器（torchao 0.9.0）不能跑 ms-swift
  - 解决：单独的 SFT 容器 `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`


---

## 2026-05-14 (Day 15)

### 4B LoRA SFT 全面失败

**发现之前 "Full FT" 训练无效**：p5-4 上 ms-swift 的 `--load_from_cache_file true` 复用了一个只有 10 条数据的旧 cache，导致所有 full FT checkpoint 实际只训了 10 个样本。

**LoRA 三次尝试全部 0%**：

| 版本 | 配置 | Loss | AIME-2024 |
|------|------|------|-----------|
| v1 | r=128, α=256, lr=1e-3, ms-swift | 0.844 | 0% |
| v3 | r=128, α=128, lr=2e-4, liger, ms-swift | 0.832 | 0% |
| v5 | r=128, α=128, lr=3e-5, +lm_head, ms-swift | 进行中 | — |

**症状**：开头泰语乱码、从不生成 `</think>`/`\boxed{}`、末尾 degeneration。训练集 prompt 也不行。

**根因**：4B-Base 的 `tie_word_embeddings=True`（lm_head 和 embed_tokens 共享），LoRA 不修改这些层 → 输出 token 分布无法改变。8B-Base 的 `tie_word_embeddings=False` 所以 LoRA 能工作。

### Teacher-Consistent Data Generation 完成

用 Qwen3-8B 8-replica TP=1 async 生成：
- p5-4: 0-40K ✅ 完成
- p5-9: 40K-80K ✅ 完成
- p5-4: 80K-100K 🔄 进行中（~65%）

数据质量：79% 有 `</think>`，80% 有 `\boxed{}`，63% 答案与 QwQ-32B 一致（sympy 验证）。

### 多节点训练尝试

**ms-swift 多节点失败**：packing + DDP 导致 deadlock（两节点 packed dataset 不一致，DistributedSampler 不对齐）。ms-swift 4.1.3 没有 `--packing_cache` 参数。

**NCCL 测试通过**：16 ranks all-reduce 正常，1.6 GB/s TCP 带宽。

**LlamaFactory 多节点**：正在设置中（p5-5 master + p5-3 worker）。

### 速度对比

| 方案 | Step Time | 预计总时间 |
|------|-----------|-----------|
| ms-swift LoRA 单节点 (bs=8, accum=2) | 41s | 9h (800 steps) |
| LlamaFactory LoRA 单节点 (bs=8, accum=2) | 55s | 12h |
| ms-swift Full FT ZeRO-1 (bs=1, accum=32) | 83s | 69h (3000 steps) |

### 下一步

1. 跑通 LlamaFactory 多节点 LoRA + lm_head
2. 如果 LoRA + lm_head 还是不行 → 放弃 LoRA，用 LlamaFactory full FT（Lightning OPD 的方案）
3. 完成 teacher data generation（剩余 20K）
4. Teacher-consistent SFT + OPD

### 基础设施笔记

- p5-4 root fs 满（其他用户 home 691G），需要 `TMPDIR` 指向 NVMe
- p5-5 和 p5-3 共享 EFS：`/mnt/wutianyi-efs`
- 多节点需要相同容器镜像（NCCL 版本必须一致）
- ms-swift 4.1.3 的 packing 不支持多节点共享 cache


---

## 2026-05-15 (Day 16)

### ms-swift 4.2 升级 + 多节点训练成功 ✅

**升级 ms-swift 4.1.3 → 4.2.0**（p5-3, p5-4, p5-5）：
- v4.2 的 `PackingDataset` 修复了多节点 packing deadlock：master 做 packing 后 `dist.broadcast_object_list()` 同步到所有 rank
- 不再需要 `packing_cache` 或共享文件系统

**多节点 Smoke Test 结果**（p5-4 master + p5-3 worker, 16 GPU）：

| 配置 | Step Time | Memory/GPU | 状态 |
|------|:---------:|:----------:|:----:|
| DDP + DeepSpeed ZeRO-2, bs=8 | 50s | 49 GiB | ✅ |
| **DDP (no DS), bs=8** | **20s** | **63 GiB** | **✅ 最优** |
| DDP, bs=10 | 25s | 67.6 GiB | ✅ |
| DDP, bs=12 | OOM | — | ❌ |
| DDP, no grad_ckpt, bs=8 | OOM | — | ❌ |
| DDP, use_flash_ckpt | 需要 dlrover | — | ❌ 不适用 |

**最终配置**：DDP 多节点, bs=8, accum=1, gradient_checkpointing=True, 无 DeepSpeed

### 4B LoRA v5 训练完成 ✅

**配置**：
- 模型：Qwen3-4B-Base + LoRA r=128, α=128 + `modules_to_save lm_head`
- 数据：sft_math_100k_v2.jsonl（100K 条）
- 多节点：p5-4 (master) + p5-3 (worker), 16×H100
- ms-swift 4.2.0, DDP, packing=True, max_length=16384
- lr=3e-5, cosine, warmup=0.1, 800 steps (~1.1 epoch)
- 17s/step, 总时间 3h 47m

**训练结果**：
- Loss: 1.097 → 0.877 (smoothed 0.872)
- Grad norm: 1.99 → 0.10
- Memory: 63-67 GiB/GPU
- Checkpoint: `/root/.cache/huggingface/sft-qwen3-4b-lora-v5-multinode/v3-20260515-024739/checkpoint-800`

### 抽查验证 — 模型学会了 thinking 格式 ✅

Merge adapter → SGLang serve → 10 个问题测试：

| 结果 | 数量 | 说明 |
|------|:----:|------|
| ✅ 正确完成 (`</think>` + `\boxed{}`) | 7/10 | 答案全部正确 |
| ❌ 超长未完成 | 3/10 | 2048 tokens 不够，正式 eval 用 16K 没问题 |

**对比之前 LoRA v1/v3（0%）**：`modules_to_save lm_head` 彻底解决了 `tie_word_embeddings=True` 导致的输出分布无法改变问题。

### Teacher Data Generation 完成 ✅

p5-4 上 80K-100K range 完成：
- 8 shards × 2500 = 20000 条
- 总计：0-40K (p5-4) + 40K-80K (p5-9) + 80K-100K (p5-4) = **100K 条**
- 下一步：合并 + 过滤（保留有 `</think>` + `\boxed{}` 的，~80K 可用）

### 踩坑记录

**p5-4 root fs 满（5G free）**：
- 100K 数据 tokenize 的 arrow cache 默认写到 Docker overlay（root fs）
- 解决：设置 `TMPDIR`、`HF_DATASETS_CACHE`、`HF_HOME`、`XDG_CACHE_HOME` 全部指向 NVMe 挂载路径

**transformers 版本不兼容**：
- ms-swift 容器（transformers 5.6.2）保存 tokenizer 时把 `additional_special_tokens` 改成了 `extra_special_tokens`（list 格式）
- slime 容器（transformers 4.57.1）加载时期望 dict → `AttributeError: 'list' object has no attribute 'keys'`
- 解决：用 base model 的原始 tokenizer 文件覆盖 merged model 目录

**p5-5 被 ccrchen 占用**：GPU 1 跑着 Qwen3-14B vLLM server，改用 p5-4+p5-3 组多节点

**端口冲突**：之前 p5-5 作为 master 时的 torchrun elastic agent 残留在 p5-4 上占用 29501 端口，需要手动 kill

### 下一步

1. 正式 AIME-2024 eval（SGLang, max_tokens=16384, n=1）
2. 合并 teacher data（100K），过滤后做 teacher-consistent SFT
3. Teacher-consistent SFT + OPD

### 文件位置

| 文件 | 用途 |
|------|------|
| `scripts/run-sft-lora-v5-multinode.sh` | 多节点训练脚本（最终版） |
| `scripts/upgrade-msswift-multinode.sh` | ms-swift 升级脚本（已废弃） |
| `sft_lora_v5_multinode_curves.png` | 训练曲线图 |
| Checkpoint (p5-4) | `/root/.cache/huggingface/sft-qwen3-4b-lora-v5-multinode/v3-20260515-024739/checkpoint-800` |
| Merged model (p5-4) | `/root/.cache/huggingface/sft-qwen3-4b-lora-v5-merged` |


---

## 2026-05-16 (Day 17)

### 4B Full FT v2 训练完成 + Eval ✅

**配置**：
- 模型：Qwen3-4B-Base, Full fine-tuning
- 数据：teacher_sft_filtered.jsonl（79,341 条，只保留有 `</think>` + `\boxed{}` 的完整样本）
- 多节点：p5-3 (master) + p5-4 (worker), 16×H100
- ms-swift 4.2.0, ZeRO-1, packing=True, max_length=16384
- lr=8e-5, cosine, warmup=0.1, 3 epochs
- bs=8, accum=2, global batch=256
- 36s/step, 759 steps, 7h 35m

**训练结果**：
- Loss: 0.539 → 0.224 (final avg 0.246)
- Grad norm: 3.7 → 0.20
- Memory: 60-62 GiB/GPU

**Eval 结果**（SGLang DP=8, max_tokens=30000, n=1, temperature=0.6）：

| 模型 | AIME-2024 | AIME-2025 |
|------|:---------:|:---------:|
| Qwen3-8B (teacher) | 73.3% | 70.0% |
| 8B-Base LoRA SFT | 50.0% | 40.0% |
| **4B-Base Full FT v2 (3ep, filtered)** | **50.0%** | **30.0%** |
| 4B-Base Full FT v1 (1ep, unfiltered) | ~0% | — |
| 4B-Base LoRA v5 (+ lm_head) | 0% | — |

**关键改进 vs v1**：
1. Filtered data（去掉 20% 截断样本）→ 模型学会正确结束 thinking
2. 3 epochs（vs 1 epoch）→ 充分训练

### Checkpoint 位置

| 文件 | 位置 |
|------|------|
| Full FT v2 checkpoint | p5-3: `/root/.cache/huggingface/sft-qwen3-4b-full-teacher-v2/v9-20260515-233350/checkpoint-759` |
| Copy on p5-4 | p5-4: `/root/.cache/huggingface/sft-qwen3-4b-full-teacher-v2-ckpt759/` |
| Training curves | `sft_full_4b_teacher_v2_curves.png` |

### 基础设施修复

**p5-4 Docker 迁移到 NVMe**：
- `/var/lib/docker` → `/opt/dlami/nvme/docker`（symlink）
- Root fs 从 0 → 88G free
- 彻底解决了 p5-4 磁盘满的问题

**p5-4 容器重建**：
- 之前删 `.dist-info` 破坏了包依赖（flash-attn, numpy, regex 都坏了）
- 重建 `k-step-opd-sft` 容器，重新安装所有包

**多节点数据一致性 bug**：
- 两节点的 smoke data 内容不一致 → PackingDataset broadcast 后 worker IndexError
- 根因：p5-4 的 5K smoke 从 unfiltered data 取，p5-3 从 filtered data 取
- 修复：确保两节点数据 md5 一致

### 8B Rollout Collection 进行中

- 机器：p5-9（Mumbai）
- 任务：用 8B SFT merged model 在 DAPO-Math-17k 上生成 rollouts
- 配置：8 sglang engines (TP=1), 24 concurrent/engine, max_tokens=8192
- 用途：Lightning OPD 的 Step 3

### Lightning OPD Pipeline 研究完成

完整 pipeline：
1. ✅ SFT（已完成，4B Full FT v2 = 50% AIME）
2. 🔄 Collect rollouts（p5-9 进行中）
3. Precompute teacher logprobs
4. Lightning OPD training（8 GPU 全给 actor，不需要 teacher server）

关键发现：
- Lightning OPD 用 offline precomputed teacher logprobs，训练时不需要 teacher server
- 所有 8 GPU 给 actor → 3.6-4x 加速
- advantage = teacher_logprob - student_logprob（每步 forward 重算 student）
- 预计 OPD 后 4B 能到 60%+ AIME

### 下一步

1. 等 p5-9 rollout collection 完成
2. Precompute teacher logprobs（启动 Qwen3-8B server）
3. 跑 Lightning OPD（~2-3h on 8×H100）
4. Eval OPD checkpoint


---

## 2026-05-17 (Day 18)

### 4B LoRA v8 训练完成 + Eval ✅

**配置**：
- 模型：Qwen3-4B-Base, LoRA r=128, α=256, **无 lm_head**
- 数据：sft_math_100k_v2.jsonl（100K 条，OpenThoughts3，非 teacher-consistent）
- 多节点：p5-4 (master) + p5-3 (worker), 16×H100, DDP
- ms-swift 4.2.0, lr=5e-4, cosine, warmup=0.05, 2 epochs
- bs=8, accum=1, global batch=128, max_length=16384, packing
- 18s/step, 1444 steps, 6h 42m

**训练结果**：
- Loss: 1.097 → 0.786 (min 0.772 @ step 1250)
- Grad norm: 0.647 → 0.034
- Training curves: `sft_lora_v8_curves.png`

**手动测试**（SGLang TP=1, max_tokens=2048-8192, temperature=0.6, 加 `<think>\n` 前缀）：

| 测试 | `</think>` | `\boxed{}` | 表现 |
|------|:---:|:---:|------|
| 简单题 (2+3) | ❌ | ❌ | 开头泰语乱码，答对但极其啰嗦，从不停止 |
| 中等题 (x²-5x+6) | ❌ | ✅ | 答对，有 boxed，但没 `</think>`，答完后重复整个解答 |
| 难题 (number theory) | ❌ | ❌ | 末尾退化成 `the the the...` 乱码 |

**AIME-2024 Eval**（SGLang DP=8, max_tokens=30000, n=1, temperature=0.6）：

| 模型 | AIME-2024 | Avg Len |
|------|:---------:|:-------:|
| **4B LoRA v8 (no lm_head, α=256, lr=5e-4)** | **0%** (0/30) | 30,777 chars |

**结论**：和 v1/v3 一样，全部 0%。30 题全部生成到 max_tokens 不停止。高 lr + 高 α + 2 epochs 都没有帮助——`tie_word_embeddings=True` 是根本问题，不加 lm_head 的 LoRA 无法改变输出分布。

**4B LoRA 实验汇总**：

| 版本 | lm_head | α | lr | Data | AIME-2024 |
|------|:-------:|:---:|:---:|------|:---------:|
| v1 | ❌ | 256 | 1e-3 | OpenThoughts3 100K | 0% |
| v3 | ❌ | 128 | 2e-4 | OpenThoughts3 100K | 0% |
| v5 | ✅ | 128 | 3e-5 | OpenThoughts3 100K | 0% (简单题 7/10) |
| v8 | ❌ | 256 | 5e-4 | OpenThoughts3 100K | 0% |
| Full FT v2 | ✅ (全参) | — | 8e-5 | teacher_sft_filtered 79K | **50%** |

### LoRA v9 准备

基于 Tinker (Thinking Machines Lab) 的 Qwen3-4B sweep 实验数据，准备 v9：
- α=32（Tinker 标准，scaling=0.25，vs v8 的 2.0）
- lr=3e-4（Tinker 4B 实测最优）
- 数据改用 teacher_sft_filtered.jsonl（teacher-consistent）
- 其他参数与 v8 一致，不加 lm_head（测试纯 Tinker recipe 效果）
- 单节点 p5-4，8×H100，accum=2 保持 global batch=128

脚本：`scripts/run-sft-lora-v9.sh`


## 2026-05-18 Session

### OPD Cumulative v2 实验 (p5-2)

**配置变更（vs v1）：**
- `opd_kl_coef`: 0.05 → **1.0**（和 Instant 对齐）
- `opd_horizon` (K): 8 → **2**
- `max_response_len`: 8192 → **16384**
- `lr`: 5e-7 → **1e-6**
- `temperature`: 0.6 → **1.0**
- `global_batch_size`: 32 → **64**
- `rollout_batch_size`: 8 → **16**
- GPU 分配: 5 rollout + 2 actor (TP=2) + 1 teacher (TP=1)
- **归一化 cumulative KL**：`cumulative_kl[t] /= actual_k`（新增）

**代码修改：**
- `slime/slime/backends/megatron_utils/loss.py`：truncated horizon 模式下 cumulative KL 除以 actual_k 归一化
- `configs/opd-cumulative.env`：更新参数
- `scripts/train-opd.sh`：默认 rollout_batch_size=16

**环境问题解决：**
- p5-2 的 `slimerl/slime:latest` image 比 p5-5 新（transformers 5.3.0 + huggingface_hub 1.9.2）
- huggingface_hub 1.9.2 对绝对路径做 repo ID 验证导致 `AutoConfig.from_pretrained("/path/...")` 报错
- 解决：降级 `huggingface_hub==1.3.0`（不做路径验证）+ 保持 `transformers==5.3.0`（sglang 需要）
- Ray port 冲突：`slime-bh` container 里的旧 Ray cluster 占了 8265 端口，`ray job submit` 连到了错误的 cluster（没有 volume mounts）
- 解决：kill 掉 `slime-bh` 的旧 Ray

**状态：** p5-2 上 OPD Cumulative v2 训练已启动

---

### Teacher Data Generation (p5-5 + p5-2)

**目的：** 用 Qwen3-8B teacher 在新 100K prompts 上生成 SFT 数据

**进度：**
- p5-5: shard 0 (50K), ~39K/50K done, rate ~2138/hr
- p5-2: shard 1 (50K), **完成** ✅

**质量：**
- p5-5: 71% 有 `</think>` + `\boxed{}`
- p5-2: 78% 有 `</think>` + `\boxed{}`
- 生成完后需过滤

---

### SFT 数据准备

- `sft_math_200k_v2.jsonl`：200,039 samples（100K 原有 + 100K 新增）
- 新增 100K 条件：math domain + `</think>` + ≤14000 words + 不和原 100K 的 (prompt, response) 重复
- 全部有完整 `<think>...</think>` 对

---

### Lightning-OPD 代码分析

**关键发现：**
- SFT: lr=8e-5, 3000 steps, 300K prompts, LlamaFactory, packing, 不过滤数据
- Rollout: `llm.chat()` + vLLM offline, temp=0.7, top_p=0.9, max_tokens=16384
- OPD training: lr=2e-6, batch=256, max_response_len=4096, 3000 rollouts
- Teacher consistency: SFT teacher 和 OPD teacher 必须是同一个 model
- `--advantage-estimator on_policy_distillation`（slime 内置）
- reward=0（纯 distillation）+ `--include-verifiable-reward`（加 task reward）

---

### 4B Full FT 失败分析

- checkpoint-253 完全退化（repetition → garbage）
- 原因：lr=8e-5 对 Full FT 可能不是问题（Lightning-OPD 也用 8e-5），但数据未过滤（包含截断样本）
- SeaFill/Qwen3-4B-SFT 用 49K filtered data 达到 AIME 20.8%
- QED-Nano-SFT: lr=3e-5, 4.3K unique problems, 620 steps

---

### 8B SFT Model 测试

- 不加 `<think>\n` prefix：model 不进入 thinking mode（0% 有 `</think>`）
- 加 `<think>\n` prefix：model 能 reasoning 但不输出 `</think>`（length explosion，78 个 `\boxed{}` 散布在 45K chars 中）
- 结论：model 学会了 reasoning 和给答案，但没学好终止信号


---

### 4B LoRA v7 (lr=1e-3, α=256, +lm_head) — 爆炸 💥

**配置**：对齐 8B recipe（lr=1e-3, α=256, linear schedule），加 lm_head
**结果**：Step 170 loss explosion（0.88 → 8.5），warmup 结束后 lm_head 的 full lr 更新太激进
**原因**：lm_head (151936×2560, 389M params) 用 full lr=1e-3 训练，4B hidden_size=2560 比 8B 的 4096 更敏感

### 4B LoRA v8 (lr=5e-4, α=256, 无 lm_head) — 训练稳定但 AIME 0%

**配置**：去掉 lm_head，lr 减半到 5e-4
**训练**：1444 steps, 6h 42m, loss 1.097→0.786, grad_norm 极稳定 (0.03)
**Eval**：AIME-2024 0%（30 题全部生成到 max_tokens 不停止）
**结论**：确认 `tie_word_embeddings=True` 下无 lm_head 的 LoRA 无法改变输出分布

### 4B LoRA v9 (lr=3e-4, α=32, 无 lm_head, teacher data) — 训练稳定但 AIME 0%

**配置**：Tinker recipe（α=32, scaling=0.25），teacher-consistent data (79K filtered)
**训练**：1010 steps, 8h 17m, loss 0.538→0.275（接近 Full FT v2 的 0.246）
**Eval**：AIME-2024 0%（同样不停止）
**手动测试**：
- 推理内容正确（能解题），但永远不生成 `</think>`
- 加 `\boxed{}` 指令到 prompt 后偶尔能生成 boxed（因为是普通 token 序列）
- `</think>` 是 special token (id 151668)，需要 lm_head 权重变化才能被选中
**结论**：teacher-consistent data + 低 α 也不能绕过 lm_head 问题。4B LoRA 不加 lm_head 就是不行。

### 4B LoRA 实验最终汇总

| 版本 | lm_head | α | lr | Data | 训练稳定 | AIME-2024 |
|------|:-------:|:---:|:---:|------|:---:|:---------:|
| v1 | ❌ | 256 | 1e-3 | OT3 100K | ✅ | 0% |
| v3 | ❌ | 128 | 2e-4 | OT3 100K | ✅ | 0% |
| v5 | ✅ | 128 | 3e-5 | OT3 100K | ✅ | 0% (简单题 7/10) |
| v7 | ✅ | 256 | 1e-3 | OT3 100K | 💥 爆炸 | N/A |
| v8 | ❌ | 256 | 5e-4 | OT3 100K | ✅ | 0% |
| v9 | ❌ | 32 | 3e-4 | teacher 79K | ✅ | 0% |
| **Full FT v2** | ✅ (全参) | — | 8e-5 | teacher 79K | ✅ | **50%** |

**根因**：Qwen3-4B-Base 的 `tie_word_embeddings=True` 使 lm_head 和 embed_tokens 共享权重。LoRA 不修改这些层 → 输出 token 分布不变 → 无法学会生成 special tokens (`</think>`)。

### Tinker Cookbook 参数分析

查看了 `tinker_cookbook.recipes.distillation.off_policy_reasoning` 的完整默认参数：
- lr=1e-3, linear schedule（无 warmup），batch=128, 1 epoch, max_length=16384
- lora_alpha 由 tinker 服务端决定（不可配置）
- adam_beta2=0.95（比 HF 默认 0.999 更激进）
- 数据量 384K（~3000 steps），linear 衰减快速降 lr
- Tinker 能用 1e-3 不爆的原因：无 warmup + 3000 steps linear 衰减 + 可能 alpha=rank

### p5-3 被 zhesu 占用

`agentic_rl_0508` 容器占满 8 GPU：
- **实验**：Qwen3-30B MoE (A3B) GRPO 训练
- **任务**：Amazon 客服 agentic tool-calling（intent/nudge/followup/item_picker）
- **框架**：verl, Megatron backend, EP=8, vLLM rollout TP=4
- **数据**：2599 条客服对话，单轮 response（max_response=512）
- **进度**：global_step 120，还在跑

### Teacher Data Generation 完成 + 合并

**Shard 0** (p5-5): 50K 条, 70.4% 有 `</think>` + `\boxed{}`
**Shard 1** (p5-2): 50K 条, 75.9% 有 `</think>` + `\boxed{}`

**合并**（不过滤，Lightning-OPD 风格）：
```
teacher_sft_filtered.jsonl (79K, 之前已过滤) + shard0 (50K raw) + shard1 (50K raw)
= teacher_sft_179k_merged.jsonl (179,341 条)
```
- 位置：p5-4 `/workspace/data/teacher_sft_179k_merged.jsonl`
- 约 85% 有完整 `</think>`，15% 被 max_tokens 截断
- 决定不过滤：Lightning-OPD 也不过滤，靠 packing cutoff + OPD reward 处理

### 下一步

1. 用 179K merged data 跑 4B Full FT（对齐 Lightning-OPD：lr=8e-5, 3 epochs, ZeRO-1）
2. 监控 OPD Cumulative v2 训练（p5-2）
3. Eval OPD v2 checkpoint


---

## 2026-05-19~20 (Day 19-20)

### 8B Full FT 179K (Greenland) — AIME 63.3% ✅

**训练**：Greenland p5.48xlarge, ZeRO-1, 487 steps (1 epoch), 179K teacher data
**Eval**（p5-4, SGLang DP=8, max_tokens=30000, n=1, temp=0.6）：

| Benchmark | pass@1 |
|-----------|:------:|
| AIME-2024 | **63.3%** |
| AIME-2025 | **50.0%** |

对比之前 8B LoRA SFT (100K OT3): AIME-2024 50% → 63.3% (+13pt)

### 4B Full FT 179K — 1ep/2ep/3ep 对比

| 实验 | Steps | Final Loss | AIME-2024 | 备注 |
|------|-------|-----------|-----------|------|
| 4B Full FT v2 (79K filtered × 3ep) | 759 | 0.201 | **50%** | 基准 |
| 4B Full FT 179K × 1ep | 487 | 0.264 | 0% | 退化 |
| 4B Full FT 179K × 2ep | 974 | 0.213 | 3.3% | 略有改善 |
| 4B Full FT 179K × 3ep | 1461 | TBD | TBD | 🔄 运行中 (~20h) |

**关键发现**：
- Loss 接近（0.213 vs 0.201）但 AIME 差距巨大（3.3% vs 50%）
- 原因：179K 数据有 15% 截断样本（无 `</think>`），教模型"thinking 可以不结束"
- 即使 loss 低，模型在 AIME 难题上退化成乱码（`222...` 重复）
- 简单题和训练集题目能正确回答（有 `</think>` + `\boxed{}`）

### 基础设施

- p5-1 (us-west-2/Oregon) 加入 SSH config，但被 slime OPD 训练占满
- p5-3 Docker 清理：释放 140G（prune unused images + build cache）
- p5-2 被 boshih 的 `slime-bh` 容器占用（MegatronTrainRayActor）
- Greenland 镜像 v2：修复 `datasets.features.Json` import error（升级 datasets 库）

### 当前运行

| 任务 | 机器 | 状态 | 预计完成 |
|------|------|------|---------|
| 4B Full FT 179K × 3ep | p5-4 | 🔄 step 61/1461 | ~20h |

### 下一步

1. 等 3ep 训练完成 → eval
2. 如果 3ep 还是不行 → 用 filtered 179K (~152K) 重训
3. 8B Full FT 179K 已经是好的 student baseline (63.3%)，可以开始 OPD
4. 收集 8B student rollouts → precompute teacher logprobs → Lightning OPD


---

## 2026-05-21 (Day 21)

### 4B Full FT 2ep (179K) Eval — AIME 3.3%

继续 2ep 训练完成，loss 0.213 vs v2 的 0.201。

**Eval (DP=8, max_tokens=30000, temp=0.6)**：
- AIME-2024: **3.3%** (1/30)
- AIME-2025: **3.3%** (1/30)

比 1ep (0%) 略好但远不如 v2 (50%)。

### 4B Full FT 3ep (179K, unfiltered) — AIME 3.3%

**配置**：lr=8e-5, cosine 0.1, 3 ep, global batch 256, 1461 steps, ~21h
**Final loss**: 0.194

**Eval**:
- AIME-2024: **3.3%** (1/30)
- AIME-2025: **6.7%** (2/30)

**详细错误模式**：
- 简单题：正确生成 `</think>` + `\boxed{}`（如 polar coordinates，正确答案 (3, π/2)）
- 训练数据题：完美工作（生成正确推理 + boxed answer）
- AIME 难题：推理 ~30K chars 后退化成乱码（`\.printStackTrace`, `0 0 0 0`, `\ \ \`）

模型本身能力没问题，是**对长推理的终止信号**没学好。

### 同题对比 v2 vs 3ep (相同 prompt, AIME-2024 
label=33):

**v2 (79K filtered × 3ep)**:
- 11,467 chars
- ✅ </think> + \boxed{33} (正确)

**3ep (179K unfiltered × 3ep)**:
- 32,047 chars (hit max)
- ❌ 没 </think>, 没 \boxed{}
- 末尾退化成
 `printStackTrace \0000...`

两个模型起点几乎一字不差，但 v2 在 11K 内收敛，3ep 推理失控。

### 数据质量分析

| 数据集 | 数量 | </think> | boxed | 平均长度 |
|--------|------|----------|-------|---------|
| 79K filtered (v2) | 79,341 | 100% | 100% | 38K chars |
| 179K merged | 179,341 | 85% | 86% | 39K chars |
| **152K think-filtered** | **152,479** | **100%** | **99.99%** | **39K chars** |

第一批 79K 和第二批 73K 的 generation 参数完全一致（temp=0.6, top_p=0.95, top_k=20, max_tokens=16384），response 长度分布也接近。

### 4B 152K filtered × 3ep (running)

启动了用 152K think-filtered 数据 + 3 epochs 重训，看是否能 match v2:
- p5-4 单节点, bs=8, accum=4, ZeRO-1
- 1461 steps, ~21h
- 当前 step 1324/1461 (91%), loss 0.194

### Ckpt-600 (1.2 epoch) 早期测试

发现 152K ckpt-600 在简单题（x²-5
x+6=0）上也退化（20K chars 乱码）。说明 1 epoch 不足，需要看最终 ckpt-1461。

### 关键观察总结

1. **8B 没问题**：用同样的 179K unfiltered 数据 1 epoch → AIME 63.3%
2. **4B 对数据量敏感**：79K × 3ep work，152K × 3ep 不 work
3. **`tie_word_embeddings=true`** 是 4B 特有问题，对噪声敏感
4. **Loss 不能预测 AIME**：3ep loss 0.194 < v2 的 0.201，但 AIME 远差
5. **退化模式一致**：~


---

## 2026-05-23 (Day 23)

### 关键诊断：4B SFT 失败的真正原因 = 混合 sampling 风格 ❌

经过一整天的 ablation 实验，把"4B SFT 大数据失败"的原因定位到 **teacher 数据的 sampling style 混合**。

#### 实验矩阵（这一天新增的数据点）

| 实验 | Data | Style | Steps | AIME-2024 | 备注 |
|------|------|------|:---:|:---:|------|
| v2 multi ckpt-600 | 79K-A | 单一 (temp=0.6) | 600 | 40% | v2 中间 ckpt |
| **v2 multi ckpt-700** | **79K-A** | **单一** | **700** | **60%** ⭐ | **新最佳 4B SFT** |
| v2 multi ckpt-759 | 79K-A | 单一 | 759 | 50% | (final，已知) |
| **v2 single-node ckpt-700** | **79K-A** | **单一** | **700** | **47%** | 8 GPU 单节点复现 |
| **73K-new ckpt-200** | **73K-B** | **单一 (temp=0.7)** | 200 | **27%** | Greenland，早期 |
| **73K-new ckpt-400** | 73K-B | 单一 | 400 | 40% | |
| **73K-new ckpt-600** | 73K-B | 单一 | 600 | 47% | |
| **73K-new ckpt-702** | **73K-B** | **单一** | **702** | **53.3%** ⭐ | **Greenland final** |
| 152K ckpt-600 | 152K-AB | **混合** | 600 | **0%** ❌ | 之前 eval |
| 152K ckpt-800 | 152K-AB | 混合 | 800 | 3.3% | |
| 152K ckpt-1400 | 152K-AB | 混合 | 1400 | 3.3% | |

#### 核心发现：data composition

`teacher_sft_179k_thinkfilter.jsonl` (152K) = `teacher_sft_filtered.jsonl` (79K) + `teacher_extra_100k_filtered.jsonl` (73K)

但**两批 teacher data 用了不同的 sampling 参数**：

| 数据批次 | Generation script | temperature | top_p |
|---|---|:---:|:---:|
| 79K v2 (style A) | `generate-teacher-8replica.sh` | **0.6** | **0.95** |
| 73K new (style B) | `generate-teacher-extra100k.sh` | **0.7** | **0.9** |
| Lightning-OPD reference | `data_curation/pipeline.py` | 0.7 | 0.9 |

Qwen3 model card 推荐 thinking mode 用 0.6/0.95，0.7/0.9 容易"endless repetition"。

#### Prompt overlap 数据

| Dataset | Samples | Unique prompts | Avg samples/prompt |
|---|---|---|:---:|
| 79K-A | 79,341 | 20,126 | 3.94 |
| 73K-B | 73,127 | 22,539 | 3.24 |
| **152K-AB** | **152,479** | **24,844** | **6.13** |

79K-A ∩ 73K-B prompt overlap = 17,821（88% of 79K-A 的 prompts 也在 73K-B 里）。152K 没有新 prompt：完全是 79K-A ∪ 73K-B。

也就是说，152K 训练的实际样子是：**~25K unique 数学题，每题平均 6 个 response，混合 temp=0.6 风格 + temp=0.7 风格**。一个 prompt 可能有 18 个 response，unique-word ratio 0.07-0.18 (严重重复退化)。

OpenThoughts3 source 本身就有大量重复 prompt（max 15 responses/prompt），我们的 dedup logic（`prompt[:200]+response[:200]` hash）没有真正 dedup prompt，只是过滤了完全相同的 (prompt, response) 对。

#### 失败机制

- **single-style** (79K 或 73K) → 4B 能学好：每 prompt ~3-4 个一致风格 response
- **mixed (A+B)** → 4B **崩**：同 prompt 同时出现 temp=0.6 (集中) 和 temp=0.7 (发散) 两种风格的 trace，4B 容量不够同时拟合两种分布 → mode collapse → 推理时退化

8B 容量足够吃 mix（179K-AB ×1ep → 63.3%）。

#### Loss vs AIME（再次确认）

| 实验 | Final loss | AIME |
|---|:---:|:---:|
| v2 multi ckpt-759 | 0.201 | 50% |
| v2 single ckpt-759 | 0.201 | 47% |
| 73K-B ckpt-702 | 0.209 | 53% |
| 152K-AB ×3ep | **0.196** | 3% |
| 179K-AB ×3ep | **0.194** | 3% |

混合数据 loss **更低** 但 AIME **更差**。loss 完全无法预测 termination quality。

#### 改动 / 新代码

- `slime/slime/backends/megatron_utils/loss.py` — `apply_opd_kl_to_advantages` 末尾加 per-sample / per-token KL dump（jsonl，含 token_ids + student_logp + teacher_logp + reverse_kl）
- `slime/slime/utils/arguments.py` — 加 `--opd-dump-kl-path`, `--opd-dump-kl-interval`, `--opd-dump-kl-max-samples`
- `slime/slime/backends/megatron_utils/actor.py` — 在 `compute_advantages_and_returns` 前把 `rollout_id` 塞进 `rollout_data`
- `scripts/train-opd.sh` — 接住三个新 env var
- `configs/opd-4b-v2-ckpt700-instant.env` — 新 OPD config，用 v2 ckpt-700 (60%) 当 4B student

#### 跑过的实验脚本 / 配置

- `scripts/run-sft-full-4b-v2-single.sh` — v2 single-node 复现（759 steps, 8 GPU）
- `scripts/run-sft-full-4b-73k-new.sh` — 73K-only Greenland
- `scripts/run-sft-full-8b-73k-new.sh` — 8B 73K-only（已写，未提交）
- `scripts/run-sft-full-4b-152k-700steps.sh` — 152K @ 700 steps Greenland（**正在跑**）
- `greenland/job_sft_full_{4b,8b}_73k_new.json`
- `greenland/job_sft_full_4b_152k_700steps.json`

#### 检查脚本

- `scripts/audit_sft_data.py` / `audit_sft_tokens.py` — 数据质量审计
- `scripts/compare_teacher_data.py` — 两批 teacher data 风格对比
- `scripts/check_chat_template.py` — Qwen3 chat template 行为检查
- `scripts/check_template_round_trip.py` — raw vs templated 对比
- `scripts/check_prompt_overlap.py` — 79K/73K/152K prompt 集合关系
- `scripts/check_ot3_dups.py` — OpenThoughts3 source 重复 prompt 统计
- `scripts/analyze_prompt_response_dups.py` — 同 prompt 多 response 长尾分析
- `scripts/eval-152k-ckpts.sh` / `eval-73k-new-ckpts.sh` / `eval-v2-ckpts.sh` — 多 ckpt eval orchestration
- `scripts/plot_v2_single_curves.py` — single vs multi 训练曲线对比
- `scripts/plot_all_sft_curves.py` — 所有 SFT 实验汇总图（重写版）

#### 所有 4B SFT baselines (按 AIME 排序)

| 候选 | AIME-2024 | 备注 |
|---|:---:|---|
| **v2 multi ckpt-700** | **60%** ⭐ | 79K-A, 16 GPU multi-node, 唯一过 50% 的 ckpt |
| 73K-new ckpt-702 | 53% | 73K-B, Greenland single-node, single-style |
| v2 multi ckpt-759 | 50% | 79K-A, multi-node final |
| v2 single ckpt-700 | 47% | 79K-A, 8 GPU single-node 复现 |
| v2 multi ckpt-600 | 40% | 79K-A, multi-node 中间 |
| 73K-new ckpt-600 | 47% | 73K-B 后期 |
| 73K-new ckpt-400 | 40% | 73K-B 中期 |
| 73K-new ckpt-200 | 27% | 73K-B 早期（28% training）|
| 152K-AB ×3ep | 3.3% | 混合数据（崩）|

#### 待跑实验

- ✅ Greenland: `4b-73k-sft` (job ID `ba889b94-68df-43db-b86f-89db26b1cfdc`) — 完成 53.3%
- 🔄 Greenland: `4b-152k-700steps` — 提交中，看 152K 在 same budget (700 steps) 下能否 work
- ⏳ 8B 73K-only Greenland — 未提交（capacity control vs 4B 73K）

#### 下一步

1. 等 152K-700steps 结果（关键：能不能在 short budget 下"逃过"混合数据陷阱）
2. **开始 OPD 实验**：4B SFT baseline 已经够用（v2-700 60% 或 73K-702 53%），8B baseline 63.3% 也 ready
3. 用改过的 slime + per-token KL dump 跑 instant OPD，离线分析 KL 长尾

#### 文件位置

| 文件 | 位置 |
|---|---|
| v2 multi ckpts | p5-3: `/opt/dlami/nvme/qzf/models/sft-qwen3-4b-full-teacher-v2/v9-20260515-233350/checkpoint-{600,700,759}` |
| v2 single ckpts | p5-3: `/opt/dlami/nvme/qzf/models/sft-qwen3-4b-full-v2-single/v0-20260523-000055/checkpoint-{100,200,...,759}` |
| 73K-new ckpts | p5-3: `/opt/dlami/nvme/qzf/models/sft-qwen3-4b-full-73k-new/checkpoint-{100,...,702}` |
| 8B 179K ckpt | S3: `s3://delphi-greenland-res-alpha/outputs/ffe02619-dddf-498e-92ec-10fbb7efce89#0/sft-checkpoint/v1-20260518-232752/` |
| 152K final ckpt | p5-4: `/opt/dlami/nvme/qzf/models/sft-qwen3-4b-full-152k-3ep/v0-20260521-235133/checkpoint-1461` |
| All curves plot | `sft_all_experiments_curves.png` |
| v2 single vs multi plot | `sft_v2_single_vs_multi.png` |


---

## 2026-05-24 (Day 25)

### 4B Instant OPD with KL Dump — 完成 ✅

**配置**：
- Student: 4B Full FT v2 ckpt-700 (60% AIME, 最强 4B SFT baseline)
- Teacher: Qwen3-8B (73.3% AIME)，远程部署在 qzf-dev (4×L40S, TP=1 DP=4)
- p5-3 本地：4 actor (TP=4) + 4 rollout engines
- Instant OPD: kl_coef=1.0, lr=5e-7, max_response_len=8192, n_samples=4, batch=64
- 300 rollouts, save every 25 → 12 checkpoints
- Per-token KL dump every 10 rollouts × 8 samples/rank

**训练时间**：~10.5h (08:28 → 19:14 UTC)

**训练趋势**：
- opd_reverse_kl: **0.165 → 0.092** (↓44%)
- truncated_ratio: 0.67 → 0.67 (稳定，~65% response 顶到 8192)
- repetition_frac: 0.0 (全程无退化)
- grad_norm: 0.5-0.8 (稳定)
- kl_from_ref (log_probs vs ref): 0.10 → 0.09 (student 没有偏离太远)

**Checkpoints**：
```
p5-3: /opt/dlami/nvme/qzf/models/opd-4b-v2-ckpt700-instant/
  iter_0000024, iter_0000049, ..., iter_0000299 (12 个)
  kl_dump/ (116 files, 357MB, r10-r290 × 4 ranks)
```

**KL Dump 统计**：
- 29 个 dump 时间点 (r10, r20, ..., r290)
- 每个 dump 4 ranks × 8 samples = 32 samples
- 总 ~928 samples，每条含 token_ids + student_logp + teacher_logp + reverse_kl + advantage

### 速度分析

| 阶段 | 时间 | 占比 |
|------|------|:----:|
| Rollout generation | ~80s | 60% |
| ref_log_probs (forward) | 10s | 7% |
| log_probs (forward) | 10s | 7% |
| actor_train (gradient) | 32s | 24% |
| update_weights | 0.3s | <1% |
| **总单步** | **~135s** | |

**瓶颈**：Rollout 生成占 60%。78% response 顶到 max_tokens=8192，4 个 sglang engine 满载 (~1400 tokens/gpu/sec)。

### 基础设施

**qzf-dev (4×L40S) 作为远程 teacher**：
- Docker: `lmsysorg/sglang:latest`, TP=1 DP=4, port 30000
- 同 VPC 私网 (172.31.31.105)，延迟 1.5ms
- 首次 TP=4 OOM（mem_fraction=0.85 + 256 并发 prefill 太激进）
- 改 DP=4 + max_running=32 后稳定运行全程

**代码修改**：
- `slime/backends/megatron_utils/data.py`：`log_rollout_data` 跳过 `_` 开头的 key（修复 `_rollout_id` 导致的 ValueError）
- 3 个 injector 脚本（`_inject_opd_args.py`, `_inject_opd_loss.py`, `_inject_opd_actor.py`）在 HEAD 上注入 cumulative + dump 功能
- `scripts/train-opd-extteacher.sh`：支持外部 teacher URL 的训练脚本
- `configs/opd-4b-v2-ckpt700-instant-extteacher.env`：外部 teacher 配置

### 下一步

1. 离线分析 KL dumps（instant KL 分布 + future KL 分布）
2. Eval OPD checkpoints（iter_0000099, 199, 299）on AIME-2024
3. 对比 instant vs cumulative OPD
4. 写 `scripts/analyze_kl_dumps.py`

### 文件位置

| 文件 | 位置 |
|------|------|
| OPD checkpoints | p5-3: `/opt/dlami/nvme/qzf/models/opd-4b-v2-ckpt700-instant/iter_*` |
| KL dumps | p5-3: `/opt/dlami/nvme/qzf/models/opd-4b-v2-ckpt700-instant/kl_dump/` |
| Teacher server | qzf-dev: `docker container teacher-sglang` (可能已停) |
| Train script | `scripts/train-opd-extteacher.sh` |
| Config | `configs/opd-4b-v2-ckpt700-instant-extteacher.env` |
| Ray job log | p5-3 container: `/tmp/ray/session_2026-05-24_08-28-14_954267_97990/logs/job-driver-raysubmit_uGJ49mwL5fRx5yNt.log` |


---

## 2026-05-26 (Day 27)

### 一句话总结

**OPD 在 4B 上是有效的（+5.6pt AIME-2024），之前 n=1 eval 完全淹没在 noise 里。Lightning-OPD 4B paper recipe 训练已起。**

---

### 关键转折：n=1 eval 的 noise 把信号埋了

之前对 4B Instant OPD（v2-700 → 300 rollouts）的 n=1 eval：

| ckpt | AIME-2024 (n=1) | AIME-2025 (n=1) |
|---|:---:|:---:|
| v2 baseline | 60.0% | 47% |
| iter_099 | 50.0% | 40.0% |
| iter_199 | 53.3% | 40.0% |
| iter_299 | 53.3% | 50.0% |

→ 看起来 OPD 让 4B **退步 -6.7pt**，反复推断"配置不对"、"k-step 救不了"。

**问题：30 题 × n=1 的 pass@1 标准误 ≈ 9pt**。OPD 的真实 +5pt 信号完全在 noise 范围内。

#### 重 eval n=16

3 台机器并行（p5-2/3/4），每模型 30 题 × 16 sample（avg pass@1）：

| 模型 | AIME-2024 (n=16) | AIME-2025 (n=16) | 综合 | avg_len |
|------|:---:|:---:|:---:|:---:|
| **baseline v2-700** | 48.8% | 40.6% | 44.7% | 55K-58K |
| OPD iter_099 | 50.2% | 46.9% | 48.6% | 55K-56K |
| OPD iter_199 | 50.4% | 42.1% | 46.3% | 53K-57K |
| **OPD iter_299** | **54.4%** | **45.2%** | **49.8%** | 52K-57K |
| Δ (iter_299 vs baseline) | **+5.6pt** | **+4.6pt** | **+5.1pt** | -2.6K |

**OPD 单调上升、avg_len 单调下降**：student 学到 teacher 的更紧凑 thinking 风格 ✓。

参考 Lightning-OPD 4B paper（n=32 standard OPD）：56.7% → 65.4% AIME-2024（+8.7pt）。我们 +5.6pt 比他们小 3pt，但**方向对、相对量级一致**。

---

### Lightning-OPD paper 完整 recipe 复现 — 新训练已起

paper Table 6 的 4B OPD setting，以及它和我们之前 instant config 的差异：

| 参数 | 之前 (instant) | **Lightning recipe** | 倍数 |
|---|:---:|:---:|:---:|
| LR | 5e-7 | **2e-6** | 4× |
| MAX_RESPONSE_LEN | 8192 | **4096** | ÷2 |
| TEMPERATURE | 0.6 | **0.8** | +33% |
| TOP_P | 0.95 | **1.0** | disable |
| TOP_K | 20 | **-1** | disable |
| NUM_ROLLOUT | 300 | **600** | 2× |
| ROLLOUT_BATCH_SIZE | 16 | **64** | 4× |
| GLOBAL_BATCH_SIZE | 64 | **256** | 4× |
| `--include-verifiable-reward` | ✗ | ✓（**logging only**, 不影响 advantage）|
| advantage estimator | grpo + opd-kl-coef=1.0 | (paper 用 `on_policy_distillation`)|

#### 关键澄清

- **Lightning-OPD 用了 verifiable reward 当 advantage 一部分？没有。** paper 把 OPD 跟 RLVR 对立，`reward_func` hardcode 返回 `[0.0]*N`，flag 只用于 wandb logging。
- **slime 主线 vs Lightning-OPD fork**：主线没有 `on_policy_distillation` advantage estimator choice，但我们 `--advantage-estimator grpo --use-opd --opd-kl-coef 1.0` reward=0 时数学等价（advantage = 0 - 1.0 × reverse_kl = log_T - log_S）。差别只在 advantage clip [-10,10] 和 distributed whitening。
- **Eval template**：Lightning-OPD `apply_chat_template` 不加 `<think>\n` prefix，靠 SFT 模型自动 emit `<think>`。我们手动加 prefix —— sanity check（v2-700 模型）显示两种格式都 100% 触发 `<think>` 自动生成 + 100% 有 `</think>` + `\boxed{}`，**等价**。

#### Recipe 配置文件

新增 `configs/opd-4b-lightning-recipe.env` + 复用 `scripts/train-opd-extteacher.sh`。

**已启动**：p5-3, 8 GPU 全部给 student（4 actor TP=4 + 4 rollout sglang），teacher 用 qzf-dev 远程 Qwen3-8B (DP=4)。预计 ~17h 完成 600 rollouts（步速 ~100s/step）。

**Step 1-3 监控**：
```
                 r1     r2     r3
opd_reverse_kl  0.139  0.142  0.144
truncated_ratio 0.92   0.95   0.96    ⚠️
response_length 4012   4041   4083
grad_norm       1.04   1.04   1.04    ✓
loss            0.139  0.142  0.144
```

⚠️ **truncated_ratio 96%** — 几乎所有 rollout 顶到 4096，未学到 termination。这是 paper 的设定（"increasing beyond 4096 doesn't help"），但他们 train DAPO + eval AIME 的 transfer 我们要看。

---

### 152K-700-steps Greenland SFT 实验回收

5/24 提交的 Greenland job（`8bfd9b6e-...`）成功完成。**目的**：测试 152K 混合数据（A+B 风格）短训练（700 step ≈ 1.18 epoch）能否避开 152K × 3ep 的崩溃。

#### 一个乌龙

S3 的 `outputs/8bfd9b6e-...#0/sft-checkpoint/` 下有**两个 versioned 子目录**：
- `v0-20260523-054547`：**遗留的 8B 训练**（args.json `model: Qwen3-8B-Base`），同实例之前 job 残留
- `v1-20260524-070537`：**正确的 4B-152K-700steps**（args.json `model: Qwen3-4B-Base`）✓

第一次随手 `aws s3 cp` 抓了 v0 → 拿到 8B 模型 → config hidden=4096 → 一度怀疑 job 配错了。其实是 ms-swift 的多版本子目录机制。教训：**Greenland job 输出可能有多个 version dir，pull 之前看 mtime/args.json 确认**。

#### 训练摘要（v1 4B-152K-700-steps）

| 指标 | 值 |
|---|---|
| Model | Qwen3-4B-Base ✓ (4022.5M params) |
| Data | teacher_sft_179k_thinkfilter.jsonl (152K filtered = 79K-A + 73K-B mix) |
| Steps | 700 (cosine over 700) |
| Epoch | 1.44 over 152K |
| Final loss | **0.219** |
| LR / schedule / packing | 8e-5 / cosine / packing=True ✓ |
| Runtime | 10h 13min on 8×H100 |
| Avg packed seq | 13.3K tokens |

| 实验 | Steps | Final loss | AIME-2024 (n=16) |
|---|:---:|:---:|:---:|
| v2 (79K-A × 3ep) | 759 | 0.246 | **48.8%** ⭐ baseline |
| 152K-3ep (mix × 3ep) | 1461 | 0.196 | 3% (n=1, 之前结论) |
| **152K-700-steps (mix × 1.18ep)** | **700** | **0.219** | **eval 中** |

#### Eval 进展

p5-4 上跑 n=16 eval，10/30 时 AIME-2024 = 51.2%（前 5 题 simple → 高估，后续会下降）。第一次 p5-3 上跑出来 36.9%（**SGLang server 被中途 pkill**，可能 corrupted），重跑确认中。

预测：最终 AIME-2024 ≈ **45-52%**（接近或略低于 v2 baseline 48.8%）。混合数据陷阱在短训练下"减轻但不消失"。

---

### Eval 基础设施改进

#### eval_math.py: avg_pass_at_1（noise-reduced metric）

**Bug 修复**：原版 `pass1 = 1 if rewards[0] > 0 else 0` 只看第一个 sample，浪费 n_samples-1 个数据点。

新增 `avg_pass_at_1 = mean(rewards over n_samples)` — 真正利用 n=16 降 noise（标准误从 ±9pt 降到 ±2-3pt）。

向后兼容：`pass_at_1` 字段保留为"first sample only"。

#### SGLang n=16 routing fix

**坑**：`/generate` 单请求 n=16 → 单 SGLang DP replica 处理 16 sample × 30K tokens = 480K KV per replica，超过 32K KV cache → SGLang server crash → 全部 400 Bad Request。

**修复**：把 16 sample 拆成 16 个独立 n=1 请求 ThreadPoolExecutor 并行发出 → SGLang 自动 DP 路由分配到 8 个 replica → 每个 replica 同时只有 ~2 sample × 30K tokens = 60K KV，正常工作。

#### Eval 脚本

新增 `scripts/eval-aime-n16.sh`：
- 入参 `MODEL_PATH` + `MODEL_NAME`
- DP=8, max_tokens=30000（4B Base 上限），temperature=0.6, top_p=0.95
- AIME-2024 + AIME-2025
- 输出 `eval_results_n16/{aime2024,aime2025}_<MODEL_NAME>.json`

#### 跨节点 SSH config

p5-3 内部 ssh → p5-2/p5-4 (172.31.x.x 私网)，rsync 7.6G HF model 用 ~17s。set up:
```
Host p5-2-int
  HostName 172.31.3.242
  User ubuntu
  IdentityFile ~/.ssh/dl-machine-ohio.pem
```

**避免本地 mac 中转**（saves 30+ min on 2-3 model copies）。

---

### KL Dump 离线分析（前一天 24 号补充）

dump 数据问题：注入脚本只存了 logprobs，**`prompt_token_ids`/`response_token_ids` 都是空 list**。所以无法做"`</think>` token 是不是 KL outlier"分析。

#### 用 reverse_kl 序列做位置分析（中点 rollout r150）

| Pearson r(instant_kl, future_kl_K) | K |
|:---:|:---:|
| 0.72 | 2 |
| 0.53 | 4 |
| 0.38 | 8 |
| 0.28 | 16 |
| 0.21 | 32 |
| 0.05 | full suffix |

→ K=8 是 cumulative OPD 的 sweet spot：方差缩小 6.5x，但和 instant KL 还有 60%+ 独立信号。

**Position 分析**：头部 0-30% pos 偏高（0.17-0.22），后部偏低（0.11-0.16）— 没有看到尾部 spike，但因为没 token IDs，无法验证 special token 假设。

文件：`scripts/analyze_kl_dumps.py`, `kl_analysis/figures/*.png`, `kl_analysis/summary.json`。

---

### Slime / Lightning-OPD eval 设定 vs 我们

| 项 | Slime (Lightning-OPD) | 我们 |
|---|---|---|
| Reward function | `get_deepscaler_rule_based_reward` | **same** ✓ |
| Aggregation | mean over (n_problems × n_samples) | mean(per_problem(mean over n_samples)) — **等价** |
| Chat template | `apply_chat_template(..., add_generation_prompt=True)` | 手动 `<\|im_start\|>...assistant\n<think>\n` |
| `<think>` prefix | ❌（让 model 自决） | ✅ 强制（不影响 v2 SFT 的行为） |
| n samples | 多次独立 n=1 + 不同 seed | 16 个独立 n=1 并发请求 |
| 默认 max_tokens | 32768 | 30000（Qwen3-4B-Base context limit） |
| paper n_samples | **32** | 我们 16 |

我们 n=16 比 paper 少一半但已足够压住 noise 到 ±2pt 范围。

---

### 当前运行任务

| 节点 | 任务 | 状态 |
|---|---|---|
| p5-3 | Lightning-recipe 4B OPD 训练 | step 3/600，~17h total |
| p5-4 | 152K-700-steps n=16 eval | 10/30 AIME-2024，预计 30 min |
| p5-2 | idle | 可用于 cumulative ablation |
| qzf-dev | Qwen3-8B teacher (DP=4) | serving，用于 Lightning-recipe |

#### 关键 checkpoint 位置

| 文件 | 位置 |
|---|---|
| 4B Instant OPD ckpts (300 rollouts) | p5-3:/opt/dlami/nvme/qzf/models/opd-4b-v2-ckpt700-instant/ |
| 4B Instant HF (3 ckpts ×8G) | p5-3 + 同步到 p5-2/p5-4 |
| 152K-700-steps SFT (8B 错误版本) | 已删除 |
| 152K-700-steps SFT (4B 正确) | p5-3:/opt/dlami/nvme/qzf/models/sft-qwen3-4b-full-152k-700steps-ckpt700/, p5-4 同 |
| n=16 eval results | 各 p5-X:/workspace/k-step-opd/eval_results_n16/ |
| Lightning-recipe ckpts (running) | p5-3:/root/.cache/huggingface/opd-4b-lightning-recipe/iter_* |

#### 关键代码 / config

- `configs/opd-4b-lightning-recipe.env` — Lightning paper recipe（new）
- `scripts/train-opd-extteacher.sh` — 外部 teacher 训练（同步到 p5-3 latest）
- `scripts/eval-aime-n16.sh` — n=16 eval（new）
- `scripts/eval-aime-n16-p5-3.sh` — 串行跑两 model
- `scripts/collect-n16-results.sh` — 跨节点拉结果
- `scripts/sanity_check_chat_template.sh` — `<think>` prefix 验证
- `scripts/analyze_kl_dumps.py` — KL dump 分析
- `eval_math.py` — patched: `avg_pass_at_1` + n=1 splitting
- `Lightning-OPD/2604.13010v2.pdf` — paper full text（10 pages + appendix）
- `sft_all_experiments_curves.png` — 加上 152K-700-steps（deeppink）

---

### 下一步

#### 高优先级（等当前任务）
1. 等 152K-700-steps eval 完整 30/30，确认 AIME 数字
2. 等 Lightning-recipe 训练 50 / 100 / 200 / 600 rollouts，分批 eval
3. 比较：Lightning-recipe vs Instant OPD 真实差距（n=16）

#### 中优先级
4. 如果 Lightning-recipe 也只到 +5pt，**换更大 teacher**（Qwen3-32B）：gap 13pt → 25pt+
5. 如果 Lightning-recipe 给 +8pt 甚至 +10pt，**继续做 cumulative K=8 ablation**（KL dump 分析支持）

#### 低优先级 / 长期
6. 重生成 SFT data：温度统一 0.7 + 不混合，避开 152K mix 陷阱
7. 重做 SFT：300K prompts × 3000 steps（Lightning-OPD 4× 我们的训练量）
8. 写 paper outline


### 教训汇总

1. **n=1 pass@1 是 noise 机器**。30 题 ×1 标准误 ±9pt。任何 ±5pt 之内的差距都是 noise。Lightning-OPD paper 用 n=32 是对的。

2. **Greenland job 多版本子目录**。同一 instance 上之前残留的 ms-swift 输出（v0-...）会和当前 job（v1-...）共存。pull 之前 list versioned dirs + 看 args.json 里 model path。

3. **slime 配置 ≠ Lightning-OPD 完整 recipe**。两者都 work 但参数差很多。slime 主线 example 用 `lr=1e-6, max_response=16384, T=1.0`，paper 用 `lr=2e-6, max_response=4096, T=0.8`。差别影响很大。

4. **`--include-verifiable-reward` 是 logging 不是 advantage**。在 Lightning-OPD fork 的 slime 里也只是给 wandb 看 task accuracy，不进入梯度。

5. **SGLang n>1 单请求 ≠ 自动 DP 分配**。一个 n=16 请求会绑定一个 replica，KV 容量不够直接 crash。要 n=1 × 16 并发让 router 分散。

6. **`<think>` prefix 不是问题**。post-trained student / teacher-consistent SFT student 都会自动 emit。我们手动加 prefix 只是把 token 从 generation 移到 prompt，等价。

7. **OPD 在 small gap setting 下增益小但真实**。4B (60%) → 8B teacher (73%) gap 13pt → OPD 提 +5.6pt（gap 的 ~43%）；paper 4B 56%→Qwen3-8B 73% gap 17pt → OPD +8.7pt（gap 的 ~51%）。增益 ∝ gap，符合直觉。


---

## 2026-05-28 (Day 28) — naming cleanup + opd-4b-B eval

### 命名重命名

之前用 "instant" / "lightning" 命名两个 OPD run 极易和 paper 概念混淆。统一改成：

| 新名 | 旧名 | 配置概述 |
|---|---|---|
| **opd-4b-A** | "instant" / "v2-ckpt700-instant" | 5/24, 保守 hand-picked (lr=5e-7, max_resp=8192, T=0.6, 300 rollouts) |
| **opd-4b-B** | "lightning-recipe" | 5/26, paper Table 6 (lr=2e-6, max_resp=4096, T=0.8, 600 rollouts) |

> ⚠️ 两者都是 **standard online OPD**。paper 的 "Lightning OPD" 是 **offline precomputed-teacher** method，我们没实现。opd-4b-B 只是借了 paper 的超参。

详见 `NAMING.md`。disk 上加了 symlinks (opd-4b-A → opd-4b-v2-ckpt700-instant 等)，configs/ 里加了 opd-4b-A.env / opd-4b-B.env（带头部说明），eval JSON 文件名保留旧后缀。

### opd-4b-B eval 完整结果

5/27 完成 600 rollouts 训练（10h+），转 HF + n=16 eval iter_99/299/599：

| ckpt | AIME-2024 | AIME-2025 | 综合 | avg_len | pass_any (24/25) |
|---|:---:|:---:|:---:|:---:|:---:|
| baseline v2-700 | 48.8% | 40.6% | 44.7% | 55K-58K | 76.7%/73.3% |
| opd-4b-A iter_99 | 50.2% | 46.9% | 48.6% | 55K-56K | 83.3%/70.0% |
| opd-4b-A iter_199 | 50.4% | 42.1% | 46.3% | 53K-57K | 76.7%/70.0% |
| opd-4b-A iter_299 | 54.4% | 45.2% | 49.8% | 52K-57K | 76.7%/76.7% |
| opd-4b-B iter_99 | 53.5% | 45.2% | 49.4% | 52K-55K | 83.3%/66.7% |
| **opd-4b-B iter_299** | **55.8%** ⭐ | 45.0% | **50.4%** | 51K-57K | 80%/73.3% |
| opd-4b-B iter_599 | 55.2% | 44.6% | 49.9% | 50K-55K | 83.3%/66.7% |

**关键发现**:
1. **opd-4b-B 比 opd-4b-A 多 +1.4pt AIME-2024 (在 iter_299)** — paper 配置确实更好
2. **opd-4b-B 在 iter_299 饱和** — iter_599 没继续涨（与 paper Figure 3b 一致：4B OPD ~50 step 后饱和）
3. **avg_len 单调下降** (55K → 50K) — student 学到 teacher 的紧凑 thinking
4. **离 paper 的 65.4% 还差 10pt** — 主要来自 SFT baseline 低 8pt（数据 4× 少）

### KL/loss 训练曲线对比

`kl_analysis/figures/opd_trajectories_kl.png` 有两个 panel：
- 左：opd-4b-A vs opd-4b-B 的 mean reverse_kl 随 rollout id（log y 轴）
- 右：opd-4b-A 的 KL 分布（mean/p90/p99）

`kl_analysis/figures/opd_trajectories_lightning_full.png`：opd-4b-B 完整 6 metrics（reverse_kl, loss, grad_norm, truncated, response_len, kl_loss）

#### 反直觉的发现

| | opd-4b-A | opd-4b-B |
|---|:---:|:---:|
| Final reverse_kl | 0.081 | **0.093** |
| Final AIME-2024 (n=16) | 54.4% | **55.8%** |

**opd-4b-B reverse_KL 反而比 opd-4b-A 高，但 AIME 更好。**

原因：
- opd-4b-B 用 lr 4× + sampling 4× 多样（T=0.8 + no top_k） → 每步把 student 推得更激进，sampled trajectory 上 KL 不容易降
- opd-4b-A 用保守 sampling → student 没怎么探索，sampled KL 容易看着"降"，但 student 没学到东西

**Takeaway**：训练时 reverse_KL 不是好的目标 metric。它是 *sampled* 量，受 sampling 策略影响大。**真正的 ground truth 是 eval-time 的 avg pass@1**。

### 数据完整性

- **opd-4b-A trajectory**：原 Ray driver log 已被自动清理。我们只有 **KL dump** (357MB) 抽样数据 — 每 10 rollouts 一个数据点，每点是 32 sequences 的 per-token KL 均值 → 用作近似估计。
- **opd-4b-B trajectory**：完整 driver stdout log 在 p5-3 (`/workspace/k-step-opd/logs/opd_lightning.log`, 166MB)，每 rollout 一个数据点 (599 点)。

### 仍要做的

1. eval opd-4b-B 中间 ckpt iter_199/399/499，画完整 7-point AIME pass@1 vs rollout 曲线（最关键的图）
2. 决定下一步：复现 paper SFT (300K data × 3000 steps multi-node) vs 换 32B teacher vs 写 paper outline


---

## 2026-05-29 (Day 29) — opd-4b-B 完整 eval + K-step bias-variance 分析

### opd-4b-B 三个 ckpt eval 结果（n=16）

转 iter_99 / iter_299 / iter_599 → HF（< 1 min/ckpt，4B 小），三台机器并行 eval：

| ckpt | AIME-2024 | AIME-2025 | 综合 | avg_len | pass_any |
|---|:---:|:---:|:---:|:---:|:---:|
| baseline v2-700 | 48.8% | 40.6% | 44.7% | 55K-58K | 76.7%/73.3% |
| **opd-4b-B iter_99** | 53.5% | 45.2% | 49.4% | 52K-55K | 83.3%/66.7% |
| **opd-4b-B iter_299** ⭐ | **55.8%** | 45.0% | **50.4%** | 51K-57K | 80%/73.3% |
| **opd-4b-B iter_599** | 55.2% | 44.6% | 49.9% | 50K-55K | 83.3%/66.7% |
| opd-4b-A iter_99 | 50.2% | 46.9% | 48.6% | 55K-56K | 83.3%/70.0% |
| opd-4b-A iter_299 | 54.4% | 45.2% | 49.8% | 52K-57K | 76.7%/76.7% |

**关键观察**:
- opd-4b-B iter_299 是新 best：**+7.0pt AIME-2024 vs baseline**（vs A 的 +5.6pt）
- iter_299 → iter_599 没继续提升 → **训练饱和点 ~300 rollouts**（与 paper Figure 3b 一致）
- iter_99 已经达到 53.5% → **超快收敛**（前 100 rollouts 已经吃掉 ~80% 的 OPD gain）

vs Lightning-OPD paper (n=32): SFT 56.7% → standard OPD 65.4% (+8.7pt). 我们 +7pt，gap ~9pt 主要来自 SFT baseline 低 8pt（数据 4× 少）。

### 命名清理

旧名"instant" / "lightning recipe" 极易和 paper 概念混淆，统一改为：

| 新名 | 旧名 | 配置概述 |
|---|---|---|
| **opd-4b-A** | "instant" / "v2-ckpt700-instant" | 5/24, hand-picked 保守 (lr=5e-7, max_resp=8192, T=0.6, 300 rollouts) |
| **opd-4b-B** | "lightning-recipe" | 5/26, paper Table 6 (lr=2e-6, max_resp=4096, T=0.8, 600 rollouts) |

> ⚠️ 两者都是 **standard online OPD**。paper 的 "Lightning OPD" 是 offline precomputed-teacher method，我们没实现。

详见 `NAMING.md`。disk 加 symlinks（旧路径仍 work），configs/ 加 opd-4b-A.env / opd-4b-B.env（带头部说明），eval JSON 名保留旧后缀。

### 训练曲线分析

#### opd-4b-B 完整 trajectory (`kl_analysis/figures/opd_trajectories_lightning_full.png`)

| Metric | Start (r1) | End (r599) | 变化 |
|---|:---:|:---:|---|
| opd_reverse_kl | 0.139 | 0.094 | -32% (但震荡 0.09-0.15) |
| train/loss | 0.139 | 0.094 | = opd_reverse_kl × kl_coef=1.0 |
| grad_norm | 1.04 | 0.37 | -65%（持续下降，收敛中）|
| truncated_ratio | 0.92 | 0.92 | 高位稳定 ⚠️ |
| response_length | 4012 tokens | 4019 tokens | flat（被 4096 cap 锁住）|
| kl_loss vs ref | 0 | 0.07 | student 慢慢漂离 SFT 起点 |

**反直觉**：opd-4b-B reverse_KL 反而比 opd-4b-A 高（0.094 vs 0.081），但 AIME 更好。
- **解释**：sampled trajectory 上的 reverse_kl 受 sampling 策略影响（lr 大 + T=0.8 + no top-k 让 student 探索更广，sampled KL 不容易降）
- 真正反映 OPD 是否 work 的是 **eval-time avg pass@1**，不是训练时 reverse_kl

### K-step Cumulative KL 深度分析（重要）

抓出 opd-4b-A 和 opd-4b-B 的 357MB / 206MB KL dump 数据，对每个 token 位置 t，分析 K-step KL 的统计性质。

#### Setup

```
mean_K[t] = (1/K) × Σ_{d=0}^{K-1} reverse_kl[t+d]   # slime cumulative v2 用的
sum_K[t]  =        Σ_{d=0}^{K-1} reverse_kl[t+d]   # RL 教材的 reward-to-go
```

#### Mean variance 随 K 缩水（slime cumulative 当前实现）

| K | A: var(mean_K) | B: var(mean_K) | 1/K ideal |
|:---:|:---:|:---:|:---:|
| 1 (instant) | 0.445 | 0.295 | 1.0 |
| 8 | 0.068 | 0.043 | 0.125 |
| 64 | 0.012 | 0.007 | 0.016 |
| full | 0.001 | 0.001 | 0.0002 |

完美 1/K-ish 衰减。这是 **平均化 noise reduction**，不直接反映 RL gradient variance。

#### Sum variance 随 K 增长（教材 reward-to-go）

| K | A: var(Σ_K) | A: var ratio | B: var(Σ_K) | B: var ratio | K ideal |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.44 | 1.0× | 0.30 | 1.0× | 1× |
| 2 | 0.93 | 2.1× | 0.62 | 2.1× | 2× |
| 8 | 4.37 | 9.9× | 2.74 | 9.3× | 8× |
| 64 | 48.1 | 109× | 28.4 | 96× | 64× |
| **full** | **163,030** | **368,000×** | **13,740** | **46,000×** | ~T |

✅ **完美验证 REINFORCE 教材 bias-variance tradeoff**：
- K-step sum variance 随 K 接近线性增长（实际略快 — token autocorrelation 让 variance 比 ideal 独立大）
- K=full 是单 trajectory 的 Monte Carlo unbiased Q estimate，variance 巨大

#### 在不同视角下 K=1 是 "高 bias" 还是 "low bias"?

**RL 视角**：K=1 = 单 token reward → 忽略 a_t 对 future state 的影响 → **biased Q estimate**。K=full = MC reward-to-go → **unbiased Q**（但高 variance）。

**Lightning-OPD paper 视角（Theorem 3.6）**：把 OPD 当 distillation surrogate（不当 RL），advantage = log_T - log_S 直接最小化 KL(π_θ || π_T)，**有 fixed point 保证**。1-step "够用"。

这两个视角是不同的合理性论证，**paper 用 surrogate 视角所以不需要 cumulative**，但 RL 视角的 cumulative 仍然是开放方向。

#### 时序分析（`kl_analysis/figures/kstep_over_time_*.png`）

K-step KL 的**统计性质在整个训练过程稳定**：
- Pearson r(instant, K=8) ≈ 0.38 全程不变
- var(K=8 mean) / var(instant) ≈ 0.15 全程不变

→ K=8 sweet spot **不需要 schedule**，也不依赖 OPD recipe（A 和 B 数字几乎相同）。

#### 数据告诉我们的 Cumulative OPD 设计空间

| 实验 | Advantage 形式 | bias | variance | magnitude | slime 现成支持 |
|---|---|:---:|:---:|:---:|:---:|
| K=1 (instant) | reverse_kl[t] | 高 | 低 | 1× | ✓ |
| **K=8 mean (slime cumulative v2)** | (1/8) Σ_{d<8} kl[t+d] | 高 | 低/8 | 1× | ✓ |
| K=8 sum (教材 reward-to-go) | Σ_{d<8} kl[t+d], coef=1/8 | 中 | 低×8 | 1× (after coef) | ❌ 需要去掉 v2 的 /K |
| K=full suffix | Σ_{d=t}^{T-1} γ^d kl[t+d] | 低 | 高 | 巨大 | ❌ 需要 coef×~600 |

**关键 insight**: slime 现在的 mean 实现**没有 reward-to-go 的 bias-correction 效应**，本质是局部 noise reduction。如果想真做 reward-to-go 实验，需要 **改 slime 加 sum 选项 + 重调 kl_coef**。

### 文件

- `kl_analysis/figures/kstep_kl_A.png` / `kstep_kl_B.png` / `kstep_kl_compare.png` — 单 run + side-by-side
- `kl_analysis/figures/kstep_over_time_A.png` / `kstep_over_time_B.png` / `kstep_over_time_compare.png` — 时序
- `kl_analysis/figures/kstep_sum_vs_mean.png` — bias-variance tradeoff 教学版
- `kl_analysis/figures/opd_trajectories_kl.png` / `opd_trajectories_lightning_full.png` — 训练 trajectory
- `kl_analysis/summary_A.json` / `summary_B.json` — 数值
- `scripts/analyze_kstep_kl.py` — K-step Pearson + variance ratio
- `scripts/analyze_kstep_kl_over_time.py` — 时序 K-step
- `scripts/analyze_kstep_sum_vs_mean.py` — sum vs mean 对比（新）
- `scripts/extract_opd_trajectories.py` — 从 driver log 提 metrics
- `scripts/plot_opd_trajectories.py` — 训练曲线 plot
- `NAMING.md` — 命名 mapping 文档
- `configs/opd-4b-A.env` / `opd-4b-B.env` — 新名 config（旧的保留）

### 未做（下次）

1. **opd-4b-B 中间 ckpt eval**: iter_199 / 399 / 499，画完整 7 点 AIME pass@1 vs rollout 曲线
2. **opd-4b-A driver log 找不到** → trajectory 只能用 KL dump 采样近似（每 10 rollouts 1 个点）。教训：以后训练永远 `2>&1 | tee /workspace/.../logs/<run>.log` 避免 Ray /tmp 自清
3. **opd-4b-C 实验设计待定**：mean K=8 (slime 现成) vs sum K=8 (需改 slime) vs suffix γ=0.95
4. **更激进的方向**：换 32B teacher（gap 13pt → 25pt），重做 SFT (300K data × 3000 step multi-node)


---

## 2026-05-31 (Day 31) — OPD 文献 AIME 分数对标

### 动机

确认"我们的 OPD 结果在文献里处于什么水平"，给后续决策(修 SFT vs 调 OPD vs 换 teacher)一个外部标尺。搜了 6 篇 OPD/distillation 论文，提取 AIME 分数。

### ⚠️ Eval 协议差异巨大，不能直接横比

各家 n_samples / max_tokens / thinking-mode / temperature 都不同，绝对分能差 10-30pt。逐篇标注协议。**我们的协议：n=16, max=30K, thinking, T=0.6, top-p=0.95。**

### 六篇 OPD 论文 AIME 对照

#### 1. Lightning OPD (NVIDIA, arXiv 2604.13010) — 我们的对标
协议：n=32, T=0.6, top-p=0.95, max=32768, thinking

| 配置 | AIME24 | AIME25 |
|---|:---:|:---:|
| 4B-Base SFT (teacher Qwen3-8B) | 56.7 | 52.1 |
| 4B + standard OPD | 65.4 | 57.9 |
| 4B + Lightning OPD | **68.1** | 58.4 |
| 8B-Base SFT (teacher Qwen3-32B) | 63.7 | 51.7 |
| 8B + standard OPD | 68.5 | 59.0 |
| 8B + Lightning OPD | **69.9** | 59.2 |
| 4B ExOPD baseline (他们复现) | 61.0 | 56.0 |

SFT hyperparams: lr=8e-5, 3000 steps, global batch 256(4B)/128(8B), max_len=16384, cosine, warmup 0.1, packing。OPD: 150 steps, batch 256, max_resp=4096, lr=2e-6 constant, T=0.8, top-p=1.0, β2=0.98。MoE Qwen3-30B-A3B → AIME24 = 71.0(单 node)。

#### 2. ExOPD / G-OPD (arXiv 2602.12125, reward extrapolation λ>1)
协议：n=32, T=1.0, top-p=1.0, max=16384, **non-thinking**, verl 框架

| 配置 (strong-to-weak, teacher Qwen3-30B-A3B-Instruct=74.7) | AIME24 | AIME25 |
|---|:---:|:---:|
| Qwen3-4B-NonThinking + OPD | 55.0 | 48.0 |
| Qwen3-4B-NonThinking + ExOPD (λ=1.25) | **58.7** | **50.8** |
| Qwen3-1.7B-NonThinking + OPD | 33.0 | 28.7 |
| Qwen3-1.7B-NonThinking + ExOPD | **37.3** | — |

要点：OPD 净增益 +2~4pt；reward extrapolation(λ=1.25)稳定再加 +3pt，same-size 多 teacher 设定下能超过 teacher。和 Lightning OPD 的核心张力：ExOPD 主张 λ>1 突破 teacher，Lightning 主张 teacher consistency + 1-step 够用。

#### 3. Entropy-Aware OPD / EOPD (arXiv 2603.07079, ICLR 2026)
协议：**Avg@8 / Pass@8**, T=1.0, top-p=0.8, max=**8192**, teacher Qwen3-8B **non-thinking**, verl

| Qwen3-4B-Base (DAPO data) | AIME24 | AIME25 |
|---|:---:|:---:|
| OPD (Avg@8) | 18.3 | 12.1 |
| EOPD (Avg@8) | 17.9 | **17.5** |
| OPD (Pass@8) | 26.7 | 30.0 |
| EOPD (Pass@8) | **36.7** | **33.3** |

绝对分低是因为 non-thinking teacher + max=8192(thinking 不够)。价值在诊断："reverse KL 在高熵 token 上不稳定，student top-1 频繁跳变"——和我们 5/29 KL dump 分析方向一致。增益主要体现在 Pass@8(diversity)。4B Pass@8 比 baseline OPD +5.05。

#### 4. Thinking Machines Lab OPD blog + Qwen3 Tech Report (祖师爷)
Qwen3-8B-Base student, Qwen3-32B teacher, OpenThoughts3 SFT

| 配置 | AIME24 |
|---|:---:|
| SFT-400K | 60 |
| + On-policy distillation (150 steps ≈ 77K prompts) | **70** |
| SFT-2M (extrapolated) | ~70 |
| Qwen3 报告: SFT | 55.0 |
| Qwen3 报告: + RL (17,920 GPU-hr) | 67.6 |
| Qwen3 报告: + OPD (1,800 GPU-hr) | **74.4** |

要点：OPD 比 RL 便宜 ~10x 且分更高(74.4 vs 67.6)。Lightning OPD 点名这条 pipeline teacher-inconsistent(SFT 用 QwQ-32B 生成的 OpenThoughts3，OPD 用 Qwen3-32B)。

#### 5. Adaptive Teacher Exposure / ATESD (arXiv 2605.11458, self-distillation)
Avg@12, Qwen3-{1.7B, 4B, 8B}, AIME24/25 + HMMT25

比 OPSD baseline 提升 Avg@12 **+0.95 / +2.05 / +2.33**(1.7B/4B/8B)。self-distillation(teacher = 自己 conditioned on reference)，无外部强 teacher。绝对分未取到，增量明确。

#### 6. DED Data-Efficient Distillation (arXiv 2508.09883) — off-policy 对照
pass@1 over 16 runs, DS-R1-Distill-Qwen-32B student, 仅 800 条数据

AIME24 **81.87** / AIME25 77.29。**注意：这是 off-policy SFT distillation 不是 OPD**，放这里当"蒸馏天花板"参照：32B + 顶级 teacher(QwQ-32B/R1)能到 80%+。

### 横向结论：同量级 OPD 后 AIME24 的"健康"水平

拉齐可比项(Qwen3-4B/8B, thinking, 强 teacher)：

| Student | 方法 | AIME24 健康水平 |
|---|---|:---:|
| Qwen3-4B | SFT baseline | 56-57 |
| Qwen3-4B | + OPD / ExOPD / Lightning | **65-68** |
| Qwen3-8B | SFT baseline | 60-64 |
| Qwen3-8B | + OPD | **68-74** |

### 我们的位置

我们：4B SFT 48.8 → OPD 55.8 (opd-4b-B iter_299, n=16)

1. **OPD 相对增益(+7pt)和文献完全一致** — 各家 OPD 净增益普遍 +5~+10pt(Lightning +8.7, TML +10, ExOPD +3~4, EOPD Pass@8 +5)。证明 OPD pipeline 本身 work。
2. **绝对分落后 ~10pt，瓶颈全在 SFT baseline**(48.8 vs 文献 56-57)。多家(Lightning/ExOPD/TML)交叉验证：4B+8B teacher 这个 gap 下，OPD 后落点 65-68。
3. **结论不变**：要够到文献线，核心动作是修 SFT(补数据量到 ~300K + 不混温度)，不是继续调 OPD 超参。和 5/29 Next Actions 中优先级第 3 条一致。

### 一个值得注意的方法论分歧

- **Lightning OPD**：teacher consistency + 1-step OPD 已足够，不需要 cumulative/extrapolation。
- **ExOPD**：λ>1 reward extrapolation 能突破 teacher 上限(我们 opd-4b-C 可考虑的方向之一，slime 现成 `opd_kl_coef` 调大近似)。
- **EOPD**：高熵 token 上 reverse-KL 不稳定，需要混 forward-KL。
- 这三条是 OPD 当前的主要改进轴。我们的 K-step cumulative 是第四条(RL reward-to-go 视角)，文献里还没人正面做过 —— 是潜在 novelty 点。

### 来源

| 论文 | arXiv | 关键数字 |
|---|---|---|
| Lightning OPD | 2604.13010 | 4B OPD 65.4/Lightning 68.1; 8B 69.9 |
| ExOPD / G-OPD | 2602.12125 | 4B-NT OPD 55.0 → ExOPD 58.7 |
| EOPD | 2603.07079 | 4B Pass@8 OPD 26.7 → 36.7 |
| TML OPD blog | (thinkingmachines.ai) | 8B SFT 60 → OPD 70; Qwen3 报告 OPD 74.4 |
| ATESD | 2605.11458 | 4B self-distill +2.05 Avg@12 |
| DED (off-policy) | 2508.09883 | 32B 800-sample 81.87(天花板参照) |


---

## 2026-05-31 (Day 31, 续) — FIPO 补充（与我们 K-step cumulative 高度相关 ⭐）

### 一句话

FIPO (arXiv 2603.19835, QwenPilot/Alibaba) **不是 OPD**，是 GRPO/DAPO 路线的纯 RLVR 算法，但它的核心机制 = **discounted Future-KL reward-to-go credit assignment**，跟我们 K-step cumulative OPD 几乎是同一个数学想法套在不同 advantage 上。是目前文献里和我们方向最接近的一篇，必须重点记录。

### FIPO 是什么

- **Future-KL Influenced Policy Optimization**：在 DAPO 的 token-level loss 上，用「未来轨迹的 discounted Δlogp 累积」重新加权每个 token 的 advantage。
- 原子信号 `Δlogp_t = logπ_θ(o_t) - logπ_old(o_t)`（probability shift，不是 teacher-student KL）。
- Future-KL：`FutureKL_t = Σ_{k=t}^{T} M_k · γ^{k-t} · Δlogp_k`
  - `γ = 2^{-1/τ}`，τ = half-life（实验用 τ=32）→ **soft decay window**，不是 hard truncation
  - `M_k` = dual-clip mask，IS ratio 超过阈值 c(≥10) 的 token 从未来累积里剔除（防 variance 爆炸）
- 再 `f_t = clip(exp(FutureKL_t), 1-ε_low, 1+ε_high)`，`Ã_t = Â_t · f_t`（乘性 reweight，clip 到 [0.8,1.2] for 32B / [1,1.2]）
- 负 advantage + 大 IS ratio 的 token 直接 reset `f_t=1`。

### FIPO 的 AIME 结果

协议：Avg@32, T=1.0, top-p=0.7, max_resp=20480(overlong penalty >16384)。Backbone = **Qwen2.5-32B-Base**（clean base，无 long-CoT SFT），DAPO-17K 数据，verl。

| Method | AIME24 Avg@32 | Cons@32 | Pass@32 | AIME25 Avg@32 |
|---|:---:|:---:|:---:|:---:|
| DAPO (baseline) | 50.0 | 60.0 | 80.0 | 38.0 |
| **FIPO** | **56.0** (peak 58.0) | 73.0 | 83.0 | **43.0** |
| 参考 DeepSeek-R1-Zero-Math-32B | ~47.0 | | | |
| 参考 o1-mini | ~56.0 | | | |

7B (Qwen2.5-7B-MATH): GRPO 22 → DAPO 36 → **FIPO 40** (AIME24)。

核心现象：FIPO 打破 DAPO 的 "length plateau"（4K→10K+ tokens），length 和 accuracy 强正相关，gradient norm 更稳。

### 为什么对我们重要

| 维度 | FIPO | 我们的 K-step cumulative OPD |
|---|---|---|
| 范式 | RLVR (GRPO/DAPO, outcome reward) | OPD (teacher-student distillation) |
| 原子信号 | `Δlogp = logπ_θ - logπ_old`（policy shift） | `reverse_kl = logπ_S - logπ_T`（teacher gap） |
| 未来累积 | `Σ γ^{k-t} Δlogp_k`（discounted suffix sum） | `Σ γ^d reverse_kl[t+d]`（我们的 cumulative v2） |
| 加权方式 | 乘性 `Ã = Â · clip(exp(FutureKL))` | 加性 `A_t = A_base - λ Σ ...`（reward-to-go 减项） |
| discount | `γ=2^{-1/τ}`, τ=32 half-life | γ=0.99, horizon K（我们试过 K=2/8） |
| 稳定化 | dual-clip mask + influence clip | mean 归一化 (÷K) |
| 结果 | AIME +6pt (32B, RL) | AIME +7pt (4B, OPD) |

**关键 takeaways：**

1. **"future-discounted credit assignment 有效" 在 RL 侧已被 FIPO 验证**（+6pt on 32B）。我们在 OPD 侧做同样的事，是合理的平行迁移。FIPO 可作为我们 cumulative OPD 的最强 motivation citation。

2. **FIPO 用乘性 reweight + exp 映射 + influence clip，我们用加性 reward-to-go + mean 归一化**。FIPO 的 dual-clip mask（剔除高 IS ratio token 的未来累积）是我们没做的稳定化 trick —— 如果 opd-4b-C 做 sum 版 reward-to-go 遇到 variance 爆炸（5/29 分析显示 sum variance ∝K，full 时 368,000×），可以借鉴 FIPO 的 masking。

3. **soft decay window (γ=2^{-1/τ}) vs 我们的 hard horizon K**：FIPO 明确论证 soft decay 比 hard truncation 好（无 boundary artifact）。我们 cumulative v2 是 hard horizon + mean，可以考虑换成 FIPO 式 exponential decay。我们 5/29 的 KL dump 分析也显示 K=8 是 sweet spot → τ≈8-12 的 half-life 可能对应。

4. **FIPO 的 novelty 边界**：它明确说自己是 RLVR、advantage 来自 outcome reward、Δlogp 是 policy self-shift。**它没有 teacher**。我们的 reverse_kl 用的是外部 teacher 的 logprob。所以 "K-step future-teacher-KL reward-to-go for OPD" 这个具体组合 **仍然是空白**，FIPO 不构成 prior art 抢占，反而是同思想跨范式的佐证。

### 更新后的方法论分歧地图（含 FIPO）

OPD/RL credit assignment 改进轴：
- **Lightning OPD**：teacher consistency + 1-step 够用（反对 cumulative）
- **ExOPD**：λ>1 reward extrapolation 突破 teacher
- **EOPD**：high-entropy token 上混 forward-KL
- **FIPO**：discounted Future-KL 重加权（RL 侧的 reward-to-go）← 和我们最近
- **我们 (K-step cumulative OPD)**：discounted future-teacher-KL reward-to-go（OPD 侧，文献空白）

### 来源

| 论文 | arXiv | 关键数字 / 关系 |
|---|---|---|
| FIPO | 2603.19835 | 32B DAPO 50 → FIPO 56(peak 58); future-KL reward-to-go，RL 侧，与我们 K-step 同构 |

> 注意：FIPO 的绝对分(Qwen2.5-32B-Base + RLVR)和我们(Qwen3-4B + OPD)不可直接比，它在表里的角色是 **方法论近邻 / motivation citation**，不是 SFT-baseline 标尺那类。


### FIPO trick 已写入 research-plan.md

FIPO 的三个 trick 整合进 `research-plan.md` 的 Phase 3（Advanced Variants）：
- **新增 variant P7 (5.5b 节)**：soft decay window + dual-clip mask + 乘性 reweight，三个正交 trick + P7 子实验矩阵
- **Trick 2 (dual-clip mask) 是重点** ⭐：剔除高 IS-ratio outlier token 的 future 累积，是让 sum-form reward-to-go(去掉 ÷K)在我们这边变可行的关键前提，直接对应 5/29 发现的 sum variance ∝K(full 368,000×)爆炸问题
- **Trick 3 (乘性 reweight) 有 caveat**：纯 OPD reward=0、A_base≈0 时失效(0 乘任何数=0)，只在 OPD+verifiable reward 混合(A_base≠0)时有意义 → 顺带定义了一个 OPD+RLVR 混合 advantage 的新设定(Phase 4 stretch)
- 同步更新了风险表(variance 爆炸行加 dual-clip mask 缓解)和参考文献附录(FIPO 条目)

详见 research-plan.md 5.5b / 5.6 / 9 / 附录 B。


---

## 2026-05-31 (Day 31, 续2) — Revisiting OPD (arXiv 2603.25562) 深读 ⚠️ 对我们方向是 double-edged

### 一句话

这篇(CASIA, Fu et al., 2026-03)**直接研究了 discounted return-to-go OPD 估计量**——和我们 K-step cumulative 数学上是同一个东西——并给出了**对"加大 future coupling"的理论 + toy 实验反对证据**。同时它的三个 failure mode 和 top-K local support fix 又和我们的诊断/设计高度互补。必须当 prior art 认真对待，不能只当普通参考。

### 它和我们 K-step cumulative 是同一个估计量

它定义 discounted return-to-go gradient estimator（Eq. 3）：
```
ĝ_γ = Σ_t (Σ_{t'≥t} γ^{t'-t} r_{t'}) g_t,   γ∈[0,1]
```
- γ=0 → token-level OPD（slime 现在的 instant，= 我们的 K=1）
- γ=1 → 完整 sequence-level causal reverse-KL（= 我们 cumulative 的 full-horizon 极限）
- **这就是我们 cumulative OPD v2 在做的事**（我们用 truncated horizon K + mean 归一化，他们用 γ 连续插值）

### ⚠️ 它的核心结论对我们不利

**理论(Appendix B)**：
| 估计量 | bias | worst-case variance |
|---|---|---|
| token-level (γ=0) | biased vs sequence | **O(T²)** |
| sequence-level (γ=1) | unbiased | **O(T⁴)** |

long-horizon(我们 4K-30K token reasoning)下，T 很大 → O(T⁴) vs O(T²) 差距巨大。这和我们 5/29 KL dump 算出的 "sum variance ∝K，full 时 368,000×" **完全一致**——只是他们从 worst-case bound 角度证明，我们从实测 dump 角度验证。两边互相印证。

**Toy 实验(Fig 1, Appendix C)**：2-task 1D 连续控制，distill REINFORCE teacher。结论：
- γ 越大 → gradient variance 越高且持续
- **γ=1 时 policy 直接 drift，不收敛到 target**
- γ=0 (token-level) 稳定收敛

→ **直接证据：naive 加大 future coupling 会害了 optimization**。这正是我们 research-plan 1.4 Go/No-Go 担心的事，也是 MiniLLM/原始担忧的复现。

### 但这不是判我们死刑，而是收窄了 novelty 空间

关键：**他们没有"修好" future coupling，而是绕开它**——退回 token-level(γ=0)，转而改善"单步比较的质量"(top-K local support 替代 sampled-token)。也就是说：

- 他们的结论是 "future coupling 不值得，把 local signal 做好更重要"
- 我们 research-plan 的核心假设(1.3)恰恰是 "**moderate lookahead helps ONLY when teacher trustworthy**" + adaptive gating
- **所以我们的差异化 novelty 必须是**：不是 naive fixed-γ(他们已证否)，而是 **conditional/gated lookahead** —— 只在 teacher 可信、variance 可控时才耦合 future。如果我们也只做 fixed-k sweep 然后发现不行，就是重复他们的 negative result，没有发表价值。

### 三个 failure mode（和我们诊断互补，值得直接复用）

他们识别 sampled-token OPD 三个 failure mode，全部和我们观察吻合：

1. **Imbalanced one-token signal**：大多数 sampled token 拿负 reward，正信号集中在少数 token → 训练被 filler/hesitation token 带偏。(我们没系统量化过，可加进 diagnostics)
2. **Teacher unreliable on student prefixes**：student 进入 teacher 罕见的 prefix(repetition loop / self-reset / 退化)时，teacher 仍给高概率 → **不惩罚坏行为**。Fig 4 显示 teacher-student log-prob gap **随 position 变宽、后段更 noisy**。→ **这正是我们 5/29 "尾部 KL" 想验证但因没存 token_ids 没做成的分析！他们做出来了。**
3. **Tokenizer/special-token mismatch**：`<think>` 被切成 `<,think,>` vs teacher 的 `<th,ink,>` → 语义对但 teacher 给低概率，污染 reward。

### 他们的 fix：Teacher top-K local support matching

不奖励单个 sampled token，而是在 teacher 的 top-K support 上做 truncated reverse-KL（renormalize 两边分布）+ 三个工程 trick：
- **Support-set renormalization**（必须，否则崩）
- **Top-p rollout sampling**（让 rollout 留在 teacher 可信区）
- **Special-token masking**（消除 tokenizer artifact）

结果(Qwen2.5-7B-It student, OpenThinker3-7B teacher, DAPO-Math)：sampled-token 36.4 → +masking 40.7 → 他们方法 41.5。注意 **special-token masking 单独就 +4.3pt**，说明 tokenizer artifact 是大头。

### 对我们的具体启发 / action items

1. **research-plan novelty 重新定位**(高优先级)：必须强调 conditional/gated/adaptive，明确把 "naive fixed-γ return-to-go" 标成 already-refuted-by-2603.25562。我们的 Go/No-Go(1.4) 要直接 cite 这篇当 baseline 反例。

2. **P6 (top-K local support) 升级为高优先级**：我们 research-plan 里 top-K 是 P6(最低)。这篇证明 top-K local support 是 sampled-token 之外最 work 的方向，且和 K-step 正交。**可以组合**："top-K local support 做好单步 + adaptive lookahead 选择性耦合 future" —— 这是没人做过的组合，比单纯 K-step 更有故事。

3. **直接复用他们的 diagnostics**：
   - teacher-student log-prob gap vs position 分布(Fig 4) —— 我们 5/29 想做没做成的，下次 KL dump **务必存 token_ids + position**
   - sampled-token reward 正负比例(Fig 2) —— 量化 imbalance
   - repetition loop 时 teacher 是否仍给高概率(Fig 3) —— 验证 "teacher 不惩罚坏行为"

4. **special-token masking 必查**：他们 +4.3pt 来自 masking。我们 Qwen3-4B/8B same-family(tokenizer 一致)，理论上没这问题，但**值得确认 `<think>`/`</think>` 在 teacher logprob 计算时没被 OPD 当 outlier**。如果我们 OPD reverse_kl 在 think-tag 位置有 spike，可能就是这个。

5. **变量命名/口径对齐**：他们的 γ-return 和我们 cumulative 完全对应，写 paper 时记号要对齐，避免 reviewer 觉得我们没意识到这篇。

### 与 FIPO 的关系（重要对比）

- **FIPO**：用 future-KL reweight，在 **RLVR(有 outcome reward)** 上 +6pt → "future coupling 有用"
- **Revisiting OPD**：在 **纯 OPD(teacher reverse-KL)** 上，future coupling 增 variance、有害 → "退回 token-level"
- **矛盾的根源**：FIPO 的 base advantage 来自 verifiable reward（信号干净），future-KL 只是 reweight；Revisiting OPD 的 future term 本身就是 noisy 的 teacher reverse-KL 累积。**这恰好支持我们 P7 Trick 3 的判断**：乘性 reweight(FIPO 式)在有 outcome reward 时才 work，纯 OPD 加性累积(Revisiting 式)会 variance 爆炸。
- → **我们 opd-4b-C 的最强假设**：future-teacher-KL 只有在 (a) 加 dual-clip mask 控 variance，且/或 (b) 配 verifiable reward 当 base advantage 时才可能 work。两篇文献从相反方向夹出了这个结论。

### 来源

| 论文 | arXiv | 对我们的角色 |
|---|---|---|
| Revisiting OPD | 2603.25562 | ⚠️ 我们 K-step 方向的最强反例(fixed-γ 有害) + 互补 fix(top-K local support) + 现成 diagnostics。必引、必差异化 |


### research-plan.md 已按 2603.25562 重构方向（4 处改动）

依据 Revisiting OPD 的反例 + FIPO 的互补结论，更新了 `research-plan.md`：

1. **§1.3 核心假设**：加 ⚠️ 边界声明——naive uniform fixed-k/fixed-γ 已被证否（O(T⁴) variance + toy γ=1 drift），立足点收窄到 conditional/gated lookahead + 先做干净单步。
2. **§1.4 + §4.7 Go/No-Go**：基线对手从 naive k=1 升级为 **k=1 + top-K local support**。fixed-k 赢 naive k=1 不算 Go，真门槛是 adaptive 赢"做好的单步"。
3. **§5.1/§5.2 重心调整**：top-K local support 从 P6 升到 **P0**，新增 P0 方法详节 + **P0+ 组合 main method**（top-K 修单步 + adaptive lookahead 选择性耦合 future，文献空白）。实验矩阵/交付物相应更新。
4. **§3.3 Diagnostics**：加 E 类 failure-mode 诊断（复用他们 Fig 2/3/4）+ 写死"KL dump 必须存 token_ids + position"（修 5/29 踩坑）。

**成本影响**：P0 (top-K local support) 需在 slime rollout 取 teacher top-K logprobs（`top_logprobs_num`，5/11 验证 sglang 支持 K=50），工作量比纯加 cumsum 大。

**新的项目重心**：从"纯 K-step cumulative"挪到"top-K local support + adaptive/gated lookahead 组合"。两篇文献（2603.25562 反对 naive future coupling / FIPO 支持有 clean base 时的 future reweight）从相反方向夹出了这个定位。


### Phase 2.5 实验计划已定（sum vs mean 决战）

讨论后定了 K-step 这条线的核心分水岭实验，写进 `research-plan.md` §4.8（Phase 2.5）。

**核心问题**：slime cumulative v2 是 mean-K（÷K），本质偏去噪；sum-K 才是教科书 reward-to-go（variance ∝K²）。要分离 "credit assignment" vs "magnitude 稀释/去噪" 两个因素。**不用 EMA**（讨论后否决，sum vs mean 同 K 对比已经能干净分离，唯一差别是 ÷K）。

**三个已定决策**：
1. sum 的 kl_coef = instant/K 起点（量级对齐，差异归因到分配结构而非量级）。sum-K=8 用 0.125。
2. K 主测 8（5/29 已证 sweet spot），顺带 4，不铺满。
3. dual-clip mask 默认只开 sum；补 R3b=mean+mask sanity 避免 2×2 缺角。

**代码现状确认**（读了 loss.py）：
- ✅ instant / mean-K（truncated ÷actual_k）/ sum-full 已支持
- ❌ sum-K（truncated 不除）不支持 → 需加 `--opd-agg {sum,mean}`
- ❌ dual-clip mask 不支持 → 需加 `--opd-dualclip-c`，IS ratio 用 `rollout_log_probs`（已确认在 rollout_data 里）
- 回归测试硬门槛：`--opd-agg mean` 必须和现 v2 逐数值一致

**Batch 1 矩阵**（7 run，复用 opd-4b-B setup，300 rollouts/run）：R0 instant(已有) / R1 mean-K8 / R2 sum-K8 裸 / **R3 sum-K8+mask(main)** / R3b mean+mask / R4 mean-K4 / R5 sum-K4+mask / R6 sum-K8 c=5。

**判读**：R3>R1 → reward-to-go story 成立；R2 vs R3 → mask 是否救 variance；mean 赢但 sum 不赢 → 诚实改成 "denoising helps"。

**下一步动手**：先改 loss.py 3 处 + 过回归测试，再跑 Batch 1。详见 research-plan §4.8。


---

## 2026-06-01 (Day 32) — Phase 2.5 R4 (mean-K=4) 启动 + 过夜运行

### 启动经过

第一个 Phase 2.5 实验 **R4 (mean-K=4)** 在 p5-3 启动（不改 loss 主逻辑，纯用现有 `--opd-cumulative --opd-horizon 4`）。复用 opd-4b-B 配置（v2-700 student, lr=2e-6, T=0.8, max_resp=4096, 300 rollouts），外部 teacher 在 qzf-dev (Qwen3-8B, DP=4)。配置 `configs/opd-4b-R4-meanK4.env`。

### 踩坑：teacher 抖动 → slime pickle bug 杀 job（已修）

- 首次启动后 ~3min job 崩：teacher 偶发 HTTP 500 → slime `reward_func` 的 `resp.raise_for_status()` 抛 `aiohttp.ClientResponseError`，该异常带 `CIMultiDictProxy`（headers）**不可 pickle** → Ray 序列化崩 → 整个 job 死。
- teacher 端 500 根因：`mem-fraction-static=0.85` + 高并发（rollout_batch=64×n=4=256）在 L40S(46GB) 上 KV 撑爆，server 进程 `SystemExit`。
- **修复 1**：teacher 重启用保守配置 `--mem-fraction-static 0.7 --max-running-requests 24 --context-length 20480`。
- **修复 2**：patch `slime/rollout/on_policy_distillation.py` 的 `reward_func` —— 加 5 次 retry + 把任何异常转成纯 `RuntimeError`（剥掉不可 pickle 的 headers）。备份在容器 `on_policy_distillation.py.bak-*`，patch 源在 `scripts/opd_reward_hardened.py`。这个加固让 teacher 抖动不再杀 job，是过夜稳定的关键。

### 运行状态（截至 Day 32 早，rollout 143/300）

| 指标 | 值 | 解读 |
|---|---|---|
| opd_reverse_kl | ~0.10 | 和 opd-4b-B 起点 0.139 同量级，健康 |
| advantages | ~-0.103 | = -kl_coef × mean-K cumulative_kl |
| truncated | ~0.93 | 4096 cap 下高位（与 opd-4b-B 一致）|
| response_length | ~4050 | 稳定，无 length explosion |
| 速度 | ~5.5 min/rollout | 剩 ~157 rollouts → ~14h |

**关键确认**：mean-K=4 的 advantage 量级（-0.10）和 instant baseline（opd-4b-B ~-0.10）一致 → mean 归一化按预期工作，penalty 量级对齐。已存 ckpt: iter_49, iter_99（save_interval=50）。KL dump 正常（带 token_ids，r10-r140，每 10 rollouts）。

### 下一步

- 等训练到 iter_149/199/299，n=16 eval（对比 opd-4b-B baseline 48.8/iter299 55.8）
- R4 是 Phase 2.5 的 mean 半边；sum-K（R2/R3/R5/R6）和 dual-clip mask 仍需改 loss.py（加 `--opd-agg` + `--opd-dualclip-c`）


---

## 2026-06-01/02 (Day 32 续) — 训练瓶颈分析 + fast config + R1 启动

### 瓶颈分析（slime timer.py 实测，R4 单 rollout）

用 slime 的 `timer.py` 打点拆解 R4（mean-K=4, 8192, recompute on）单 rollout ~330s：

| 阶段 | 耗时 | 占比 | 干什么 |
|---|---|---|---|
| train_wait (rollout 生成) | ~80s | 24% | student 生成 256×4096 token + teacher logprob |
| ref_log_probs | ~38s | 11% | actor 跑 ref model logprob |
| log_probs | ~37s | 11% | actor 跑 policy logprob |
| **actor_train** | **~131s** | **40%** | **梯度 forward+backward（真瓶颈）** |
| update_weights | ~0.4s | <1% | 权重同步 |

**修正之前的误判**：瓶颈不是 rollout/teacher，而是 **actor_train（40%）**，且 actor GPU 显存只用 24-26GB/80GB（严重空闲），却开着不必要的 full recompute。

### max_tokens_per_gpu 对 RL/PPO 训练无影响（确认）

读 slime arguments.py：`--max-tokens-per-gpu` 只控制 dynamic batch 的 **micro-batch 切分**，不改 `global_batch_size`（slime 在 parse 时 assert 固定）。梯度数学等价（micro-batch 累积到同一 global batch），LayerNorm/RMSNorm 与 batch 无关，`accumulate-allreduce-grads-in-fp32` 吸收累加顺序噪声。**纯吞吐参数，调大零风险只加速。**

### fast config 加速效果（R1 实测 vs R4）

新增 `scripts/train-opd-extteacher-fast.sh`（可配置 `RECOMPUTE` / `MAX_TOKENS_PER_GPU` / `LOG_PROBS_MAX_TOKENS_PER_GPU`）。R1 用 `MAX_TOKENS_PER_GPU=32768, LOG_PROBS_MAX_TOKENS_PER_GPU=40960, RECOMPUTE=0`：

| 阶段 | R4 (8192, recompute on) | R1 (32768, recompute off) | 加速 |
|---|---|---|---|
| actor_train | ~131s | 0.9-35s（稳态 <35s） | **4-100x** |
| log_probs | ~37s | 7s | ~5x |
| ref_log_probs | ~38s | 7-31s | ~2-5x |
| train_wait (rollout) | ~80s | ~76s | 持平（外部 teacher 限制）|
| **单 rollout 总计** | **~330s** | **~120s** | **~2.7x** |

→ 关 recompute + 大 token budget 把 actor_train 从主瓶颈打掉。**新瓶颈是 train_wait（rollout 生成 ~76s）**，受外部 L40S teacher + 4096 token 限制。后续若要再快，只能换本地 H100 teacher 或减 max_response_len（后者破坏可比性，不做）。

### R1 (mean-K=8) 已启动 on p5-2

- 复用 opd-4b-B recipe + fast config，外部 teacher 共用 qzf-dev（96 并发压测零错误，验证可双训练共用）
- mean-K=8（5/29 sweet spot），300 rollouts，~120s/step → ~10h
- opd_reverse_kl=0.139（和 opd-4b-B 起点一致），健康

### 踩坑记录（p5-2 环境）

1. **断 symlink**：p5-2 的 `sft-qwen3-4b-full-v2-ckpt700` 是指向已删除目录的断链 → 删除后从 p5-3 rsync 真实 HF（8.8GB）+ torch_dist（7.5G）
2. **transformers 5.3.0 拒绝本地绝对路径**：`AutoConfig.from_pretrained("/abs/path")` 被 `validate_repo_id` 拒（`count("/")>1`）。`HF_HUB_OFFLINE` 无效。降 hub 到 0.36.2 → transformers import 崩（缺 is_offline_mode）。**解法：降 transformers 5.3.0 → 5.2.0**（megatron-bridge 要求 ≤5.2.0；sglang pin 5.3.0 但实测 5.2.0 也能 import+运行）。
3. **共享 Ray cluster**：`--net=host` 下 k-step-opd/k-step-opd-sft/slime-bh 共享同一 Ray（同 session/gcs/cluster-id）。失败 job 残留的 prestart worker（192 个）持有旧 transformers 内存镜像 → patch 不生效。解法：ray stop --force 清干净后重起，worker fresh spawn 用 5.2.0。
4. **data 路径**：p5-2 是 `/workspace/data/dapo-math-17k.jsonl`，R1 config 期望嵌套目录 → 加 symlink

### 当前运行

| 节点 | 实验 | 配置 | 速度 | 状态 |
|---|---|---|---|---|
| p5-3 | R4 (mean-K=4) | 8192, recompute on | ~330s/step | rollout ~150/300 |
| p5-2 | R1 (mean-K=8) | fast (32768, no recompute) | ~120s/step | rollout ~2/300 |
| qzf-dev | Qwen3-8B teacher | DP=4, 共用 | — | serving 两个训练 |


### 实时状态快照（Day 32, 写入时）

| 节点 | 实验 | 进度 | opd_reverse_kl | truncated | ckpts |
|---|---|---|---|---|---|
| p5-3 | R4 (mean-K=4) | rollout 184/300 | 0.102 | 0.93 | iter_49/99/149 |
| p5-2 | R1 (mean-K=8) | rollout 2/300 | 0.140 | 0.95 | (pending iter_50) |

- R4 预计 ~24h 完成（慢配置 ~330s/step，已过半）
- R1 预计 ~10h 完成（fast 配置 ~120s/step）
- 两者都健康：opd_reverse_kl 稳定、无 length explosion、truncated ~0.93-0.95（4096 cap 下符合预期，与 opd-4b-B 一致）

### 文件清单（本次新增/修改）

| 文件 | 说明 |
|---|---|
| `scripts/train-opd-extteacher-fast.sh` | fast 训练脚本（可配 RECOMPUTE / MAX_TOKENS_PER_GPU / LOG_PROBS_MAX_TOKENS_PER_GPU + HF offline env）|
| `scripts/opd_reward_hardened.py` | 加固 reward_func（retry + 剥离不可 pickle 的 ClientResponseError）|
| `configs/opd-4b-R4-meanK4.env` | R4 config（mean-K=4，外部 teacher）|
| `configs/opd-4b-R1-meanK8.env` | R1 config（mean-K=8，fast 配置）|

### 下一步

1. 等 R4/R1 出 iter_99/199/299 → n=16 eval，对比 mean-K=4 vs mean-K=8 vs baseline(opd-4b-B 48.8/55.8)
2. **趁 mean 实验在跑，改 loss.py 加 sum-K + dual-clip mask**（`--opd-agg {sum,mean}` + `--opd-dualclip-c`）→ 解锁 R2/R3/R5/R6（Phase 2.5 sum 半边，真正的 reward-to-go 主结果）
3. 回归测试：`--opd-agg mean` 必须与现 v2 数值一致


### R1 OOM 修复 + KL dump token_ids 问题（Day 32 续2）

**KL dump 检查**：R4 的 dump 在产出（72 文件），但 **token_ids 仍是空 `[]`** —— 5/23 踩坑**未修复**，position/special-token 分析还是做不了。这是 `_dump_opd_kl` 函数本身没填 token_ids 的 bug，留到改 loss.py 做 sum-K 时一起修。R4 已过半无法补救。

**R1 OOM**：`MAX_TOKENS_PER_GPU=32768 + RECOMPUTE=0` 太激进，4B actor 在第 3 个 actor_train step OOM（GPU0 仅 1.65GB free）。教训：关 recompute 后显存翻几倍，32768 token budget 对 4B 太大。
- **修复**：降到 `MAX_TOKENS_PER_GPU=16384, LOG_PROBS_MAX_TOKENS_PER_GPU=24576`（仍关 recompute）→ 稳定。
- 修正后 actor_train ~30-33s（仍比 R4 的 131s 快 ~4x），显存 GPU0-3 用 34-36GB/80GB，有余量。

**R1 稳定运行**：rollout 52/300，OOM=0，opd_reverse_kl 0.140→0.017。

**⚠️ 更正前述误判**：读 loss.py 确认 `opd_reverse_kl` log 的是 **instant per-token KL**（`reverse_kl = student_logp - teacher_logp`），不是 cumulative/mean 后的值（cumulative_kl 只进 advantage、不进 log）。所以 R4(0.10) 和 R1(0.017) **跨 K 是可比的**，之前说"K 越大 mean 后数值小所以不可比"是错的。R1 instant KL 降到 0.017 是 mean-K=8 advantage 推着 student 收敛更激进的**真实信号**。但 5/29 教训仍成立：sampled instant KL 低 ≠ eval 好（可能 mode collapse），最终看 n=16 pass@1。

**安全 fast config 定论**：4B + 外部 teacher 下，`MAX_TOKENS_PER_GPU=16384 + RECOMPUTE=0` 是显存安全的加速点（32768 会 OOM）。已写回 `configs/opd-4b-R1-meanK8.env`。


### loss.py 改动：sum-K + dual-clip mask + token_ids 修复（Day 32 续3）

实现了 Phase 2.5 sum 半边所需的全部 loss.py 改动（本地 repo，已过回归测试，待部署）：

**1. `--opd-agg {mean,sum}`**：cumulative 模式的聚合方式。mean=除以 K（v2 默认，去噪）；sum=不除（教科书 reward-to-go，full magnitude，配小 kl_coef≈1/K）。

**2. `--opd-dualclip-c`**（FIPO trick）：>0 时，IS ratio `exp(log_probs - rollout_log_probs)` 超阈值的 token 从 cumulative 累积里 mask 掉（防 sum 的 variance 爆炸）。`rollout_log_probs` 在 advantage 阶段可用（已确认）。

**3. token_ids dump 修复**：`_dump_opd_kl` 之前找 `rollout_data["unconcat_tokens"]`（advantage 阶段不存在 → 空 `[]`），改成 `rollout_data["tokens"]`（actor.py 填的 per-sample list）。**这修了 5/23 起一直存在的 KL dump token_ids 空 bug**，后续 dump 才能做 position/special-token 分析。

**4. opd_reverse_kl log 注释**：明确它始终是 instant per-token KL，跨 K/agg 可比。

#### ⚠️ 回归测试抓到的重要发现

写回归测试（`scripts/test_opd_agg_regression.py`）对拍新旧实现，第一版**失败**——发现 **slime v2 的 full-horizon (K=-1) 路径其实是 sum（从不除），只有 truncated (K>0) 路径才除 actual_k（mean）**。即"slime v2 = mean-K"这个说法只对 truncated 成立。

修正后让 `--opd-agg mean` **严格复刻 v2**：full-horizon 不除（保持旧行为），truncated mean 除 actual_k。回归测试现在全过：
- `new(mean) == old v2`（所有 gamma/K/T 逐数值一致）✅
- `sum == mean × K`（interior token）✅
- dual-clip mask 正确剔除 outlier ✅

含义：我们 R1/R4 用的是 truncated K=4/8，所以实际是除 actual_k 的 mean，结论不受影响。但 Batch 2 若做 full-horizon sum 实验，要知道 K=-1 在 mean 模式下也不除（=sum）。

#### 待部署

- 改动在本地 `slime/slime/backends/megatron_utils/loss.py` + `slime/slime/utils/arguments.py`
- 部署到机器后需在容器内再跑一次回归 sanity（确认容器 torch 版本下数值一致）
- 然后可起 R2/R3/R5/R6（sum 半边）：R3 = sum-K8 + dualclip-c10 + kl_coef=0.125（main candidate）


---

## 2026-06-09/10 (Day ~40-41) — Phase 2.5 完整 eval + KL dump 系统性诊断

### TL;DR

1. **R1 是 Phase 2.5 winner**：mean K=8 + kl_coef=1.0（无 mask flag）AIME-24 iter299 = **59.17%**（baseline 48.75, opd-4b-B 55.83），AIME-25 = 48.33%。pass@any-24 = 83.33% 比 baseline 76.67% 还高 → **不是 mode collapse**。
2. **R3 几乎追平 R1**：sum K=8 + mask + kl_coef=0.125 数学上 interior token = R1（mean K=8 ÷K = sum K=8 / 8），eval 也几乎一致（57.92 vs 59.17）。
3. **R3b/R5/R4 末段 instant_kl ~0.10 不收敛**：mean K=4 (R4) horizon 短，mean K=8+mask flag (R3b/R5) 触发了 dualclip 代码路径副作用（即使 IS ratio < c=10 mask 数学上不触发）。
4. **训练健康度三个 concern 都有数据支撑**：(a) R1/R3 前 25 步 instant_kl 11x 暴降；(b) R1/R3b/R4 各有一个 grad_norm spike（10-12，是 median 的 ~150x），PPO clip 接住没崩；(c) R1 末段 kl_ref ≈ 0.092 vs B 的 0.07，student 离 SFT init 远。
5. **KL dump 揭示 reverse_kl 的 token-level temporal structure**：
   - **Spike clustering 真实存在**（lift(d=1) ≈ 1.5, lift(d=8) ≈ 1.22, lift(d=32) ≈ 1.14）
   - **Spike amplitude 接近独立**（ρ\|x\| ≈ 0.08）
   - active token 占 ~32%，spike spacing 中位数 = 2 token
   - K-window catch rate K=8 → 81%, K=16 → 92%, K=32 → 97%
6. **K-step OPD 的真实物理基础**：mean-K **同时**做 noise 平均 + 利用 spike clustering 的 reasoning-chunk structure，不是单纯 denoising 也不是简单 future-credit assignment。
7. **K 没有数据驱动的硬上限**：SNR 单调上升，lift 慢衰减无拐点。K=16/32/full(=4096) 都值得 sweep。
8. **mode-collapse 警报推翻**：R1/R3 instant_kl ~0.003 不是 mode collapse，是 **deeper convergence**。pass@any、pass@1（greedy）、AIME-25 全部支持。

---

### 1. Phase 2.5 完整 eval (n=16, AIME 24/25)

| run | 末段 instant_kl | iter99 AIME24 | iter199 AIME24 | **iter299 AIME24** | iter299 AIME25 | iter299 pass@any-24 |
|---|---:|---:|---:|---:|---:|---:|
| baseline (SFT v2-700) | — | — | — | 48.75 | 40.62 | 76.67 |
| opd-4b-A (instant) | 0.081 | 50.21 | — | 54.37 | 45.21 | 76.67 |
| **opd-4b-B (instant)** | 0.093 | 53.54 | — | **55.83** | 45.00 | 80.00 |
| **R1** (mean K=8, no mask) | **0.003** ⚠️ | 53.12 | 57.50 | **59.17** ⭐ | **48.33** ⭐ | **83.33** ⭐ |
| **R3** (sum K=8 + mask, kl=0.125) | **0.000** | 50.83 | 57.92 | 57.92 | 45.62 | 80.00 |
| R3b (mean K=8 + mask, kl=1.0) | 0.100 | 54.58 | 50.62 | 50.83 | 45.00 | 73.33 |
| R4 (mean K=4 no mask) | 0.100 | 54.17 | 51.04 | 55.00 | 43.96 | 76.67 |
| **R5** (mean K=8 + soft mask) | 0.10 (~172/300) | 训练中 | — | — | — | — |

eval JSONs in `kl_analysis/phase25/eval/`，汇总脚本 `scripts/extract_eval_summary.py` 和 `scripts/diversity_check.py`。

### 2. 修正：worklog 5/29 "K=8 sweet spot" 是错的

5/29 那次分析用的 SNR 定义是 "K-step KL 跟 instant KL 的相关性"，不是 mean / std。重做 **mean / std SNR(K)** 在 5 个 run 上一致：

| run | SNR(K=1) | SNR(K=4) | SNR(K=8) | SNR(K=16) | SNR(K=32) |
|---|---:|---:|---:|---:|---:|
| B | 0.18 | 0.35 | 0.47 | 0.65 | 0.87 |
| R3b | 0.18 | 0.34 | 0.47 | 0.64 | 0.86 |

**SNR 单调上升 5 倍**，K=32 / K=1 ≈ √32 = 5.66，对应近独立 noise √K 平均。**没 sweet spot**。

K=8 当时报告"sweet spot"实际是 truncation 引起的 plot artifact + 不同 SNR 定义。

### 3. KL dump 上的真实信号 structure（核心 finding）

**Sparsity / 量级**：
| 量 | 实测 |
|---|---:|
| frac(reverse_kl exact 0) | 16-23% |
| frac(\|reverse_kl\| > 0.05) | 31-34% |
| spike spacing 中位数 | 2 token |
| mean(\|reverse_kl\|) | 0.16-0.22（随训练略降）|

**Pearson autocorr 是误导的指标**：
原始 reverse_kl 的 ρ(d) ≈ 0.03-0.05 看似"独立"，但被 sparse 0 稀释（0-0 配对贡献正协方差但不反映 spike 时间结构）。\|reverse_kl\| autocorr ρ ≈ 0.08，仍然被 0-0 baseline 撑住。

**正确的 spike clustering 指标 = lift**：
`lift(d) = P(active[t+d]=1 \| active[t]=1) / P(active=1)`
- lift = 1: 独立
- lift > 1: spike clustering（reasoning chunk）
- lift < 1: spike repulsion

实测：
| run | p_active | lift(1) | lift(2) | lift(4) | lift(8) | lift(16) | lift(32) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 0.342 | **1.45** | 1.33 | 1.24 | 1.17 | 1.15 | 1.13 |
| R4 | 0.317 | **1.52** | 1.38 | 1.29 | 1.23 | 1.18 | 1.15 |
| R3b | 0.313 | **1.53** | 1.40 | 1.30 | 1.23 | 1.18 | 1.14 |
| R5 | 0.317 | **1.53** | 1.39 | 1.29 | 1.23 | 1.20 | 1.15 |

**Spike clustering 真实存在**：active token 在 reasoning chunk 内 cluster，d=1 比 random 高 50%，d=8 仍高 22%，d=32 仍高 14%。

### 4. K-window catch rate（K 的物理意义）

```
P(K-window contains ≥1 active token, threshold |x| > 0.05):
K=1   K=2   K=4   K=8   K=16  K=32  K=64
0.32  0.48  0.65  0.81  0.92  0.97  0.99
```

K=8 让 81% 的 token 拿到非零 advantage，K=1 只让 32% 拿到。**mean-K 提供了 OPD 训练 signal 的 update density**，instant-K=1 是 sparse update。

R1 的 +3.3pt over instant baseline 来自三个机制叠加：
- Update density 提升（80% vs 32% token 有非零 advantage）
- √K noise 平均（K=8 把 amplitude noise 压 2.83x）
- 利用 spike clustering 的 chunk structure（lift(8) = 1.23）

### 5. 训练阶段（early/mid/late）的 stationarity

| 量 | early | mid | late | 变化 |
|---|---:|---:|---:|---|
| frac(active) | 33-38% | 32-35% | 29-31% | 缓慢下降 |
| mean(\|x\|) | 0.20-0.23 | 0.19-0.22 | 0.16-0.17 | -20% |
| ρ\|x\|(1) | 0.08 | 0.08 | 0.08 | 不变 |
| K=8 catch rate | 84% | 83% | 80% | 略降 |

→ reverse_kl 的 temporal structure 是训练 stationary 的（lift / autocorr 不变），变的只是 amplitude（student 整体逼近 teacher）。

### 6. 三个训练健康度 concern 的具体数据

**Concern 1: 前期暴降**
| run | rollouts 0-9 instant_kl | rollouts 20-29 instant_kl | 暴降倍数 |
|---|---:|---:|---:|
| R1 | 0.137 | 0.024 | 5.7× |
| R3 | 0.136 | ~0.025 | 5.4× |
| B/R3b/R4 | 0.13 | 0.11-0.13 | <1.2× |

R1/R3 在 30 步内把 instant_kl 推下 5×；其它 run 几乎不动。

**Concern 2: grad_norm spike**
| run | median grad | spike rollout id | spike grad |
|---|---:|---:|---:|
| B | 0.394 | — | 无 spike (0) |
| R1 | 0.072 | 186, 201 | 11.59 / 4.35 |
| R3 | 0.071 | 217 | 0.41 |
| R3b | 0.072 | 213 | 10.15 |
| R4 | 0.121 | 20 | 11.15 |

每个 cumulative run 都有 1-2 个 ~10x median 的孤立 spike，PPO clip 接住没崩，但每 spike 是一次 stale gradient。

**Concern 3: low instant_kl + high kl_ref**
R1 末段：instant_kl=0.003, kl_ref=0.092。意思是 **student 在 sampled token 上贴 teacher（mode-seeking）+ 整体分布离 SFT init 远**。这是 reverse-KL OPD 的设计目标（MiniLLM §2.1 mode-seeking），不是 collapse。**diversity 实测 OK**：pass@any-24 = 83% > baseline 76%。

### 7. R3 vs R3b 的 anomaly 仍未解但故事变了

R3 (sum K=8 + mask + kl=0.125) 和 R3b (mean K=8 + mask + kl=1.0) 数学上 interior token 等价（mean = sum/K，coef 抵消 K），boundary 处差 1×→8×。R3 几乎和 R1 一致（57.92 vs 59.17），R3b 垮 8pt（50.83）。

**Hypothesis（更准确版）**：R3b 在 sequence 末尾 K-1=7 个 token 处 advantage magnitude 比 R3 大 K=8 倍 → 末尾 7 token 被过度惩罚 → 训练动态崩塌。R3 因为 sum + 小 kl_coef，末尾 token 反而 advantage **更弱**，避免了 boundary 过度惩罚。

R1 没有 mask flag、走 unpatched code path，advantage 计算最干净，所以最稳定也最强。

**真正未解的是**：R5 (mean K=8 + soft mask) 应该跟 R3b 等价（两者都 mask 不触发），但 R5 训练 trajectory 看起来跟 R3b 接近。等 R5 跑完 eval 验证。

### 8. 文献立场更新（4 篇综合）

| 文献 | 立场 | 跟 R1 +3.3pt 关系 |
|---|---|---|
| **MiniLLM** (2306.08543, OPD 起源) | future R_t 有用但需 single-step + length-norm + teacher-mix；naive R_t 会爆 | **不冲突** — R1 = mean K (≈ length-norm 近似)，正好在他们安全区 |
| **TML blog** (2026) | discount > 0 没看见 improve（脚注，无数据）| weakly conflict |
| **Revisiting OPD** (2603.25562) | fixed-γ return-to-go variance O(T⁴)，γ=1 toy drift | weakly — 他们用 raw sum 我们用 ÷K |
| **Rethinking OPD** (2604.13016) | 不讨论 future coupling，关注 thinking pattern + new knowledge + token overlap | **正交** — 我们 setup 满足两个 success condition |

R1 +3.3pt 在文献框架下是 **MiniLLM 的"安全 future" + Rethinking OPD 的"两个 success condition 满足"** 的合理产物，不是发明新 trick。

### 9. R3 / R3b / R5 的 dualclip mask 在我们 setup 是 dormant

KL dump 实测 IS ratio 分布：

| 量 | 值 |
|---|---:|
| IS ratio median | 1.000 |
| IS ratio p99 | 1.12 |
| IS ratio max | 1.4-1.8 |
| frac(IS > c=10) | **0.000%** |

slime 是 single-step on-policy GRPO，rollout 和 train 间 weight gap 极小，IS ratio 永远 ≪ 10。FIPO 的 dualclip mask 阈值 c=10 在我们 setup 下数学上**永远不触发**，hard mask / soft mask 在物理上等价 R1（无 mask）。

但 R3b vs R1 实测 trajectory 完全分叉 → mask flag 触发 code path 副作用（rollout_log_probs 被 attach 到 advantage 阶段、autograd graph 多一个 exp 节点等），不是 mask 数学起作用。

### 10. 当前未跑实验（推荐 priority）

1. **R5 跑完 eval iter99/199/299**（自动）
2. **R1 second seed 复现**（必须 — single seed +3.3pt 不够 paper-grade）
3. **K=16/32/full sweep**（验证 SNR / lift 单调预测；可能继续涨）
4. **kl_coef sweep on R1** (1.0 / 0.5 / 0.25)（验证 robustness 不依赖大 kl_coef）
5. **Single-step decomposition (MiniLLM)**：把 (∇L)_Single 加进 R1 看能否再 +1pt
6. **Top-K dump format**：补 student/teacher top-20 logp 到 dump，能算 overlap_ratio + entropy_gap（Rethinking OPD 诊断）

### 11. Paper-grade contribution 候选

1. **Reverse-KL token-level temporal structure phenomenology**：lift / autocorr / spacing 的 stationary 结构，文献空白
2. **K-window catch rate as the right K-selection metric**：物理上替代 SNR 假象，统一 instant / mean-K / sum-K
3. **R1 winning + diversity-preserved**：4B+8B 上 +3.3pt 同时 pass@any 涨，给 4B+8B teacher 这个 config 的 OPD 上限提供 baseline

### 12. 文件清单

| 文件 | 用途 |
|---|---|
| `scripts/extract_eval_summary.py` | 从 eval JSON 拉 avg_pass_at_1 / pass@any / avg_len |
| `scripts/diversity_check.py` | mode-collapse signature 表（avg ↑ + pany ↓）|
| `scripts/analyze_phase25_eval_vs_kl.py` | eval ↔ training metric 相关性 |
| `scripts/analyze_kstep_autocorr.py` | raw autocorr + SNR(K) |
| `scripts/analyze_kstep_v2.py` | sparsity-aware (\|x\| autocorr + K-window catch) |
| `scripts/analyze_kstep_lift.py` | lift = P(active\|active) / P(active) — 真正的 spike clustering 指标 |
| `scripts/analyze_kstep_per_phase.py` | 训练阶段（early/mid/late）分别算 |
| `scripts/plot_phase25_kl.py` | 6-panel vstack trajectory plot |
| `kl_analysis/phase25/dump_summary.json` | per-rollout KL dump 数值汇总 |
| `kl_analysis/phase25/kstep_v2_summary.json` | K-window catch + sparsity |
| `kl_analysis/phase25/kstep_lift_summary.json` | lift(d) 表 |
| `kl_analysis/phase25/eval/aime20{24,25}_*.json` | n=16 eval 结果 |
| `kl_analysis/phase25/phase25_trajectories_vstack.png` | 训练曲线 6-panel 图 |
| `kl_analysis/phase25/phase25_dump_diagnostic.png` | KL dump 4-panel 诊断 |
| `kl_analysis/phase25/kstep_window_coverage.png` | K-window catch rate 曲线 |
| `kl_analysis/phase25/kstep_lift.png` | lift(d) 曲线 |
| `kl_analysis/phase25/kstep_window_per_phase.png` | per-phase K-window |


---

## 2026-06-10 (Day 41 续) — Phase 3 candidate: Hybrid-K OPD with Rao-Blackwellized future

### Motivation

R1 (mean K=8 sampled) 拿到 +3.3pt over instant baseline，但这是 K-step OPD 在 sampled-$c$ estimator 下的天花板。我们 Phase 2.5 KL dump 的 lift 数据（lift(d=1)=1.5, lift(d=8)=1.22, lift(d=32)=1.14）说明 reverse-KL spike 在 token 序列里是 chunk-clustered，不是 random Poisson。这给 Rao-Blackwell future term（用 distribution-level KL 替代 sampled log-ratio）提供了机制层面的 motivation：sampled-$c$ 在 chunk 内随机抽到 trivial vs spike token，丢失 chunk-level 信号；distribution-level $d(h)$ 直接看到该位置 chunk-level disagreement。

### 目标量与 chain-rule decomposition

Sequence-level joint reverse KL：

```
KL(S_{1:T}|h || T_{1:T}|h) = sum_{i=1}^T  E_{y_{<i} ~ S} [ d(h_i) ]
where  d(h_i) = KL(S(·|h_i) || T(·|h_i))
```

每个位置 $i$ 的 contribution 是 state-level full-vocab KL $d(h_i)$。两种 unbiased estimator：

| Estimator | 公式 | RB layer | Variance source |
|---|---|---|---|
| Sampled $c_t$ | log π_S(y_t\|h_t) − log π_T(y_t\|h_t) | 无 | token sampling + path |
| Distribution $d(h_t)$ | sum_v π_S(v) [log π_S(v) − log π_T(v)] | y_t marginalized | path only |

`E_{y_t ~ π_S(·|h_t)}[c_t] = d(h_t)` → distribution estimator 是 sampled estimator 的 Rao-Blackwellization，variance ≤。

### Hybrid design

K-step cumulative advantage 每一项可选 sampled vs distribution：

| 项 | sampled $c_{t+j}$ | distribution $d(h_{t+j})$ | 选哪个 |
|---|---|---|---|
| j=0 (current) | 依赖 y_t，PG signal 保留 | 仅依赖 h_t，对 PG 是 baseline → 梯度=0 | **必须 sampled** |
| j ≥ 1 (future) | 高 variance | 通过 prefix path 间接依赖 y_t，RB 减 variance | **distribution 优** |

j=0 的 baseline 化论证：`E_{y_t}[d(h_t) · ∇log π_S(y_t)] = d(h_t) · E[∇log π_S] = 0`。如果想 RB current，必须走 direct KL loss form（不通过 PG），即 MiniLLM Eq.3 的 (∇L)_Single 项。

### 三种 form

**Form A (Pure PG hybrid)**
```
A_t = -c_t - sum_{j=1}^{K-1} γ^j · d(h_{t+j})
L_PG = -E_τ sum_t  ratio_t · A_t · clip(...)
```
j=0 sampled, j ≥ 1 RB。文献空白点。

**Form B (推荐: MiniLLM Single + RB Long)**
```
L = sum_t d(h_t)                                      # exact-grad current (MiniLLM Single)
  + E_τ sum_t  ratio_t · sum_{j=1}^{K-1} γ^j d(h_{t+j})  # PG with RB future
```
Current 和 future 都 RB 了：current 通过 direct grad（对 π_θ 的偏导），future 通过 PG with RB advantage。是 4 种 estimator 组合中 variance 最低的 unbiased 版本。

**Form C (top-K approximation)**
将 $d(h)$ 替换为 $d_K(h) = \mathrm{KL}(S_K \| T_K)$ over student/teacher top-K support（K=20 已验证 SGLang 支持），bandwidth 从 V=152K 降到 K=20，bias 来自 tail mass 但 Rethinking OPD §4.1 实测 top-K 占 97-99% mass。

### 跟现有方法的 estimator combination matrix

| 方法 | j=0 | j ≥ 1 | 标号 |
|---|---|---|---|
| MiniLLM | exact-grad d(h_t) | sampled c | (RB, sampled) |
| slime current (R1) | sampled c | sampled c | (sampled, sampled) |
| FIPO | (verifiable reward base) | sampled c + IS-mask | RLVR setup, n/a |
| Rethinking OPD top-K | top-K d_K(h_t) | (no future) | (RB, ∅) |
| **Form A hybrid** | sampled c | distribution d(h) | (sampled, RB) ⭐ 空白 |
| **Form B (推荐)** | exact-grad d(h_t) | distribution d(h) via PG | (RB, RB) ⭐⭐ 空白 |

### Bias-variance-cost

| 选项 | Bias | Variance | Compute |
|---|---|---|---|
| Sampled $c_{t+j}$ (R1) | 0 | full | 1 logp |
| Top-K $d_K(h_{t+j})$ | small (tail mass) | ≤ sampled | K logits |
| Full $d(h_{t+j})$ | 0 | strict ≤ sampled (Rao-Blackwell) | full vocab |

Variance reduction 大小取决于 $\mathrm{Var}_{y_{t+j}}[c_{t+j}]$ 占 $\mathrm{Var}_\tau[c_{t+j}]$ 的比例。Phase 2.5 数据：active% ≈ 32% + spike amplitude 接近独立 → 单步 sampling noise 占大头，**预期 RB future 的 variance 减少显著**（具体倍数需实测）。

### MVP 实验设计

复用 R1 setup（4B student v2-ckpt700 + 8B teacher + dapo-math + lr=2e-6 + max_resp=4096 + 300 rollouts + K=8 + GRPO advantage estimator）。

| variant | j=0 | j ≥ 1 | 实现改动 |
|---|---|---|---|
| **A0 = R1 baseline** | sampled c | sampled c | 无（已跑）|
| **B1 = Form A** | sampled c | top-20 d_K(h_{t+j}) | rollout 加 top_logprobs_num=20，loss.py 加 d_K branch |
| **B2 = Form B** | exact-grad d_K(h_t) | top-20 d_K(h_{t+j}) via PG | + MiniLLM Single 直接求导支线 |

每 variant 2 seeds。判读：
- B1 vs A0：纯 RB future 效应
- B2 vs B1：MiniLLM Single 边际增量
- 全部 < noise band → R1 已 saturate K-step OPD signal，承认 negative

### 实现复杂度

| 改动点 | Form A | Form B |
|---|---|---|
| SGLang rollout (top-K logp) | 已支持，开 `top_logprobs_num=20` | 同左 |
| Teacher full vocab vs top-K | top-20 即可 (cost +20-30%) | 同左 |
| Student top-K student logp | 训练时 forward 自然有 full vocab logits，取 top-K immediate | 同左 |
| loss.py 改 advantage 计算 | 加 d_K(h) 分支替代 j ≥ 1 项 | 加 single-step exact-grad 项 |
| reward / reward_func | 不动 | 不动 |

预计实现 + 单 run 训练 ≈ 1-2 天。

### Novelty 评估

| 维度 | 评分 |
|---|---|
| 数学 novelty | low — Rao-Blackwell 标准 trick |
| 文献空白程度 | medium — (sampled, RB) 和 (RB, RB) 这两个组合都没人完整写过 |
| 实证 contribution | 取决于 +pt 大小：>2pt 是 paper-grade，<1pt 是 ablation appendix |
| 跟我们 Phase 2.5 数据的 alignment | 强 — lift cluster 数据为 RB motivation 提供机制层面证据 |

### Paper story 候选

> "On-Policy Distillation Meets Temporal Credit Assignment: A Phenomenology and a Hybrid Estimator"
>
> Section 3 (Phenomenology): reverse-KL token-level temporal structure (lift, autocorr, spacing)
> Section 4 (Theory): four-way estimator matrix (sampled vs RB) × (current vs future), bias-variance-cost
> Section 5 (Method): Hybrid-K OPD = MiniLLM Single + RB Long with top-K KL
> Section 6 (Experiments): AIME 4B+8B with R1 baseline, 2 seeds × 3 variants

### 风险

1. **R1 已经 saturate**：mean-K denoising 已经把 sampled-$c$ 噪声压下来了，RB 的边际 variance reduction 可能小到 < noise band
2. **top-K bias 没控制住**：tail mass 可能比预期重要（特别是 student 在 OOD prefix 时分布扁平）
3. **Compute overhead 真实**：teacher top-20 logp 比 sampled 多 20x bandwidth，rollout 慢 30% → 同样时间内只能跑 200 rollouts 而非 300
4. **Form B 实现复杂**：MiniLLM Single 项的 exact-grad 跟 PG Long 项混合需要小心 backward graph 不互相干扰

### 跟 Phase 2.5 主线的关系

Phase 2.5 主线先收尾（R5 跑完 + eval + 2 seeds R1' 复现）。Phase 3 hybrid 是其后的延伸，**条件触发**：
- 如果 R1 second seed 复现 +3pt → Phase 3 启动 (positive base)
- 如果 R1 second seed 跌到 +0-1pt → Phase 3 暂缓，重新审视 R1 是否是 single-seed lucky run
- 如果 R5 比 R3b 显著好 → 说明 mask flag 副作用是真的，Phase 3 也要避开 mask flag

### 文件 / 待办

- 写 hybrid loss.py prototype 在 `slime/.../loss.py` 加 `--opd-future-rb` flag
- SGLang `top_logprobs_num=20` 的 teacher logp 已在 worklog 5/11 验证
- 跑通 regression test：`--opd-future-rb=False` 数值等价 R1（必须）


---

## 2026-06-11 (Day 42) — R5 eval + IS dormant 诊断 + OPD vs multi-step PPO 本质冲突

### TL;DR

1. **R5 (mean K=8 + soft mask) eval 完成**：iter299 AIME-24 = 55.20%, AIME-25 = 43.33%, pass@any-24 = 80.00%。比 R3b (50.83) 高 +4.4pt，比 R1 (59.17) 低 4pt。
2. **三个数学等价 run（R1/R3b/R5）实测 8pt spread**：mask 不触发 → 三个 run 应该逐 token advantage 一致，但 eval 落 50.83 / 55.20 / 59.17 三个不同点。这是 single-seed noise / 平台漂移 / 浮点 accumulate 顺序的总和。
3. **R5 IS 分布精确测量（29 个 dump rollouts，全训练）**：median=1.00, max=**1.80**, p99=1.12, **frac(IS>c=10) = 0.0000%**, **mean(soft_w) = 1.000000**, mean signal_loss = 0。soft mask 在 R5 严格等价 R1。
4. **OPD 不能用 multi-step PPO 是结构性原因，不是工程偶然**：reward 跟 student weight 耦合，rollout 复用违反 PPO unbiased assumption。
5. **dualclip-c=10 在 single-step OPD 是 dead code**：c=10 是 FIPO 在 multi-step PPO setting 下的合理阈值。我们 setup 下要触发 mask 必须 c ≈ 1.5，或者改 mask metric。
6. **R5 KL dump 关键差异**：R5 dump 文件 keys 多了 `rollout_log_probs`（patch_p53_softmask_logging.py 装上了 dump 那一段）。R3b/R3 dump 没存 rollout_log_probs，无法离线算 IS。

---

### 1. R5 完整 eval 结果（n=16）

| iter | AIME-24 avg_p1 | AIME-25 avg_p1 | pass@any-24 | pass@any-25 | avg_len |
|---:|---:|---:|---:|---:|---:|
| 99  | 52.71 | 42.50 | 80.00 | 70.00 | 51,006 |
| 199 | 53.75 | 44.58 | 80.00 | 73.33 | 50,545 |
| **299** | **55.21** | **43.33** | **80.00** | 66.67 | 51,778 |

eval files in `kl_analysis/phase25/eval/` (待补，p5-3 上 `/workspace/k-step-opd/eval_results_n16/`).

### 2. Phase 2.5 完整 K=8 ranking 更新

| run | dualclip flag | mask 类型 | iter299 AIME-24 | AIME-25 | pass@any-24 |
|---|---|---|---:|---:|---:|
| **R1** | 不传 | 无 | **59.17** ⭐ | **48.33** ⭐ | **83.33** |
| **R3** (sum kl=0.125) | hard c=10 | hard | 57.92 | 45.62 | 80.00 |
| **R5** | hard c=10 + soft | soft | **55.21** | 43.33 | 80.00 |
| **R3b** | hard c=10 | hard | 50.83 | 45.00 | 73.33 |

R3 跟 R3b 都 hard mask 但差 7pt。R5 soft mask 居中。**唯一不变量**是 mask 都不触发 → 数学等价。spread 8.34pt 来自 setup 之外的差异。

### 3. R5 IS ratio 完整诊断（29 dump rollouts × ~131K tokens）

| 量 | 全训练 aggregate |
|---|---:|
| Median IS ratio | **1.0000** |
| Mean IS ratio | 1.0000 |
| p99 IS | 1.12 |
| p99.9 IS | 1.31 (early) → 1.17 (late) |
| **Max IS over 整训练** | **1.80** |
| frac(IS > 1.5) | 0.0000% |
| frac(IS > 2) | 0.0000% |
| frac(IS > 5) | 0.0000% |
| **frac(IS > c=10)** | **0.0000%** |
| **frac(soft_w < 1)** | **0.0000%** |
| Mean soft_w | **1.000000** |
| Worst soft_w_min | 1.0000 |
| Mean signal_loss = 1−mean(soft_w) | **0.000000** |

→ R5 soft mask 数学上**严格逐 token 等价 R1**。mean(soft_w)=1.000000 不是近似，是真的没有任何 token 被 attenuated。

数据 / 图 / 脚本：
- `kl_analysis/phase25/r5_is_ratio_summary.json`
- `kl_analysis/phase25/r5_is_ratio_diagnostic.png`
- `scripts/analyze_r5_is_ratio.py`

### 4. 数学等价 run 的 8pt spread = single-seed noise

R1 / R3b / R5 在 mask 不触发条件下 advantage 计算逐 token 一致，但 eval：
- R1 (p5-2, 6/2, unpatched loss.py) → **59.17**
- R5 (p5-3, 6/9, patched loss.py + soft flag) → **55.21**
- R3b (p5-3, 6/7, patched loss.py + hard flag) → **50.83**

差异变量：
- 平台不同（p5-2 vs p5-3）
- patched vs unpatched loss.py（前者多了 dump 路径、agg/mask 分支节点 → autograd graph 不同 → 浮点 op 顺序漂移）
- 训练时机不同（容器 / 共享 Ray cluster 状态）
- seed 不同（R1 是 6/2 启动的 seed，R3b/R5 都是 6/7-9 启动）

任何一个都可能贡献几 pt eval 漂移。8pt spread 在 single-seed eval ±2-3pt noise band 上叠加多个漂移源，**正常**。

**含义**：
- "R1 +3.34pt over baseline" 这个 finding **不能只靠 single seed claim**
- 真正的 effect size 可能在 ±4pt 内不可分辨
- 必须 multi-seed 复现才能写 paper

### 5. OPD ≠ multi-step PPO（理论上不能直接搬）

为什么 slime / TML / MiniLLM 全是 single-step PPO（rollout 1 次 update 1 次），而不是标准 multi-step PPO（rollout 1 次 update 多次）：

**根本原因**：reward 跟 student weight 耦合
- RLVR (GRPO/DAPO): reward = 0/1 verify(answer)，**不依赖 student weight**。multi-step PPO 修正 IS 后多次 update unbiased。
- OPD: reverse_kl[t] = log π_S(y_t) − log π_T(y_t)，**显式依赖 student weight**。第 1 次 update 后 student 变了，reverse_kl 也变了 — multi-step PPO 假设 reward fixed 不成立。

**5 个具体冲突**：
1. **数学不 well-defined**：multi-step IS correction 假设 r 跟 θ 独立。OPD 的 r 就是 θ 的函数。
2. **Reverse KL mode-seeking 需 fresh rollout**：KL(S || T) 是 expectation under S。多步 update 后 S 变了，旧 sample 不再 from current S。
3. **Teacher forward 不能省**：OPD 主成本是 teacher logp（teacher 通常比 student 大）。multi-step PPO 设计目标是**省 rollout**，但 OPD 必须每次 update 重算 teacher logp（理由 1） → 不省反增。
4. **PPO clip 跟 OPD mode-seeking 对冲**：clip 限制 student 在某 token 上的概率剧烈改变；OPD mode-seeking 正是要把概率集中。multi-step 真触发 clip 时压抑 OPD signal。
5. **工程惯性 follow RLHF**：MiniLLM/slime/VeRL 把 OPD 当 RLVR 换 reward 的特例，复用 PPO trainer 但 single-step。

**理论 workarounds**（没人实操）：
- Lightning OPD 的 offline teacher logp pre-compute，本质还是 single-step on student weights，每 epoch 重新算 student logp
- "K-step PPO with reward recompute" 数学上重写 reverse_kl IS correction 复杂度爆炸

→ multi-step OPD 在 paper-grade 实现上是 unsolved territory。

### 6. dualclip-c=10 在 single-step OPD 是 dead code

FIPO (arXiv 2603.19835) 在 Qwen2.5-32B + DAPO 上跑 multi-step PPO，IS ratio 真能跑到 5-50（policy drift 大），c=10 是合理阈值。

我们 single-step setup 下 IS 完全是数值精度残差（fp16 rollout vs bf16 train、flash attn vs SP-attn、dynamic batching shape 差异），**真实 policy drift = 0**：
- max IS = 1.80（exp(0.59)），对应 logp 差 0.59
- 这 0.59 是 SGLang vs Megatron 同 weight 不同 backend 的浮点漂移
- **不是 PPO 真的在 step 之间 drift**

→ dualclip-c=10 永远不触发 → mask 是 dead code。R3 / R3b / R5 都用了这 flag 但实测都不变 advantage。

### 7. 让 mask 真触发的几条路径

1. **降 c**：c=1.3 或 1.5 让数值噪声过阈触发。但这就脱离 FIPO motivation，变成"剔除 fp 漂移大的 token"，paper 故事弱。
2. **改 mask metric**（OPD-native）：
   - **reverse_kl 阈值**：剔除单 token \|reverse_kl\| > τ 的 outlier。**直接对 OPD 信号自身的噪声**。
   - **Teacher confidence**：mask = 1[H(π_T) ≤ τ]。剔除 teacher 自己不自信位置的 noisy reward。Revisiting OPD (2603.25562) 的 Fig.4 发现 teacher 在 student 长 prefix 上越来越不可信，对应这条。
   - **Token entropy / decision token gating**：剔除高熵 decision token，跟 EOPD (arXiv 2603.07079) 思路对接。
3. **改成 multi-step PPO**：理论冲突（理由见上），不推荐。

→ **OPD-native mask metric** 是合理的 paper-grade 改进方向。当前 phase 2.5 的 R3/R3b/R5 用 PPO IS 当 metric，是**抄 FIPO 没考虑 setup 差异**的失误。

### 8. 跟 Phase 3 candidate 的关系

Phase 3 (Day 41) Hybrid-K OPD with RB future term 不依赖 mask trigger，跟 IS dormant 这件事正交。但本日发现的 8pt spread 直接 raise 一个**pre-condition**：

**Phase 3 必须 multi-seed**（≥ 2 seeds × 3 variants）。single-seed 的 RB future variance reduction (+1-2pt 预期) 完全在 noise band 内不可见。

加进 Phase 3 风险表：
- **Risk 0 (新 highest)**: single-seed 8pt spread 表明 4B+8B OPD 的 platform/seed/code-version 漂移就 ±4pt。任何 Phase 3 改进必须 ≥ 2 seeds × 同平台同 code 版本，否则结论不可信。

### 9. R5 的另一个 paper-relevant finding

R5 vs R3b 都 mean K=8 + dualclip flag (mask 不触发)，但 R5 = 55.21 vs R3b = 50.83，**差 +4.4pt**。两者唯一差是 hard 还是 soft mask 写法。code path：

- R3b hard: `keep_mask = (is_ratio <= dualclip_c).to(reverse_kl.dtype); masked_kl = reverse_kl * keep_mask`
- R5 soft: `soft_weight = clamp(c / clamp(IS,1), 1); masked_kl = reverse_kl * soft_weight`

两个 code path 的输出在 IS<=10 时**逐元素相等**（hard keep_mask=1, soft weight=1）。但 autograd graph：
- hard: bool cast → no grad
- soft: clamp + div + clamp → 有 grad（虽然 ratio 是 detached 不会回 student，但 autograd 节点存在）

→ 这两个看起来等价的 code path 通过 autograd graph 浮点积累的差别在 300 rollouts 累成 4pt。这是 **slime 这个 PyTorch + Megatron + SGLang 多组件 stack 在 single-seed 下的真实 sensitivity**。Paper 写不写？写它 = "single-seed OPD eval 不可信"是 negative finding，但 actionable。

### 10. 推荐 next steps（更新后的 priority）

1. **R1 second seed 复现**（最高优先级）：在 p5-3 用 patched loss.py + 不传 dualclip flag 跑 R1' (seed=B)。这是 8pt spread 谜的关键诊断。
   - 如果 R1' ≈ R1 (59.17) → 8pt spread 来自 mask flag 触发 code path 副作用
   - 如果 R1' ≈ R5 (55.21) → 8pt spread 来自 platform / code 版本，R1=59.17 是 single-seed luck
   - 如果 R1' ≈ R3b (50.83) → R3b 也是 luck，所有 spread 都是 noise band 内的 spurious
2. **写 paper 时把 R1 vs B 的 +3.3pt 改写为 ±4pt CI**（直到 multi-seed 数据补上）
3. Phase 3 Hybrid-K 暂缓启动，等 R1' 数据出来再决定要不要起
4. 写 follow-up paper 想法：reverse_kl 阈值 / teacher entropy 当 OPD-native mask metric，跟 Revisiting OPD top-K local support matching 接通

### 11. 文件 / commit

- `scripts/analyze_r5_is_ratio.py` — R5 IS ratio 详细分析（lift 不算这里，全分布算这里）
- `kl_analysis/phase25/r5_is_ratio_summary.json` / `r5_is_ratio_diagnostic.png`
- worklog Day 42 entry（本条）
