# Session Summary — 2026-05-11

## 项目概述

**K-Step Reward-to-Go OPD (On-Policy Distillation) for Language Models**

研究目标：验证 k-step return 在 OPD 中的效果。当前阶段：建立 single-step OPD baseline（SFT cold-start → OPD）。

## 当前状态

### 已完成

1. **SFT Cold-Start 训练** — 三个数据量版本
   - Qwen3-8B-Base + OpenThoughts3 math subset LoRA SFT
   - 30K / 50K / 100K 数据量
   - 框架: ms-swift 4.1.3, LoRA rank 128, lr 1e-3, batch 128

2. **Eval 结果** (n=1, max_tokens=8192, temp=0.6)

| 模型 | MATH-500 | AIME-2024 | AIME-2025 |
|------|:---:|:---:|:---:|
| Qwen3-8B-Base (raw) | 54.4% | 16.7% | 13.3% |
| SFT 30K | 76.8% | 跑中 | 跑中 |
| SFT 50K | 73.6% | 跑中 | 跑中 |
| SFT 100K | 79.6% | 30.0% | 20.0% |
| Teacher (Qwen3-8B post-trained) | 91.8% | 56.7% | 23.3% |

3. **Greenland 部署完成** — 镜像 + S3 + job JSON 全部就绪

### 正在进行

1. **Greenland OPD 训练** (`job_opd_100k.json`) — SFT 100K → OPD with Qwen3-8B teacher
   - 预计 ~13 小时
   - 之前在 p5-2 跑到 step 8 被 boshih 进程 kill

2. **30K/50K AIME eval** — p5-5 上跑中（刚重启，~15 分钟完成）

### 待做

- OPD checkpoint eval
- 如果 OPD 有效 → 实现 k-step return
- 考虑 Qwen3-32B teacher（更大 gap）

---

## 关键技术细节

### 模型和路径

| 模型 | HF 路径 | torch_dist 路径 |
|------|---------|----------------|
| Qwen3-8B-Base | `/root/.cache/huggingface/Qwen3-8B-Base` | - |
| Qwen3-8B (teacher) | `/root/.cache/huggingface/Qwen3-8B` | - |
| SFT 100K merged | `checkpoints/sft-qwen3-8b-base-lora-merged` | `checkpoints/sft-qwen3-8b-base-lora-merged_torch_dist` |
| SFT 50K merged | `checkpoints/sft-50k-merged` | - |
| SFT 30K merged | `checkpoints/sft-qwen3-8b-base-lora-30k-merged` | - |

### S3 结构 (`delphi-greenland-res-alpha/qzf/`)

```
qzf/
├── code/
│   ├── k-step-opd.tar.gz          # scripts/ + configs/ + greenland/ + eval_math.py
│   ├── bootstrap_sft.sh
│   └── bootstrap_slime.sh
├── data/
│   ├── sft_math_30k_v2.jsonl
│   ├── sft_math_50k_v2.jsonl
│   ├── sft_math_100k_v2.jsonl
│   └── dapo-math-17k.jsonl
├── models/
│   ├── Qwen3-8B-Base/
│   └── Qwen3-8B/
└── checkpoints/
    ├── sft-100k-torch_dist/
    └── sft-100k-merged/
```

### Greenland ECR 镜像

- `654654486179.dkr.ecr.us-east-2.amazonaws.com/k-step-opd-sft:greenland-v1`
- `654654486179.dkr.ecr.us-east-2.amazonaws.com/k-step-opd-slime:greenland-v1`

两个镜像都基于 `slimerl/slime:latest`（PyTorch 2.9 + SGLang 0.5.9 + Megatron-LM）。
- SFT 镜像额外装了 ms-swift 4.1.3 + torchao ≥0.16.0
- Slime 镜像保持 torchao==0.9.0

### OPD 训练配置 (Greenland)

```
Student: SFT 100K checkpoint (MATH-500 79.6%)
Teacher: Qwen3-8B (MATH-500 91.8%)
Data: dapo-math-17k (17398 prompts)
Actor: TP=4 (GPU 0-3)
Rollout: 2 engines (GPU 4,5), 1 GPU per engine
Teacher: TP=2 (GPU 6,7)
num_rollout: 200, global_batch: 32, n_samples: 4
max_response_len: 8192, lr: 5e-7, opd_kl_coef: 1.0
max_tokens_per_gpu: 4096
```

### 已知问题

1. **Actor TP=2 OOM**: 8B model + ref model + optimizer 每卡 77GB，H100 80GB 不够。必须 TP=4。
2. **100% truncation**: SFT 模型 response ~24K chars (~8K tokens)，max_response_len=8192 全部截断。
3. **Eval server crash**: n=8 + workers=16 会 OOM crash。稳定配置: n=1 + workers=8 + max_running_requests=8。
4. **Eval 需要 `<think>\n` prefix**: SFT 后的 base model 不会自动生成 `<think>` tag，需要在 prompt 末尾加 `<|im_start|>assistant\n<think>\n`。
5. **Zombie 进程**: 容器长时间运行积累 zombie，需要 `docker restart` 清理。
6. **p5-2 被 boshih 占用**: `boshih_slime` 容器里的 MegatronTrainRayActor 会抢 GPU。

---

## 节点状态

| 节点 | Region | 状态 | 备注 |
|------|--------|------|------|
| p5-2 | Ohio | 🔴 被占 | boshih 的训练在跑 |
| p5-4 | Ohio | 🟢 空闲 | chenluy 偶尔用 |
| p5-5 | Ohio | 🟡 跑 eval | 30K/50K AIME eval 中 |
| p5-6 | Ohio | 未检查 | |
| qzf-dev | Ohio | 🟢 4×L40S | 可做 teacher serving |

---

## 文件结构

```
/Volumes/workplace/k-step-opd/
├── worklog.md                    # 完整工作日志
├── session-summary-20260511.md   # 本文件
├── eval_math.py                  # Eval 脚本 (SGLang + deepscaler reward)
├── prepare_sft_data_v2.py        # 数据准备 (token-level 过滤)
├── scripts/
│   ├── train-opd.sh              # OPD 训练脚本 (slime)
│   ├── run-sft-lora.sh           # SFT 训练脚本 (ms-swift)
│   ├── eval-sft-checkpoint.sh    # Eval 全套 (AIME + MATH-500)
│   └── convert-ckpt.sh           # HF → torch_dist 转换
├── configs/
│   └── phase2-sft100k-opd.env    # OPD 配置
├── greenland/
│   ├── Dockerfile.sft            # SFT 镜像
│   ├── Dockerfile.slime          # OPD 镜像
│   ├── bootstrap_sft.sh          # Greenland SFT 启动脚本
│   ├── bootstrap_slime.sh        # Greenland OPD 启动脚本
│   ├── build_and_push.sh         # 构建+测试+推送
│   ├── test_offline.sh           # 离线验证 (--network=none)
│   ├── job_sft.json              # SFT 100K job
│   ├── job_sft_50k.json          # SFT 50K job
│   └── job_opd_100k.json         # OPD job
└── slime/                        # THUDM/slime 代码 (git submodule)
```

---

## Credentials / Access

- **AWS 654654486179**: `ada credentials update --account 654654486179 --role IibsAdminAccess-DO-NOT-DELETE --provider conduit --once`
- **ECR login**: 需要先 ada 拿 credentials，然后 `aws ecr get-login-password | docker login`
- **p5 SSH key**: `~/.ssh/dl-machine-ohio.pem` (所有 Ohio p5 共用)
- **p5 间直连**: 需要把 key 复制到源机器 `~/.ssh/dl-machine-ohio.pem`，用 private IP 连接
- **S3 bucket**: `delphi-greenland-res-alpha` (Greenland 可访问), `qzf-k-step-opd-us-east-2` (需要 bucket policy)

---

## 下一个 Session 的优先事项

1. **检查 Greenland OPD job 结果** — 看 loss 趋势、truncation、repetition
2. **检查 30K/50K AIME eval 结果** — 补全对比表
3. **如果 OPD 有效 (pass@1 提升 ≥2pt)**:
   - 实现 k-step return (改 `slime/backends/megatron_utils/loss.py`)
   - k ∈ {2, 4, 8} sweep
4. **如果 OPD 无效**:
   - 增加 max_response_len 到 16384（解决 100% truncation）
   - 用 Qwen3-32B teacher（gap 更大）
   - 减少 SFT 数据量（50K checkpoint gap 18pt 更合适）
5. **基础设施改进**:
   - 在 qzf-dev 上部署 teacher server（释放 p5 的 2 个 GPU 给 rollout）
   - 考虑双节点训练
