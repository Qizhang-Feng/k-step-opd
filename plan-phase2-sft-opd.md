# Phase 2: SFT Cold-Start → OPD Positive Baseline

## 目标
得到一个 k=1 OPD positive baseline：SFT checkpoint → OPD 后 MATH-500 pass@1 至少 +2pt。

---

## 路线 A：用 TML Tinker SFT Checkpoint（最快）

### 前提
- 有 Tinker API key
- TML 的 checkpoint 对你的账号开放

### 步骤

**A1. 安装 tinker-cookbook**
```bash
pip install tinker-cookbook
export TINKER_API_KEY="你的key"
```

**A2. 下载 TML SFT checkpoint**
```bash
tinker checkpoint download \
  "tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final" \
  -o /root/tml_ckpts --force
```

**A3. Merge 成 HF model**
```python
from tinker_cookbook import weights

adapter_dir = weights.download(
    tinker_path="tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final",
    output_dir="/root/tml_ckpts/adapter",
)

weights.build_hf_model(
    base_model="Qwen/Qwen3-8B-Base",
    adapter_path=adapter_dir,
    output_path="/root/models/Qwen3-8B-Base-TML-SFT",
    dtype="bfloat16",
    trust_remote_code=True,
    merge_strategy="auto",
)
```

**A4. Sanity check**
- 用 SGLang 起 server，发几个 math prompt
- 确认有 `<think>...</think>` + `\boxed{}`
- 确认不 repetition、正常停止

**A5. Eval SFT checkpoint**
- MATH-500 pass@1 目标：55-75%
- 如果 >85%：太强，不适合做 OPD baseline

**A6. 转成 slime/Megatron torch_dist**
```bash
cd /root/slime && source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/models/Qwen3-8B-Base-TML-SFT \
  --save /root/ckpts/Qwen3-8B-Base-TML-SFT_torch_dist
```

**A7. 跑 OPD**
- Teacher: Qwen3-8B（贴近 TML recipe）或 Qwen3-32B（贴近 slime 官方）
- Data: DeepMath-103K prompts（贴近 TML）或 dapo-math-17k（贴近 slime）
- 配置见下方"OPD 配置"

**A8. Eval OPD checkpoints**
- 每 20 步 eval 一次
- 看 pass@1 是否上涨

### 时间估算
- A1-A3: 30 分钟（下载 + merge）
- A4-A5: 1 小时（eval）
- A6: 10 分钟
- A7: 2-4 小时（OPD 训练）
- A8: 1 小时（eval）
- **总计：~5-6 小时**

### 风险
- TML checkpoint 可能没权限下载
- LoRA merge 可能有兼容性问题
- 8B 模型需要更多 GPU 内存

---

## 路线 B：自己做 SFT Cold-Start（更可控）

### 步骤

**B1. 下载数据和模型**
```bash
# 模型
hf download Qwen/Qwen3-8B-Base --local-dir /root/models/Qwen3-8B-Base
hf download Qwen/Qwen3-32B --local-dir /root/models/Qwen3-32B  # teacher

# 数据
hf download --repo-type dataset open-thoughts/OpenThoughts3-1.2M --local-dir /root/data/OpenThoughts3
hf download --repo-type dataset zwhe99/DeepMath-103K --local-dir /root/data/DeepMath-103K
```

**B2. 准备 SFT 数据**
从 OpenThoughts3 提取 math subset，过滤：
- 只取 math domain
- response 包含 `\boxed{}`
- response < 16K tokens
- 无严重 repetition
- 无 truncation

目标：50K-100K 条高质量 math reasoning traces

**B3. SFT 训练**
两种方式：

方式 1：用 slime/Megatron 做 SFT（如果支持）
方式 2：用 TRL / LLaMA-Factory / 其他框架做 SFT

配置：
```
model: Qwen3-8B-Base
data: OpenThoughts3 math 100K
training: full fine-tune
lr: 5e-6 ~ 2e-5
batch: global 64
seq_len: 16384
epochs: 1
```

**B4. Eval SFT checkpoint**
- 目标：MATH-500 pass@1 60-80%
- 如果 <50%：数据不够或 lr 太低
- 如果 >85%：数据太多或太强

**B5. 转成 slime torch_dist**
```bash
cd /root/slime && source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/models/Qwen3-8B-Base-SFT \
  --save /root/ckpts/Qwen3-8B-Base-SFT_torch_dist
```

**B6. 跑 OPD**
同路线 A 的 A7

**B7. Eval OPD checkpoints**
同路线 A 的 A8

### 时间估算
- B1: 1-2 小时（下载 32B 模型大）
- B2: 1 小时（数据处理）
- B3: 4-8 小时（SFT 训练，8B 模型 100K 数据）
- B4: 1 小时
- B5: 10 分钟
- B6: 2-4 小时
- B7: 1 小时
- **总计：~10-16 小时**

### 风险
- SFT 可能过强或过弱，需要调数据量
- 8B full fine-tune 需要 8×H100
- OpenThoughts3 数据质量参差不齐

---

## OPD 配置（两条路线共用）

### 贴近 TML recipe
```
Student: SFT checkpoint
Teacher: Qwen3-8B (TP=1, GPU 7)
Data: DeepMath-103K prompts
n_samples_per_prompt: 4
global_batch_size: 64
max_response_len: 8192
temperature: 1.0
opd_kl_coef: 1.0
lr: 5e-7
num_rollout: 100-150
```

### 贴近 slime 官方
```
Student: SFT checkpoint
Teacher: Qwen3-32B (TP=2, GPU 6,7)
Data: dapo-math-17k
n_samples_per_prompt: 4
global_batch_size: 64
max_response_len: 16384
temperature: 1.0
opd_kl_coef: 1.0
lr: 1e-6
num_rollout: 300
```

### GPU 分配（8×H100）
- 8B student: Actor GPU 0,1 (TP=2) + Rollout GPU 2,3,4,5
- 8B teacher: GPU 7 (TP=1)
- 32B teacher: GPU 6,7 (TP=2)

---

## 成功标准

| 指标 | 要求 |
|------|------|
| MATH-500 pass@1 提升 | ≥ +2pt |
| AIME pass@1 | 不下降 |
| avg response len | < SFT 的 1.5x |
| truncation | < 20% |
| repetition | 不明显上升 |

---

## 缩小版（如果 8B 太重）

用 1.7B 做缩小版验证：
```
Student: Qwen3-1.7B-Base
SFT: OpenThoughts3 math 50K (用 Qwen3-8B 生成 responses)
Teacher: Qwen3-8B
OPD: dapo-math-17k, 200 steps
目标 SFT: MATH-500 60-75%
目标 OPD: +5pt 以上
```

---

## 推荐执行顺序

1. 先试路线 A（最快，30 分钟就知道 TML checkpoint 能不能用）
2. 如果 A 不行（权限/兼容性），走路线 B
3. 如果 8B 太重，先用 1.7B 缩小版验证 recipe
