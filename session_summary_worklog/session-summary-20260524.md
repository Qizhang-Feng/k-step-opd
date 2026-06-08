# Session Summary — 2026-05-24

## Project: K-Step OPD (On-Policy Distillation)
**Status**: 4B Instant OPD training complete (300 rollouts). KL dumps collected. Ready for offline analysis + eval.

---

## TL;DR

1. **4B Instant OPD training completed** — 300 rollouts, 12 checkpoints, ~10.5h on p5-3 (8×H100).
2. **Per-token KL dumps collected** — 29 dumps × 4 ranks × 8 samples = ~928 samples with full token-level reverse KL data (357MB).
3. **OPD reverse KL dropped 44%** — from 0.165 → 0.092 over training, no repetition/degeneration.
4. **Remote teacher architecture validated** — qzf-dev (4×L40S, DP=4) serving Qwen3-8B over same-VPC private network, stable for 10+ hours.
5. **Speed bottleneck identified** — rollout generation (60% of step time) due to 78% truncation at max_tokens=8192.

---

## Training Results

### OPD Training Curve (300 rollouts)

| Metric | Start (rollout 1) | End (rollout 299) | Change |
|--------|:--:|:--:|:--:|
| opd_reverse_kl | 0.165 | 0.092 | ↓44% |
| truncated_ratio | 0.67 | 0.67 | stable |
| repetition_frac | 0.0 | 0.0 | none |
| grad_norm | 0.5-0.8 | 0.5-0.8 | stable |
| kl_from_ref | ~0.10 | ~0.09 | minimal drift |
| response_length | 7277 | 7657 | slight increase |

### Checkpoints Saved

12 checkpoints at: iter_0000024, 049, 074, 099, 124, 149, 174, 199, 224, 249, 274, 299

Location: `p5-3:/opt/dlami/nvme/qzf/models/opd-4b-v2-ckpt700-instant/`

---

## Architecture: Remote Teacher

### Setup
- **p5-3** (8×H100): 4 actor GPUs (TP=4) + 4 rollout engines (sglang TP=1 × 4)
- **qzf-dev** (4×L40S): Teacher Qwen3-8B, sglang TP=1 DP=4, port 30000
- Network: same VPC private IP (172.31.31.105), 1.5ms RTT

### Why DP=4 not TP=4 for teacher
- Teacher only does prefill (logprob computation), no decoding
- TP=4 OOM'd on concurrent prefill (activation memory, not weights)
- DP=4 = 4 independent replicas, each handles 1/4 of concurrent requests
- 8B bf16 = 16GB weights per replica, 30GB free for KV → no OOM

### Performance
- Teacher throughput: stable, no crashes over 10.5h
- Network overhead: negligible (logprobs are ~10KB per request)

---

## Per-Token KL Dump

### Format (jsonl, one line per sample)
```json
{
  "rollout_id": 10,
  "rank": 0,
  "sample_idx": 3,
  "prompt_length": 245,
  "response_length": 8192,
  "reward": 0.0,
  "prompt_token_ids": [...],
  "response_token_ids": [...],
  "student_log_probs": [...],
  "teacher_log_probs": [...],
  "reverse_kl": [...],
  "advantage": [...]
}
```

### Stats
- 29 dump timepoints (every 10 rollouts: r10, r20, ..., r290)
- 4 ranks × 8 samples per dump = 32 samples/dump
- Total: ~928 samples, 357MB
- Location: `p5-3:/opt/dlami/nvme/qzf/models/opd-4b-v2-ckpt700-instant/kl_dump/`

---

## Speed Analysis

### Per-step breakdown (~135s total)

| Phase | Time | % |
|-------|------|---|
| **Rollout generation** | **~80s** | **60%** |
| ref_log_probs | 10s | 7% |
| log_probs | 10s | 7% |
| actor_train | 32s | 24% |
| update_weights | 0.3s | <1% |

### Bottleneck: Rollout
- 64 samples × avg 7500 tokens = 480K tokens per rollout
- 4 sglang engines @ 1400 tokens/gpu/sec = 5600 tokens/sec total
- 480K / 5600 = 86s (matches observed ~80s)
- 78% of responses hit max_tokens=8192 (student generates long thinking chains)

### Potential speedups for future runs
1. Reduce max_response_len: 8192 → 4096 (halves rollout time)
2. Reduce n_samples: 4 → 2 (halves rollout time)
3. More rollout engines: 6 engines + actor TP=2

---

## Code Changes This Session

### New files
- `scripts/train-opd-extteacher.sh` — OPD training with external teacher URL
- `configs/opd-4b-v2-ckpt700-instant-extteacher.env` — config for this run
- `scripts/run-opd-4b-v2-ckpt700-with-kl-dump.sh` — bootstrap script (unused, superseded)
- `scripts/_inject_opd_args.py` — inject cumulative + dump flags into arguments.py
- `scripts/_inject_opd_loss.py` — inject cumulative KL + _dump_opd_kl into loss.py
- `scripts/_inject_opd_actor.py` — inject _rollout_id into actor.py
- `scripts/_inject_log_skip.py` — fix log_rollout_data to skip private keys

### Modified (in container only, not committed)
- `/root/slime/slime/utils/arguments.py` — +6 OPD flags
- `/root/slime/slime/backends/megatron_utils/loss.py` — cumulative + dump
- `/root/slime/slime/backends/megatron_utils/actor.py` — _rollout_id injection
- `/root/slime/slime/backends/megatron_utils/data.py` — skip `_` keys in logging

---

## Bugs Fixed

1. **`critic_train_only` AttributeError** — local arguments.py was older than container's HEAD. Fixed by reverting to HEAD and injecting only new flags.
2. **Teacher OOM (TP=4)** — switched to DP=4 with conservative max_running_requests=32.
3. **`_rollout_id` ValueError in log_rollout_data** — patched data.py to skip keys starting with `_`.

---

## Next Session Priorities

### High priority
1. **Eval OPD checkpoints** — convert iter_099/199/299 to HF, eval on AIME-2024 with max_tokens=30000
2. **Offline KL analysis** — write `scripts/analyze_kl_dumps.py`:
   - Instant KL distribution (mean/std/percentiles over training)
   - Future KL (K-step cumulative) for K ∈ {1,2,4,8,16,32,full}
   - Token-type analysis (which tokens have highest KL)
   - Signal-to-noise ratio of future KL vs instant KL

### Medium priority
3. **Compare with 8B OPD baseline** — 8B instant OPD was +10pt (50→60%). Does 4B also get +10pt (60→70%)?
4. **Cumulative OPD run** — use KL analysis to pick optimal K and γ, then run cumulative variant

### Lower priority
5. **152K-700steps Greenland result** — check if it's done (last ablation for mixed-data hypothesis)
6. **Lightning-OPD offline path** — precompute teacher logprobs for faster iteration

---

## Cluster State

| Machine | Status | Use |
|---------|--------|-----|
| p5-3 | idle (training done) | OPD ckpts + KL dumps on NVMe |
| qzf-dev | teacher-sglang container (may be stopped) | Can restart for eval |
| p5-4 | available | Has 8B SFT ckpt, teacher data |
