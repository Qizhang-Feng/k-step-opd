#!/usr/bin/env python3
"""Inject cumulative-KL + per-token KL dump into apply_opd_kl_to_advantages.

This script monkey-patches /root/slime/slime/backends/megatron_utils/loss.py
in-place. Idempotent: re-running is a no-op.
"""
import sys
from pathlib import Path

p = Path("/root/slime/slime/backends/megatron_utils/loss.py")
src = p.read_text()

if "_dump_opd_kl" in src and "opd_cumulative" in src:
    print("Cumulative + dump already injected, skipping.")
    sys.exit(0)

# --- Replace the body of apply_opd_kl_to_advantages with extended version ---
ANCHOR_START = '''def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:'''

# We want to replace the existing function body wholesale.
# Find the function and the next top-level "def " that follows it.
start_idx = src.find(ANCHOR_START)
if start_idx < 0:
    print("ERROR: cannot find apply_opd_kl_to_advantages signature.", file=sys.stderr)
    sys.exit(1)

# Find next top-level def after start
search_from = start_idx + len(ANCHOR_START)
next_def_idx = src.find("\ndef ", search_from)
if next_def_idx < 0:
    print("ERROR: cannot find following top-level def.", file=sys.stderr)
    sys.exit(1)

new_func = '''def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted
    penalty to advantages in-place.

    Modes:
      - Instant (default): per-token penalty = reverse_kl[t]
      - Cumulative (--opd-cumulative): per-token penalty = sum_{d=0..K-1} gamma^d * reverse_kl[t+d]

    Optionally dumps per-token KL data (with token ids, student/teacher logp,
    reverse_kl, advantage) to jsonl every --opd-dump-kl-interval rollouts.
    """
    if student_log_probs is None:
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    device = student_log_probs[0].device
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

    reverse_kls = []
    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]

        if getattr(args, "opd_cumulative", False):
            gamma = float(getattr(args, "opd_gamma", 1.0))
            horizon = int(getattr(args, "opd_horizon", -1))
            T = int(reverse_kl.numel())

            if horizon <= 0 or horizon >= T:
                # full sequence
                if gamma == 1.0:
                    cumulative_kl = torch.flip(torch.cumsum(torch.flip(reverse_kl, [0]), dim=0), [0])
                else:
                    cumulative_kl = torch.zeros_like(reverse_kl)
                    running = torch.zeros((), dtype=reverse_kl.dtype, device=reverse_kl.device)
                    for t in range(T - 1, -1, -1):
                        running = reverse_kl[t] + gamma * running
                        cumulative_kl[t] = running
            else:
                # truncated horizon: each token only looks K steps ahead
                cumulative_kl = torch.zeros_like(reverse_kl)
                for t in range(T):
                    end = min(t + horizon, T)
                    actual_k = end - t
                    if gamma == 1.0:
                        cumulative_kl[t] = reverse_kl[t:end].sum() / actual_k
                    else:
                        weights = gamma ** torch.arange(actual_k, device=reverse_kl.device, dtype=reverse_kl.dtype)
                        cumulative_kl[t] = (reverse_kl[t:end] * weights).sum() / actual_k

            advantages[i] = adv - args.opd_kl_coef * cumulative_kl
        else:
            advantages[i] = adv - args.opd_kl_coef * reverse_kl

        reverse_kls.append(reverse_kl)

    rollout_data["opd_reverse_kl"] = reverse_kls

    # Optionally dump per-token KL data with token ids for offline analysis.
    dump_path_template = getattr(args, "opd_dump_kl_path", None)
    if dump_path_template:
        rollout_id = rollout_data.get("_rollout_id", -1)
        dump_interval = max(1, int(getattr(args, "opd_dump_kl_interval", 1)))
        if rollout_id < 0 or rollout_id % dump_interval == 0:
            _dump_opd_kl(
                args=args,
                rollout_id=rollout_id,
                rollout_data=rollout_data,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                reverse_kls=reverse_kls,
                dump_path_template=dump_path_template,
            )


def _dump_opd_kl(
    *,
    args: Namespace,
    rollout_id: int,
    rollout_data: RolloutBatch,
    student_log_probs: list[torch.Tensor],
    teacher_log_probs: list[torch.Tensor],
    reverse_kls: list[torch.Tensor],
    dump_path_template: str,
) -> None:
    """Write per-token OPD data (token_ids, student/teacher logp, reverse_kl) to jsonl.

    Writes one file per rank to avoid contention. Each line is one sample with
    keys: rollout_id, rank, sample_idx, prompt_length, response_length, reward,
    prompt_token_ids, response_token_ids, student_log_probs, teacher_log_probs,
    reverse_kl, advantage.
    """
    import json
    from pathlib import Path

    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0

    response_lengths = rollout_data.get("response_lengths", [])
    total_lengths = rollout_data.get("total_lengths", [])
    unconcat_tokens = rollout_data.get("unconcat_tokens", [])
    rewards = rollout_data.get("rewards", [])
    advantages = rollout_data.get("advantages", [])

    path = Path(dump_path_template.format(rollout_id=rollout_id, rank=rank))
    path.parent.mkdir(parents=True, exist_ok=True)

    max_samples = int(getattr(args, "opd_dump_kl_max_samples", -1))
    n_to_dump = len(reverse_kls) if max_samples < 0 else min(len(reverse_kls), max_samples)

    with open(path, "w") as f:
        for i in range(n_to_dump):
            tlen = total_lengths[i] if i < len(total_lengths) else None
            rlen = response_lengths[i] if i < len(response_lengths) else int(reverse_kls[i].numel())
            tokens_i = unconcat_tokens[i] if i < len(unconcat_tokens) else None

            if tokens_i is not None and tlen is not None:
                prompt_len = max(int(tlen) - int(rlen), 0)
                response_token_ids = tokens_i[prompt_len:prompt_len + int(rlen)].detach().to("cpu").tolist()
                prompt_token_ids = tokens_i[:prompt_len].detach().to("cpu").tolist()
            else:
                response_token_ids = []
                prompt_token_ids = []

            slp = student_log_probs[i].detach().to(dtype=torch.float32, device="cpu").tolist()
            tlp = teacher_log_probs[i].detach().to(dtype=torch.float32, device="cpu").tolist()
            rkl = reverse_kls[i].detach().to(dtype=torch.float32, device="cpu").tolist()
            if i < len(advantages):
                adv_i = advantages[i].detach().to(dtype=torch.float32, device="cpu").tolist()
            else:
                adv_i = []
            try:
                reward_i = float(rewards[i]) if i < len(rewards) and not isinstance(rewards[i], dict) else None
            except Exception:
                reward_i = None

            record = {
                "rollout_id": int(rollout_id),
                "rank": int(rank),
                "sample_idx": int(i),
                "prompt_length": len(prompt_token_ids),
                "response_length": int(rlen),
                "reward": reward_i,
                "prompt_token_ids": prompt_token_ids,
                "response_token_ids": response_token_ids,
                "student_log_probs": slp,
                "teacher_log_probs": tlp,
                "reverse_kl": rkl,
                "advantage": adv_i,
            }
            f.write(json.dumps(record) + "\\n")


'''

new_src = src[:start_idx] + new_func + src[next_def_idx + 1:]
p.write_text(new_src)
print(f"Patched apply_opd_kl_to_advantages + added _dump_opd_kl. New file lines: {len(new_src.splitlines())}")
