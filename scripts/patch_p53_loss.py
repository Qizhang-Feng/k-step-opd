#!/usr/bin/env python3
"""Surgically patch p5-3's slime to add --opd-agg and --opd-dualclip-c.

Run inside the p5-3 k-step-opd container. Modifies in-place:
  /root/slime/slime/utils/arguments.py
  /root/slime/slime/backends/megatron_utils/loss.py

Idempotent: safe to re-run; bails if patches already applied.
"""
from pathlib import Path
import re
import shutil
import sys
import time


def patch_arguments():
    p = Path("/root/slime/slime/utils/arguments.py")
    s = p.read_text()
    if "--opd-agg" in s and "--opd-dualclip-c" in s:
        print("arguments.py already patched, skipping")
        return False
    # Insert new flags right after --opd-horizon block.
    anchor = '"--opd-horizon",'
    if anchor not in s:
        print("ERROR: anchor for --opd-horizon not found in arguments.py; aborting")
        sys.exit(1)
    # Find the closing line that terminates the --opd-horizon parser.add_argument(...) call.
    # The line is "            )" — 12 spaces + ).
    idx = s.index(anchor)
    end_paren = s.index("\n            )\n", idx)
    insertion_point = end_paren + len("\n            )\n")
    new_args = """
            parser.add_argument(
                "--opd-agg",
                type=str,
                default="mean",
                choices=["mean", "sum"],
                help=(
                    "Aggregation for cumulative OPD (when --opd-cumulative). "
                    "'mean' divides by horizon (slime v2 default, denoising). "
                    "'sum' is textbook reward-to-go (variance grows with K). "
                    "Phase 2.5 main ablation."
                ),
            )
            parser.add_argument(
                "--opd-dualclip-c",
                type=float,
                default=-1.0,
                help=(
                    "FIPO-style dual-clip mask threshold (arXiv 2603.19835). "
                    "When >0, tokens with IS ratio exp(student_logp - rollout_logp) > c "
                    "are excluded from the cumulative sum. Only used in cumulative mode."
                ),
            )
"""
    s2 = s[:insertion_point] + new_args + s[insertion_point:]
    backup = p.with_suffix(p.suffix + f".pre-patch-{int(time.time())}")
    shutil.copy(p, backup)
    p.write_text(s2)
    print(f"patched arguments.py (backup: {backup})")
    return True


def patch_loss():
    p = Path("/root/slime/slime/backends/megatron_utils/loss.py")
    s = p.read_text()
    if "opd_agg" in s and "opd_dualclip_c" in s:
        print("loss.py already patched, skipping")
        return False
    # The cumulative-OPD block we want to replace. Use the exact block shown by sed.
    old_block = """    reverse_kls = []
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
"""
    new_block = """    # Phase 2.5: agg = "mean" (slime v2 default, denoising) or "sum" (reward-to-go).
    agg = getattr(args, "opd_agg", "mean")
    # Optional FIPO dual-clip mask: exclude tokens whose IS ratio exceeds threshold.
    dualclip_c = float(getattr(args, "opd_dualclip_c", -1.0))
    rollout_log_probs = rollout_data.get("rollout_log_probs") if dualclip_c > 0 else None

    reverse_kls = []
    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]

        if getattr(args, "opd_cumulative", False):
            gamma = float(getattr(args, "opd_gamma", 1.0))
            horizon = int(getattr(args, "opd_horizon", -1))
            T = int(reverse_kl.numel())

            if rollout_log_probs is not None:
                is_ratio = torch.exp(student_log_probs[i] - rollout_log_probs[i].to(device=device))
                keep_mask = (is_ratio <= dualclip_c).to(reverse_kl.dtype)
                masked_kl = reverse_kl * keep_mask
            else:
                masked_kl = reverse_kl

            if horizon <= 0 or horizon >= T:
                # full sequence (v2 semantics: never divide; agg only matters for truncated)
                if gamma == 1.0:
                    cumulative_kl = torch.flip(torch.cumsum(torch.flip(masked_kl, [0]), dim=0), [0])
                else:
                    cumulative_kl = torch.zeros_like(reverse_kl)
                    running = torch.zeros((), dtype=reverse_kl.dtype, device=reverse_kl.device)
                    for t in range(T - 1, -1, -1):
                        running = masked_kl[t] + gamma * running
                        cumulative_kl[t] = running
            else:
                # truncated horizon: each token only looks K steps ahead
                cumulative_kl = torch.zeros_like(reverse_kl)
                for t in range(T):
                    end = min(t + horizon, T)
                    actual_k = end - t
                    if gamma == 1.0:
                        s_ = masked_kl[t:end].sum()
                    else:
                        weights = gamma ** torch.arange(actual_k, device=reverse_kl.device, dtype=reverse_kl.dtype)
                        s_ = (masked_kl[t:end] * weights).sum()
                    cumulative_kl[t] = (s_ / actual_k) if agg == "mean" else s_

            advantages[i] = adv - args.opd_kl_coef * cumulative_kl
        else:
            advantages[i] = adv - args.opd_kl_coef * reverse_kl

        reverse_kls.append(reverse_kl)
"""
    if old_block not in s:
        print("ERROR: cumulative block in loss.py doesn't match expected; aborting")
        sys.exit(1)
    s2 = s.replace(old_block, new_block)
    backup = p.with_suffix(p.suffix + f".pre-patch-{int(time.time())}")
    shutil.copy(p, backup)
    p.write_text(s2)
    print(f"patched loss.py (backup: {backup})")
    return True


def main():
    patch_arguments()
    patch_loss()
    print("DONE")


if __name__ == "__main__":
    main()
