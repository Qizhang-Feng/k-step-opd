#!/usr/bin/env python3
"""Add --opd-soft-mask support to p5-3's slime (surgical patch).
Requires that --opd-dualclip-c and opd_agg are already patched (from patch_p53_loss.py).

Run inside p5-3 k-step-opd container. Idempotent.
"""
from pathlib import Path
import shutil
import time
import sys


def patch_arguments():
    p = Path("/root/slime/slime/utils/arguments.py")
    s = p.read_text()
    if "--opd-soft-mask" in s:
        print("arguments.py: --opd-soft-mask already present, skipping")
        return
    # Insert after the --opd-dualclip-c block
    anchor = '"--opd-dualclip-c",'
    if anchor not in s:
        print("ERROR: --opd-dualclip-c not found; run patch_p53_loss.py first")
        sys.exit(1)
    # Find the closing )\n of that add_argument
    idx = s.index(anchor)
    end = s.index("\n            )\n", idx)
    insertion = end + len("\n            )\n")
    new_block = '''            parser.add_argument(
                "--opd-soft-mask",
                action="store_true",
                default=False,
                help=(
                    "Use soft IS-weighted decay instead of hard 0/1 dual-clip mask. "
                    "When enabled (with --opd-dualclip-c > 0), tokens with IS ratio > c "
                    "get weight = c/IS (soft decay, never zero) instead of being removed."
                ),
            )
'''
    s2 = s[:insertion] + new_block + s[insertion:]
    backup = p.with_suffix(p.suffix + f".pre-softmask-{int(time.time())}")
    shutil.copy(p, backup)
    p.write_text(s2)
    print(f"arguments.py: added --opd-soft-mask (backup: {backup.name})")


def patch_loss():
    p = Path("/root/slime/slime/backends/megatron_utils/loss.py")
    s = p.read_text()
    if "opd_soft_mask" in s:
        print("loss.py: opd_soft_mask already present, skipping")
        return
    # Find the existing mask block and replace with soft-mask aware version
    old = """            if rollout_log_probs is not None:
                is_ratio = torch.exp(student_log_probs[i] - rollout_log_probs[i].to(device=device))
                keep_mask = (is_ratio <= dualclip_c).to(reverse_kl.dtype)
                masked_kl = reverse_kl * keep_mask
            else:
                masked_kl = reverse_kl"""
    if old not in s:
        print("ERROR: expected mask block not found in loss.py; check patch_p53_loss.py ran correctly")
        sys.exit(1)
    new = """            if rollout_log_probs is not None:
                is_ratio = torch.exp(student_log_probs[i] - rollout_log_probs[i].to(device=device))
                if getattr(args, "opd_soft_mask", False):
                    # Soft IS-weighted decay: weight = min(c/IS, 1) ∈ (0, 1]
                    soft_weight = torch.clamp(dualclip_c / torch.clamp(is_ratio, min=1.0), max=1.0)
                    masked_kl = reverse_kl * soft_weight
                else:
                    keep_mask = (is_ratio <= dualclip_c).to(reverse_kl.dtype)
                    masked_kl = reverse_kl * keep_mask
            else:
                masked_kl = reverse_kl"""
    s2 = s.replace(old, new)
    backup = p.with_suffix(p.suffix + f".pre-softmask-{int(time.time())}")
    shutil.copy(p, backup)
    p.write_text(s2)
    print(f"loss.py: added soft-mask branch (backup: {backup.name})")


if __name__ == "__main__":
    patch_arguments()
    patch_loss()
    print("DONE — soft-mask patched")
