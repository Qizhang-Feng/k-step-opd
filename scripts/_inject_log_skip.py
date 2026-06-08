#!/usr/bin/env python3
"""Patch log_rollout_data to skip private keys starting with underscore."""
import sys
from pathlib import Path

p = Path("/root/slime/slime/backends/megatron_utils/data.py")
src = p.read_text()

ANCHOR = '''        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_routed_experts",
                "max_seq_lens",
                "dynamic_global_batch_size",
            ]:
                continue
'''

PATCHED = '''        for key, val in rollout_data.items():
            if key.startswith("_") or key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_routed_experts",
                "max_seq_lens",
                "dynamic_global_batch_size",
            ]:
                continue
'''

if 'key.startswith("_") or' in src:
    print("Already patched.")
    sys.exit(0)

if ANCHOR not in src:
    print("ERROR: anchor not found.")
    sys.exit(1)

src = src.replace(ANCHOR, PATCHED)
p.write_text(src)
print("Patched data.py to skip private keys.")
