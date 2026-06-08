#!/usr/bin/env python3
"""Inject `rollout_data['_rollout_id'] = rollout_id` before
compute_advantages_and_returns calls in actor.py. Idempotent."""
import sys
from pathlib import Path

p = Path("/root/slime/slime/backends/megatron_utils/actor.py")
src = p.read_text()

if '_rollout_id' in src and 'rollout_data["_rollout_id"]' in src:
    print("actor.py already has _rollout_id injection, skipping.")
    sys.exit(0)

# Patch the call inside train_actor (line ~457)
ANCHOR_TRAIN = (
    "                # Calculate adv and returns. Need to performed before training (instead of on the fly),\n"
    "                # because we may need normalize the whole rollout.\n"
    "                compute_advantages_and_returns(self.args, rollout_data)\n"
)
INJECTED_TRAIN = (
    "                # Calculate adv and returns. Need to performed before training (instead of on the fly),\n"
    "                # because we may need normalize the whole rollout.\n"
    '                rollout_data["_rollout_id"] = rollout_id\n'
    "                compute_advantages_and_returns(self.args, rollout_data)\n"
)

# Patch the train_critic call too (line ~383)
ANCHOR_CRITIC = (
    "        compute_advantages_and_returns(self.args, rollout_data)\n"
)
INJECTED_CRITIC = (
    '        rollout_data["_rollout_id"] = rollout_id\n'
    "        compute_advantages_and_returns(self.args, rollout_data)\n"
)

modified = False

if ANCHOR_TRAIN in src:
    src = src.replace(ANCHOR_TRAIN, INJECTED_TRAIN)
    modified = True
    print("Patched train_actor call.")
else:
    print("WARNING: train_actor anchor not found.")

# Replace any remaining unpatched call (the critic site uses identical text)
if ANCHOR_CRITIC in src:
    src = src.replace(ANCHOR_CRITIC, INJECTED_CRITIC, 1)
    modified = True
    print("Patched train_critic call.")

if modified:
    p.write_text(src)
    print(f"actor.py now {len(src.splitlines())} lines.")
else:
    print("No anchors matched, nothing changed.")
    sys.exit(1)
