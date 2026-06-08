#!/usr/bin/env python3
"""Inject OPD-cumulative + OPD-dump-KL flags into the container's arguments.py
without disturbing other definitions. Idempotent."""
import sys
from pathlib import Path

p = Path("/root/slime/slime/utils/arguments.py")
src = p.read_text()

INJECT_BLOCK = '''
            parser.add_argument(
                "--opd-cumulative",
                action="store_true",
                default=False,
                help=(
                    "If set, use cumulative (reward-to-go) reverse KL on the advantage instead of "
                    "per-token instant KL. Each position t gets sum_{d=0..K-1} gamma^d * KL[t+d]."
                ),
            )
            parser.add_argument(
                "--opd-gamma",
                type=float,
                default=1.0,
                help="Discount factor for cumulative OPD KL penalty. Only used when --opd-cumulative is set.",
            )
            parser.add_argument(
                "--opd-horizon",
                type=int,
                default=-1,
                help=(
                    "Max number of future tokens to include in cumulative OPD KL. "
                    "-1 means full sequence (no truncation). Only used when --opd-cumulative is set."
                ),
            )
            parser.add_argument(
                "--opd-dump-kl-path",
                type=str,
                default=None,
                help=(
                    "If set, dump per-sample / per-token reverse-KL data (with token ids, "
                    "student logp, teacher logp) every --opd-dump-kl-interval rollouts. "
                    "Path is a template string, e.g. '/path/opd_kl/r{rollout_id}_rank{rank}.jsonl'."
                ),
            )
            parser.add_argument(
                "--opd-dump-kl-interval",
                type=int,
                default=1,
                help="Dump per-token KL data every N rollouts. Only used with --opd-dump-kl-path.",
            )
            parser.add_argument(
                "--opd-dump-kl-max-samples",
                type=int,
                default=-1,
                help=(
                    "Max samples per rank per dump (cap to keep file sizes reasonable). "
                    "-1 means dump all samples."
                ),
            )
'''

ANCHOR = (
    '            parser.add_argument(\n'
    '                "--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."\n'
    '            )\n'
)

if "--opd-cumulative" in src:
    print("OPD cumulative flag already present, skipping.")
    sys.exit(0)

if ANCHOR not in src:
    print("ERROR: anchor not found in arguments.py. Aborting.", file=sys.stderr)
    sys.exit(1)

new_src = src.replace(ANCHOR, ANCHOR + INJECT_BLOCK)
p.write_text(new_src)
n_added = INJECT_BLOCK.count("parser.add_argument")
print(f"Injected {n_added} OPD flags. Total lines: {len(new_src.splitlines())}")
