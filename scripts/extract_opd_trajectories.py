#!/usr/bin/env python3
"""Extract OPD training trajectories from log files into CSV.

Reads grep'd log lines like:
  rollout 5: {'rollout/opd_reverse_kl': 0.123, 'rollout/truncated': 0.92, ...}
  step 5: {'train/loss': 0.123, 'train/grad_norm': 1.04, ...}

Run on a machine with the log file, output CSV to stdout.
"""
import argparse
import json
import re
import sys

ROLLOUT_RE = re.compile(r"rollout (\d+): (\{.*\})")
STEP_RE = re.compile(r"step (\d+): (\{.*\})")


def parse_dict_line(line):
    # Convert single-quoted dict to JSON
    s = line.replace("'", '"')
    # tensor() and Status.X etc — strip
    s = re.sub(r"tensor\([^)]*\)", "0", s)
    try:
        return json.loads(s)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logfile")
    p.add_argument("--out", default="-")
    args = p.parse_args()

    rollouts = {}  # id -> dict
    steps = {}
    with open(args.logfile) as f:
        for line in f:
            for regex, store in [(ROLLOUT_RE, rollouts), (STEP_RE, steps)]:
                m = regex.search(line)
                if not m:
                    continue
                idx = int(m.group(1))
                d = parse_dict_line(m.group(2))
                if d:
                    store[idx] = d
                break

    # Merge by step/rollout id
    fields = [
        "id",
        "rollout/opd_reverse_kl",
        "rollout/truncated",
        "rollout/response_lengths",
        "rollout/log_probs",
        "rollout/teacher_log_probs",
        "rollout/ref_log_probs",
        "train/loss",
        "train/pg_loss",
        "train/kl_loss",
        "train/entropy_loss",
        "train/grad_norm",
        "train/lr-pg_0",
    ]

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    out.write(",".join(fields) + "\n")
    all_ids = sorted(set(rollouts) | set(steps))
    for i in all_ids:
        row = [str(i)]
        for f in fields[1:]:
            val = None
            if f.startswith("rollout/"):
                val = rollouts.get(i, {}).get(f)
            elif f.startswith("train/"):
                val = steps.get(i, {}).get(f)
            row.append(f"{val:.6f}" if isinstance(val, (int, float)) else "")
        out.write(",".join(row) + "\n")
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
