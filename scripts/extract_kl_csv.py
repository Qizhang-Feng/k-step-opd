#!/usr/bin/env python3
"""Extract per-rollout KL metrics + per-step train metrics from an OPD driver log
into a compact CSV (id, instant_kl, neg_advantage=accumulated_kl, train_loss,
kl_loss, grad_norm). Designed to run remotely on the log host.

instant_kl   = rollout/opd_reverse_kl  (instant per-token reverse KL, comparable across K)
accum_kl     = -rollout/advantages     (= opd_kl_coef * cumulative_kl; the aggregated penalty)
train_loss   = train/loss              (= pg_loss, the OPD penalty entering the gradient)
kl_loss      = train/kl_loss           (KL from ref, logging-only since coef=0)
grad_norm    = train/grad_norm
"""
import json
import re
import sys

ROLLOUT_RE = re.compile(r"rollout (\d+): (\{[^}]*\})")
STEP_RE = re.compile(r"step (\d+): (\{[^}]*\})")


def parse_dict(s):
    s = s.replace("'", '"')
    s = re.sub(r"tensor\([^)]*\)", "0", s)
    try:
        return json.loads(s)
    except Exception:
        return None


def main():
    logfile = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "-"
    rollouts, steps = {}, {}
    with open(logfile, errors="ignore") as f:
        for line in f:
            m = ROLLOUT_RE.search(line)
            if m:
                d = parse_dict(m.group(2))
                if d:
                    rollouts[int(m.group(1))] = d
                continue
            m = STEP_RE.search(line)
            if m:
                d = parse_dict(m.group(2))
                if d:
                    steps[int(m.group(1))] = d

    ids = sorted(set(rollouts) | set(steps))
    fh = sys.stdout if out == "-" else open(out, "w")
    fh.write("id,instant_kl,accum_kl,train_loss,kl_loss,grad_norm,truncated,resp_len\n")
    for i in ids:
        r = rollouts.get(i, {})
        s = steps.get(i, {})
        instant = r.get("rollout/opd_reverse_kl")
        adv = r.get("rollout/advantages")
        accum = -adv if isinstance(adv, (int, float)) else None
        row = [
            i,
            instant,
            accum,
            s.get("train/loss"),
            s.get("train/kl_loss"),
            s.get("train/grad_norm"),
            r.get("rollout/truncated"),
            r.get("rollout/response_lengths"),
        ]
        fh.write(",".join("" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v)) for v in row) + "\n")
    if fh is not sys.stdout:
        fh.close()
        print(f"wrote {out}: {len(ids)} rows")


if __name__ == "__main__":
    main()
