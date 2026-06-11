"""Lift / conditional-spike analysis (Pearson autocorr is dominated by 0-0
co-occurrence; this is the right metric for point-process clustering).

For each lag d compute:
    p_active            = P(|x[t]| > thr)                      ← marginal
    p_cond              = P(|x[t+d]| > thr  |  |x[t]| > thr)   ← conditional
    lift(d)             = p_cond / p_active

Interpretation:
    lift = 1  → spike at t is independent of spike at t+d
    lift > 1 → spike clustering (reasoning chunks)
    lift < 1 → spike repulsion (after-spike refractory)

Run inside k-step-opd container on p5-3.
"""
import json
import os
from glob import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/root/.cache/huggingface"
RUNS = [
    ("B (instant K=1)",          "opd-4b-B",                 "#444444"),
    ("opd-4b-A (instant)",       "opd-4b-v2-ckpt700-instant","#888888"),
    ("R4 (mean K=4)",            "opd-4b-R4-meanK4",         "#2ca02c"),
    ("R3b (mean K=8 + mask)",    "opd-4b-R3b-meanK8-mask",   "#ff7f0e"),
    ("R5 (mean K=8 + soft)",     "opd-4b-R5-meanK8-softmask","#9467bd"),
]
THR = 0.05
D_MAX = 32


def list_rollouts(run_dir):
    pat = os.path.join(run_dir, "kl_dump", "r*_rank*.jsonl")
    files = glob(pat)
    by_rid = defaultdict(list)
    for f in files:
        name = os.path.basename(f)
        try:
            rid = int(name.split("_")[0][1:])
        except ValueError:
            continue
        by_rid[rid].append(f)
    return sorted(by_rid.items())


def load_arrays(paths, max_samples=200):
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                rk = np.asarray(d.get("reverse_kl", []), dtype=np.float64)
                if rk.size > 64:
                    out.append(np.abs(rk))
                if len(out) >= max_samples:
                    return out
    return out


def lift(samples, d_max, thr):
    """Compute lift(d) = P(active[t+d] | active[t]) / P(active[t]) at each lag d."""
    p_active_total = 0.0
    n_total = 0
    p_cond_num = np.zeros(d_max + 1)
    p_cond_den = np.zeros(d_max + 1, dtype=np.int64)

    for x in samples:
        if len(x) < d_max + 4:
            continue
        active = x > thr
        p_active_total += active.mean() * len(active)
        n_total += len(active)
        for d in range(d_max + 1):
            n = len(active) - d
            if n <= 0:
                continue
            mask_t = active[: n]
            both = mask_t & active[d : d + n]
            p_cond_num[d] += both.sum()
            p_cond_den[d] += mask_t.sum()

    p_active = p_active_total / max(1, n_total)
    valid = p_cond_den > 0
    p_cond = np.zeros(d_max + 1)
    p_cond[valid] = p_cond_num[valid] / p_cond_den[valid]
    lift_vals = np.zeros(d_max + 1)
    lift_vals[valid] = p_cond[valid] / max(p_active, 1e-9)
    return p_active, p_cond, lift_vals


def main():
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax_p, ax_l = axes

    for label, name, color in RUNS:
        run_dir = os.path.join(BASE, name)
        rollouts = list_rollouts(run_dir)
        if not rollouts:
            continue
        samples = []
        for rid, paths in rollouts[::3]:
            samples.extend(load_arrays(paths, max_samples=10))
        if not samples:
            continue
        print(f"[{label}] n_samples={len(samples)}")

        p_active, p_cond, lift_vals = lift(samples, D_MAX, THR)
        summary[label] = {
            "p_active": float(p_active),
            "p_cond": p_cond.tolist(),
            "lift": lift_vals.tolist(),
        }

        ax_p.plot(range(D_MAX + 1), p_cond, "o-", color=color, lw=1.5, ms=3, label=label)
        ax_p.axhline(p_active, color=color, lw=0.8, ls=":", alpha=0.6)
        ax_l.plot(range(D_MAX + 1), lift_vals, "o-", color=color, lw=1.5, ms=3, label=label)

    ax_p.set_title(f"P(active[t+d] | active[t])  thr |x|>{THR}\n(dotted: marginal P(active))")
    ax_p.set_xlabel("lag d")
    ax_p.set_ylabel("conditional probability")
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=9)

    ax_l.axhline(1.0, color="red", lw=1, ls="--", alpha=0.7, label="independence (lift=1)")
    ax_l.set_title("Lift(d) = P(active[t+d] | active[t]) / P(active)\n>1 = spike clustering, <1 = spike repulsion, =1 = independent")
    ax_l.set_xlabel("lag d")
    ax_l.set_ylabel("lift")
    ax_l.grid(alpha=0.3)
    ax_l.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("/tmp/kstep_lift.png", dpi=130)
    print("saved /tmp/kstep_lift.png")

    with open("/tmp/kstep_lift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved /tmp/kstep_lift_summary.json")

    print()
    print(f"{'run':30s} {'p_active':>10s} {'lift(1)':>9s} {'lift(2)':>9s} {'lift(4)':>9s} {'lift(8)':>9s} {'lift(16)':>10s} {'lift(32)':>10s}")
    for label, _name, _ in RUNS:
        if label not in summary:
            continue
        s = summary[label]
        l = s["lift"]
        print(f"{label:30s} {s['p_active']:>10.4f} "
              f"{l[1]:>9.3f} {l[2]:>9.3f} {l[4]:>9.3f} {l[8]:>9.3f} "
              f"{l[16]:>10.3f} {l[32]:>10.3f}")


if __name__ == "__main__":
    main()
