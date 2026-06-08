#!/usr/bin/env python3
"""Plot Phase 2.5 KL comparison: instant KL vs accumulated (mean-K) KL vs kl_loss
across R1 (mean-K=8), R4 (mean-K=4), and opd-4b-B (instant, K=1).

Metrics (see scripts/extract_kl_csv.py):
  instant_kl = rollout/opd_reverse_kl  — instant per-token reverse KL (comparable across K)
  accum_kl   = -rollout/advantages     — aggregated penalty entering advantage
               (mean-K: divided by K so magnitude ~ instant; instant: == instant_kl)
  kl_loss    = train/kl_loss           — KL from ref policy (coef=0, logging-only)
  grad_norm  = train/grad_norm

Usage: python3 scripts/plot_phase25_kl.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.join("kl_analysis", "phase25")
RUNS = [
    ("kl_B.csv", "opd-4b-B (instant, K=1)", "#888888"),
    ("kl_R4.csv", "R4 (mean-K=4)", "#1f77b4"),
    ("kl_R1.csv", "R1 (mean-K=8)", "#d62728"),
]


def load(path):
    cols = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v not in ("", None) else np.nan)
    return {k: np.array(v) for k, v in cols.items()}


def smooth(y, w=11):
    """Centered moving average with shrinking window at the edges.

    Plain np.convolve(mode="same") zero-pads beyond the array bounds, which
    drags the first/last few points toward 0 (a fake end-drop). Here each
    output point averages only the real samples that fall inside the window,
    so the edges stay faithful to the data.
    """
    y = np.asarray(y, dtype=float)
    # nan-safe: interpolate missing values first
    mask = np.isnan(y)
    if mask.any():
        idx = np.arange(len(y))
        y = y.copy()
        y[mask] = np.interp(idx[mask], idx[~mask], y[~mask])
    n = len(y)
    if n < 3:
        return y
    half = w // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def main():
    data = {label: load(os.path.join(HERE, fn)) for fn, label, _ in RUNS}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: instant KL (the comparable-across-K metric)
    ax = axes[0][0]
    for fn, label, color in RUNS:
        d = data[label]
        ax.plot(d["id"], d["instant_kl"], color=color, alpha=0.25, lw=0.8)
        ax.plot(d["id"], smooth(d["instant_kl"]), color=color, lw=2, label=label)
    ax.set_title("Instant per-token reverse KL (opd_reverse_kl)\nlogged identically across K → directly comparable")
    ax.set_xlabel("rollout id")
    ax.set_ylabel("instant reverse KL")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: accumulated KL (= -advantage = aggregated penalty)
    ax = axes[0][1]
    for fn, label, color in RUNS:
        d = data[label]
        ax.plot(d["id"], d["accum_kl"], color=color, alpha=0.25, lw=0.8)
        ax.plot(d["id"], smooth(d["accum_kl"]), color=color, lw=2, label=label)
    ax.set_title("Accumulated KL entering advantage (-advantages)\nmean-K divides by K → magnitude stays ~instant")
    ax.set_xlabel("rollout id")
    ax.set_ylabel("accumulated (mean-K) KL")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: instant KL on log-y to compare convergence speed
    ax = axes[1][0]
    for fn, label, color in RUNS:
        d = data[label]
        y = smooth(d["instant_kl"])
        y = np.clip(y, 1e-4, None)
        ax.plot(d["id"], y, color=color, lw=2, label=label)
    ax.set_yscale("log")
    ax.set_title("Instant KL (log-y) — convergence speed across K")
    ax.set_xlabel("rollout id")
    ax.set_ylabel("instant reverse KL (log)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # Panel 4: kl_loss (KL from ref) + grad_norm twin
    ax = axes[1][1]
    for fn, label, color in RUNS:
        d = data[label]
        ax.plot(d["id"], smooth(d["kl_loss"]), color=color, lw=2, label=f"{label} kl_loss")
    ax.set_title("KL from ref (train/kl_loss, coef=0 logging-only)\n+ grad_norm (dashed, right axis)")
    ax.set_xlabel("rollout id")
    ax.set_ylabel("kl_loss (KL from ref / SFT start)")
    ax2 = ax.twinx()
    for fn, label, color in RUNS:
        d = data[label]
        ax2.plot(d["id"], smooth(d["grad_norm"]), color=color, lw=1.2, ls="--", alpha=0.7)
    ax2.set_ylabel("grad_norm (dashed)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Phase 2.5 — KL trajectories: instant vs accumulated (mean-K) vs kl_loss", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "phase25_kl_comparison.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # Quick numeric summary
    print("\n=== Final (last 10 rollouts mean) ===")
    print(f"{'run':28s} {'instant_kl':>12s} {'accum_kl':>12s} {'kl_loss':>10s} {'grad_norm':>10s}")
    for fn, label, color in RUNS:
        d = data[label]
        def tail(k):
            v = d[k][-10:]
            return np.nanmean(v)
        print(f"{label:28s} {tail('instant_kl'):12.4f} {tail('accum_kl'):12.4f} {tail('kl_loss'):10.4f} {tail('grad_norm'):10.4f}")
    print("\n=== Start (first 5 rollouts mean) ===")
    for fn, label, color in RUNS:
        d = data[label]
        def head(k):
            return np.nanmean(d[k][:5])
        print(f"{label:28s} {head('instant_kl'):12.4f} {head('accum_kl'):12.4f}")


if __name__ == "__main__":
    main()
