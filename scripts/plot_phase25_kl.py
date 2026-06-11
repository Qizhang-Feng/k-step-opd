#!/usr/bin/env python3
"""Plot Phase 2.5 trajectories as a vertical column of panels with shared x.

Panels (top → bottom):
    1. instant per-token reverse KL (opd_reverse_kl)
    2. accumulated KL entering advantage (= -advantage)
    3. KL from reference policy (train/kl_loss, logging-only)
    4. grad_norm (own panel, log-y)
    5. response length (avg per rollout)
    6. truncation rate

All panels share the same x-axis (rollout id) so events line up across rows.

Run: python3 scripts/plot_phase25_kl.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.join("kl_analysis", "phase25")
RUNS = [
    ("kl_B.csv",   "opd-4b-B (instant K=1)",     "#888888"),
    ("kl_R4.csv",  "R4 (mean K=4, no mask)",     "#2ca02c"),
    ("kl_R1.csv",  "R1 (mean K=8, no mask)",     "#d62728"),
    ("kl_R3.csv",  "R3 (sum K=8 + mask)",        "#1f77b4"),
    ("kl_R3b.csv", "R3b (mean K=8 + mask)",      "#ff7f0e"),
]


def load(path):
    cols = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v not in ("", None) else np.nan)
    return {k: np.array(v) for k, v in cols.items()}


def smooth(y, w=11):
    """Centered moving average that shrinks at the edges (no zero-pad bias)."""
    y = np.asarray(y, dtype=float)
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

    panels = [
        ("instant_kl", "Instant reverse KL  (rollout/opd_reverse_kl)\nlog π_S(a) − log π_T(a)  per token", "linear"),
        ("accum_kl",   "Accumulated KL entering advantage  (= − rollout/advantages)",                       "linear"),
        ("kl_loss",    "KL from reference  (train/kl_loss, coef=0, logging-only)",                          "linear"),
        ("grad_norm",  "Gradient norm  (train/grad_norm)",                                                  "log"),
        ("resp_len",   "Average response length (tokens / rollout)",                                        "linear"),
        ("truncated",  "Truncation rate  (frac of samples hitting max_response_len=4096)",                  "linear"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.6 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (col, title, yscale) in zip(axes, panels):
        for fn, label, color in RUNS:
            d = data[label]
            if col not in d:
                continue
            y_raw = d[col]
            y_smooth = smooth(y_raw)
            ax.plot(d["id"], y_raw, color=color, alpha=0.18, lw=0.7)
            ax.plot(d["id"], y_smooth, color=color, lw=1.7, label=label)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        if yscale == "log":
            ax.set_yscale("log")
        ax.set_ylabel(col)

    axes[-1].set_xlabel("rollout id")
    axes[0].legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.95)

    fig.suptitle("Phase 2.5 — training trajectories aligned on rollout id", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out = os.path.join(HERE, "phase25_trajectories_vstack.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # Numeric summary (last 10 rollouts mean)
    print("\n=== Final (mean of last 10 rollouts) ===")
    cols = ["instant_kl", "accum_kl", "kl_loss", "grad_norm", "resp_len", "truncated"]
    header = " ".join(f"{c:>10s}" for c in cols)
    print(f"{'run':30s} {header}")
    for fn, label, _color in RUNS:
        d = data[label]
        vals = []
        for c in cols:
            if c in d:
                vals.append(f"{np.nanmean(d[c][-10:]):>10.4f}")
            else:
                vals.append(f"{'—':>10s}")
        print(f"{label:30s} {' '.join(vals)}")


if __name__ == "__main__":
    main()
