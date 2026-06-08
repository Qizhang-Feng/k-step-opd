#!/usr/bin/env python3
"""Plot Instant OPD vs Lightning recipe OPD training trajectories.

Sources:
  - Lightning recipe: kl_analysis/trajectories/lightning.csv (600 rollouts, full per-rollout aggregates)
  - Instant: kl_analysis/summary.json (29 sampled rollouts, only mean instant_kl)

Lightning trajectory has full metrics (loss, grad_norm, kl_loss, ...).
Instant trajectory only has aggregate KL stats from the dump samples (mean/p99/max etc).

Output:
  kl_analysis/figures/opd_trajectories_kl.png
  kl_analysis/figures/opd_trajectories_lightning_full.png
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Load Lightning trajectory CSV
# ---------------------------------------------------------------------------


def load_lightning_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    out = {}
    for k in rows[0].keys():
        out[k] = []
        for r in rows:
            v = r.get(k, "")
            try:
                out[k].append(float(v) if v else np.nan)
            except ValueError:
                out[k].append(np.nan)
    return out


# ---------------------------------------------------------------------------
# Load Instant from summary.json
# ---------------------------------------------------------------------------


def load_instant_summary(path):
    with open(path) as f:
        d = json.load(f)
    per_r = d["per_rollout_instant_kl"]
    rids = [r["rollout_id"] for r in per_r]
    means = [r["mean"] for r in per_r]
    p90s = [r["p90"] for r in per_r]
    p99s = [r["p99"] for r in per_r]
    return {"id": rids, "mean": means, "p90": p90s, "p99": p99s}


# ---------------------------------------------------------------------------
# Plot 1: KL trajectory comparison
# ---------------------------------------------------------------------------


def plot_kl_comparison(lightning, instant, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: opd_reverse_kl (per-rollout mean)
    ax = axes[0]
    ax.plot(
        lightning["id"],
        lightning["rollout/opd_reverse_kl"],
        "-",
        color="tab:orange",
        lw=1,
        alpha=0.4,
        label=None,
    )
    # smooth
    kl = np.array(lightning["rollout/opd_reverse_kl"])
    valid = ~np.isnan(kl)
    ids_arr = np.array(lightning["id"])
    smooth = np.convolve(kl[valid], np.ones(20) / 20, mode="valid")
    ax.plot(
        ids_arr[valid][9 : 9 + len(smooth)],
        smooth,
        "-",
        color="tab:orange",
        lw=2.5,
        label="opd-4b-B (paper recipe, lr=2e-6, max_resp=4096, T=0.8, 600 rollouts)",
    )

    ax.plot(
        instant["id"],
        instant["mean"],
        "-o",
        color="tab:blue",
        lw=2,
        label="opd-4b-A (conservative, lr=5e-7, max_resp=8192, T=0.6, 300 rollouts) — sampled per 10",
    )

    ax.set_xlabel("rollout id")
    ax.set_ylabel("opd reverse KL  (mean per token)")
    ax.set_title("OPD reverse KL — training trajectory")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Right: KL distribution (p90, p99 over time, log scale)
    ax = axes[1]
    ax.plot(instant["id"], instant["mean"], "-o", color="tab:blue", lw=1, label="opd-4b-A mean")
    ax.plot(instant["id"], instant["p90"], "-s", color="tab:cyan", lw=1, label="opd-4b-A p90")
    ax.plot(instant["id"], instant["p99"], "-^", color="navy", lw=1, label="opd-4b-A p99")
    ax.set_xlabel("rollout id")
    ax.set_ylabel("instant reverse KL")
    ax.set_yscale("log")
    ax.set_title("opd-4b-A — KL distribution (mean / p90 / p99)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Lightning full metrics (loss, grad_norm, response_length, etc)
# ---------------------------------------------------------------------------


def plot_lightning_full(lightning, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ids = lightning["id"]

    panels = [
        ("rollout/opd_reverse_kl", "OPD reverse KL", "tab:orange", True),
        ("train/loss", "train/loss (= reverse_kl, scalar)", "tab:red", False),
        ("train/grad_norm", "grad_norm", "tab:green", False),
        ("rollout/truncated", "truncated_ratio", "tab:purple", False),
        ("rollout/response_lengths", "avg response length (tokens)", "tab:brown", False),
        ("train/kl_loss", "kl_loss vs ref (logged but coef=0)", "tab:olive", False),
    ]

    for ax, (key, label, color, log) in zip(axes.flat, panels):
        y = np.array(lightning[key])
        ax.plot(ids, y, "-", color=color, lw=0.8, alpha=0.4)
        valid = ~np.isnan(y)
        if valid.sum() > 20:
            smooth = np.convolve(y[valid], np.ones(20) / 20, mode="valid")
            ax.plot(
                np.array(ids)[valid][9 : 9 + len(smooth)],
                smooth,
                color=color,
                lw=2.5,
                label="20-rollout MA",
            )
        if log:
            ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("rollout id")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("opd-4b-B (paper recipe) — full training trajectory (600 rollouts)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    out_dir = Path("kl_analysis/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    lightning = load_lightning_csv("kl_analysis/trajectories/lightning.csv")
    instant = load_instant_summary("kl_analysis/summary.json")

    plot_kl_comparison(lightning, instant, out_dir / "opd_trajectories_kl.png")
    plot_lightning_full(lightning, out_dir / "opd_trajectories_lightning_full.png")

    # Print a summary table
    print("\n=== Summary ===")
    print(f"Lightning OPD: {len(lightning['id'])} rollouts (full metrics per rollout)")
    print(f"  Final opd_reverse_kl: {lightning['rollout/opd_reverse_kl'][-1]:.4f}")
    print(f"  Final loss: {lightning['train/loss'][-1]:.4f}")
    print(f"  Final grad_norm: {lightning['train/grad_norm'][-1]:.4f}")
    print(f"  Avg truncated_ratio: {np.nanmean(lightning['rollout/truncated']):.3f}")
    print(f"  Avg response_length: {np.nanmean(lightning['rollout/response_lengths']):.0f}")
    print()
    print(f"Instant OPD: {len(instant['id'])} sampled rollouts (every 10)")
    print(f"  First (r10) mean instant_kl: {instant['mean'][0]:.4f}")
    print(f"  Last (r{instant['id'][-1]}) mean instant_kl: {instant['mean'][-1]:.4f}")


if __name__ == "__main__":
    main()
