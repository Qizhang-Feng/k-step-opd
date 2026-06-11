"""Sparsity-aware analysis of reverse_kl temporal structure.

The naive autocorr ρ(d) is dominated by zeros (filler tokens), giving
ρ ≈ 1/spike_spacing regardless of true causal structure. We need:

  Metric A:  autocorr of |reverse_kl| (energy/magnitude signal)
  Metric B:  conditional autocorr — among active tokens only
  Metric C:  spacing distribution of active tokens (decision tokens)
  Metric D:  spike-window probability — frac of K-windows containing ≥1 spike

Run: docker exec -it k-step-opd python3 /tmp/analyze_kstep_v2.py
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
    ("R4 (mean K=4 no mask)",    "opd-4b-R4-meanK4",         "#2ca02c"),
    ("R3b (mean K=8 + mask)",    "opd-4b-R3b-meanK8-mask",   "#ff7f0e"),
    ("R5 (mean K=8 + soft)",     "opd-4b-R5-meanK8-softmask","#9467bd"),
]

D_MAX = 32
THRESHOLDS = [0.01, 0.05, 0.1, 0.3]   # active-token thresholds on |reverse_kl|


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
                if rk.size > D_MAX + 4:
                    out.append(rk)
                if len(out) >= max_samples:
                    return out
    return out


def autocorr(samples, d_max, transform=None):
    """Average autocorrelation up to lag d_max across samples after `transform`."""
    rho = np.zeros(d_max + 1)
    counts = np.zeros(d_max + 1, dtype=np.int64)
    for x in samples:
        if len(x) < d_max + 4:
            continue
        if transform is not None:
            x = transform(x)
        x = x - x.mean()
        var = x.var()
        if var < 1e-12:
            continue
        for d in range(d_max + 1):
            n = len(x) - d
            if n <= 0:
                continue
            cov = (x[: n] * x[d : d + n]).mean()
            rho[d] += cov / var
            counts[d] += 1
    valid = counts > 0
    out = np.zeros(d_max + 1)
    out[valid] = rho[valid] / counts[valid]
    return out


def conditional_pairwise(samples, d_max, threshold):
    """For each lag d, average product of x[t]*x[t+d] over pairs where BOTH are
    'active' (|x|>thr). Return both correlation (Pearson on active pairs) and
    P(both active at lag d)."""
    rho_active = np.zeros(d_max + 1)
    p_both = np.zeros(d_max + 1)
    counts = np.zeros(d_max + 1, dtype=np.int64)

    for x in samples:
        if len(x) < d_max + 4:
            continue
        active = np.abs(x) > threshold
        n_total = len(x)
        for d in range(d_max + 1):
            n = n_total - d
            mask = active[:n] & active[d : d + n]
            n_pairs = mask.sum()
            n_pos = n
            if n_pos == 0:
                continue
            p_both[d] += n_pairs / n_pos
            if n_pairs >= 2:
                a = x[:n][mask]
                b = x[d : d + n][mask]
                # Pearson on the pair set (after subtracting their own means)
                ma, mb = a.mean(), b.mean()
                num = ((a - ma) * (b - mb)).sum()
                den = np.sqrt(((a - ma) ** 2).sum() * ((b - mb) ** 2).sum())
                if den > 1e-12:
                    rho_active[d] += num / den
                    counts[d] += 1

    valid = counts > 0
    rho_out = np.zeros(d_max + 1)
    rho_out[valid] = rho_active[valid] / counts[valid]
    p_both /= max(1, len(samples))
    return rho_out, p_both


def spike_spacing(samples, threshold):
    """Distribution of distances between consecutive active tokens."""
    spacings = []
    for x in samples:
        active_idx = np.where(np.abs(x) > threshold)[0]
        if len(active_idx) > 1:
            spacings.extend(np.diff(active_idx).tolist())
    return np.asarray(spacings)


def k_window_active(samples, K_list, threshold):
    """Prob that a K-window contains at least one active token."""
    out = {}
    for K in K_list:
        n_pos = 0
        n_with_active = 0
        for x in samples:
            if len(x) < K + 1:
                continue
            active = (np.abs(x) > threshold).astype(np.int8)
            cs = np.cumsum(active)
            n_active_in_window = cs[K - 1 :] - np.concatenate([[0], cs[: -K]])
            n_active_in_window = n_active_in_window[: len(active) - K + 1]
            n_pos += len(n_active_in_window)
            n_with_active += (n_active_in_window > 0).sum()
        out[K] = n_with_active / max(1, n_pos)
    return out


def main():
    summary = {}

    fig_a, ax_a = plt.subplots(figsize=(10, 5.5))
    fig_b, axes_b = plt.subplots(1, 2, figsize=(15, 5.5))
    fig_c, ax_c = plt.subplots(figsize=(10, 5.5))
    fig_d, ax_d = plt.subplots(figsize=(10, 5.5))

    K_LIST = [1, 2, 4, 8, 16, 32, 64]
    THR = 0.05  # main threshold for "active token"

    for label, name, color in RUNS:
        run_dir = os.path.join(BASE, name)
        rollouts = list_rollouts(run_dir)
        if not rollouts:
            continue
        # Pool first 100 samples across all rollouts
        all_samples = []
        for rid, paths in rollouts[::3]:  # every 3rd rollout
            all_samples.extend(load_arrays(paths, max_samples=20))
        if not all_samples:
            continue
        print(f"[{label}] {len(all_samples)} samples")

        # Sparsity stats
        all_vals = np.concatenate([np.abs(x) for x in all_samples])
        sparsity = (all_vals == 0).mean()

        # Metric A: autocorr of |x|
        rho_abs = autocorr(all_samples, D_MAX, transform=np.abs)
        # raw autocorr (for reference)
        rho_raw = autocorr(all_samples, D_MAX, transform=None)
        # squared (energy)
        rho_sq = autocorr(all_samples, D_MAX, transform=lambda x: x ** 2)

        # Metric B: conditional autocorr & P(both active)
        rho_active, p_both = conditional_pairwise(all_samples, D_MAX, THR)

        # Metric C: spike spacing
        spacings = spike_spacing(all_samples, THR)
        spacing_summary = {}
        if len(spacings):
            spacing_summary = {
                "median": float(np.median(spacings)),
                "mean": float(np.mean(spacings)),
                "p25": float(np.quantile(spacings, 0.25)),
                "p75": float(np.quantile(spacings, 0.75)),
            }

        # Metric D: K-window active probability
        kw = k_window_active(all_samples, K_LIST, THR)

        summary[label] = {
            "sparsity": float(sparsity),
            "active_frac": float((all_vals > THR).mean()),
            "spacing": spacing_summary,
            "rho_abs": rho_abs.tolist(),
            "rho_raw": rho_raw.tolist(),
            "rho_sq": rho_sq.tolist(),
            "rho_active_conditional": rho_active.tolist(),
            "p_both_active": p_both.tolist(),
            "k_window_has_active": kw,
        }

        # Plot A: |reverse_kl| autocorr
        ax_a.plot(range(D_MAX + 1), rho_abs, "o-", color=color, lw=1.5, ms=3, label=label)

        # Plot B: P(both active at lag d) and conditional-active correlation
        axes_b[0].plot(range(D_MAX + 1), p_both, "o-", color=color, lw=1.4, ms=3, label=label)
        axes_b[1].plot(range(D_MAX + 1), rho_active, "o-", color=color, lw=1.4, ms=3, label=label)

        # Plot C: K-window active probability
        ks = sorted(kw.keys())
        ax_c.plot(ks, [kw[k] for k in ks], "o-", color=color, lw=1.6, ms=4, label=label)

        # Plot D: spacing histogram
        if len(spacings):
            ax_d.hist(np.clip(spacings, 0, 200), bins=50, alpha=0.35, color=color, label=label, density=True)

    # Finalize
    for ax in [ax_a]:
        ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
        ax.axhline(0.1, color="red", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel("lag d (tokens)")
        ax.set_ylabel("autocorrelation ρ(d)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    ax_a.set_title("Autocorrelation of |reverse_kl|  (energy/magnitude signal)\nshould be > 0 if decision tokens cluster within reasoning chunks")
    fig_a.tight_layout(); fig_a.savefig("/tmp/kstep_abs_autocorr.png", dpi=130)

    axes_b[0].set_title(f"P(both tokens active at lag d)  threshold |x|>{THR}")
    axes_b[0].set_xlabel("lag d")
    axes_b[0].set_ylabel("probability")
    axes_b[0].legend(fontsize=9); axes_b[0].grid(alpha=0.3)
    axes_b[1].set_title("Conditional Pearson(x[t], x[t+d])  on active-active pairs")
    axes_b[1].set_xlabel("lag d")
    axes_b[1].set_ylabel("ρ | both active")
    axes_b[1].legend(fontsize=9); axes_b[1].grid(alpha=0.3)
    axes_b[1].axhline(0, color="grey", lw=0.5)
    fig_b.tight_layout(); fig_b.savefig("/tmp/kstep_conditional.png", dpi=130)

    ax_c.set_xscale("log", base=2)
    ax_c.set_xlabel("K (window size)")
    ax_c.set_ylabel(f"P(K-window contains ≥1 active token)  thr |x|>{THR}")
    ax_c.set_title("How often does a K-window catch a decision token?\nK below this curve's elbow → most windows are pure-zero noise")
    ax_c.legend(fontsize=9); ax_c.grid(alpha=0.3, which="both")
    ax_c.set_ylim(0, 1.02)
    fig_c.tight_layout(); fig_c.savefig("/tmp/kstep_window_coverage.png", dpi=130)

    ax_d.set_xlabel(f"spacing between consecutive active tokens (|x|>{THR})")
    ax_d.set_ylabel("density")
    ax_d.set_title("Decision-token spacing distribution")
    ax_d.legend(fontsize=9); ax_d.grid(alpha=0.3)
    fig_d.tight_layout(); fig_d.savefig("/tmp/kstep_spacing.png", dpi=130)

    with open("/tmp/kstep_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved /tmp/kstep_v2_summary.json")
    print("saved 4 figures: /tmp/kstep_{abs_autocorr,conditional,window_coverage,spacing}.png")

    # Numeric report
    print()
    print(f"{'run':30s} {'frac=0':>7s} {'active%(>0.05)':>15s} {'spacing_med':>12s} "
          f"{'ρ|x|(1)':>10s} {'ρ|x|(4)':>10s} {'ρ|x|(8)':>10s} {'ρ|x|(16)':>10s}")
    for label, _name, _ in RUNS:
        if label not in summary:
            continue
        s = summary[label]
        spacing_med = s["spacing"].get("median", float("nan"))
        rho = s["rho_abs"]
        print(f"{label:30s} {s['sparsity']:>7.3f} {s['active_frac']:>15.4f} "
              f"{spacing_med:>12.1f} "
              f"{rho[1]:>10.4f} {rho[4]:>10.4f} {rho[8]:>10.4f} {rho[16]:>10.4f}")
    print()
    print(f"K-window active probability (P that K consecutive tokens contain ≥1 active):")
    print(f"{'run':30s} " + " ".join(f"{f'K={K}':>7s}" for K in K_LIST))
    for label, _name, _ in RUNS:
        if label not in summary:
            continue
        kw = summary[label]["k_window_has_active"]
        print(f"{label:30s} " + " ".join(f"{kw[K]:>7.3f}" for K in K_LIST))


if __name__ == "__main__":
    main()
