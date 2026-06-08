#!/usr/bin/env python3
"""Show the SUM vs MEAN K-step KL variance scaling.

For each sample, at each position t, compute:
  sum_K[t]  = Σ_{d=0}^{K-1} reverse_kl[t+d]
  mean_K[t] = sum_K[t] / K

Aggregate per-K statistics across all positions and samples (in midpoint rollout).

Output:
  kl_analysis/figures/kstep_sum_vs_mean.png  — 4-panel: var, mean, std/mean (CV), distribution
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "scripts")
from analyze_kstep_kl_over_time import list_rollout_ids, load_samples


def compute_sum_at_each_t(kl: np.ndarray, K: int | None) -> np.ndarray:
    """Σ_{d=0}^{K-1} kl[t+d], clamped at end. K=None = full suffix."""
    n = len(kl)
    if n == 0:
        return kl
    csum = np.concatenate([[0.0], np.cumsum(kl)])
    if K is None:
        return csum[n] - csum[:n]
    end = np.minimum(np.arange(n) + K, n)
    return csum[end] - csum[:n]


def analyze(dump_dir: Path, name: str, Ks: list, midpoint_rid: int):
    samples = load_samples(dump_dir, midpoint_rid)
    print(f"[{name}] r{midpoint_rid}: {len(samples)} samples")

    # For each K, collect ALL sum_K[t] across all samples and positions
    sum_all = {k: [] for k in Ks}
    for s in samples:
        kl = np.asarray(s["reverse_kl"], dtype=np.float32)
        if kl.size < 32:
            continue
        for K in Ks:
            sk = compute_sum_at_each_t(kl, K)
            sum_all[K].append(sk)
    sum_all = {k: np.concatenate(v) if v else np.array([]) for k, v in sum_all.items()}

    # Stats
    stats_sum = {}
    stats_mean = {}
    for K in Ks:
        s = sum_all[K]
        if s.size == 0:
            continue
        stats_sum[K] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "var": float(s.var()),
            "p99": float(np.percentile(s, 99)),
            "max": float(s.max()),
            "n": int(s.size),
        }
        m = s if K == 1 else (s / K) if K is not None else (s / 1)  # mean version
        # For K=full, normalize by 1 (treat as sum, can compare)
        if K is not None and K != 1:
            m = s / K
        elif K is None:
            # for full suffix mean, divide by length-of-suffix per sample (handled per-sample)
            # approximate: divide by mean window length
            # Better: recompute as suffix mean
            mean_arr = []
            for sample in samples:
                kl = np.asarray(sample["reverse_kl"], dtype=np.float32)
                if kl.size < 32:
                    continue
                csum = np.concatenate([[0.0], np.cumsum(kl)])
                denom = (kl.size - np.arange(kl.size)).astype(np.float64)
                mean_arr.append((csum[kl.size] - csum[:kl.size]) / denom)
            m = np.concatenate(mean_arr) if mean_arr else np.array([])
        stats_mean[K] = {
            "mean": float(m.mean()),
            "std": float(m.std()),
            "var": float(m.var()),
            "p99": float(np.percentile(m, 99)),
            "max": float(m.max()),
            "n": int(m.size),
        }

    return {
        "name": name,
        "midpoint_rid": midpoint_rid,
        "Ks": Ks,
        "stats_sum": stats_sum,
        "stats_mean": stats_mean,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot(result_a: dict, result_b: dict, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    Ks = result_a["Ks"]
    K_xticks = [str(k) if k is not None else "full" for k in Ks]
    K_xpos = list(range(len(Ks)))

    # 1. Var(Σ_K) — should grow with K (linear-ish)
    ax = axes[0, 0]
    for run, color, mk in [(result_a, "tab:blue", "o"), (result_b, "tab:orange", "s")]:
        v = [run["stats_sum"][k]["var"] if k in run["stats_sum"] else None for k in Ks]
        ax.plot(K_xpos, v, marker=mk, color=color, lw=2, ms=8, label=run["name"])
    # Add ideal K-linear reference (anchored at K=2)
    if 2 in result_a["stats_sum"]:
        v0 = result_a["stats_sum"][2]["var"]
        K_nums = [k if k is not None else 4096 for k in Ks]
        ref = [v0 * k / 2 if k != 4096 else None for k in K_nums]
        ax.plot(K_xpos[:-1], ref[:-1], "--k", lw=1, alpha=0.4, label="∝K (independent ideal)")
    ax.set_xticks(K_xpos)
    ax.set_xticklabels(K_xticks)
    ax.set_xlabel("K (window size)")
    ax.set_ylabel("Var( Σ_{d<K} reverse_kl[t+d] )")
    ax.set_yscale("log")
    ax.set_title("Variance of K-step SUM — grows with K (REINFORCE-style)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 2. Var(mean_K) — should shrink with K (1/K-ish)
    ax = axes[0, 1]
    for run, color, mk in [(result_a, "tab:blue", "o"), (result_b, "tab:orange", "s")]:
        v = [run["stats_mean"][k]["var"] if k in run["stats_mean"] else None for k in Ks]
        ax.plot(K_xpos, v, marker=mk, color=color, lw=2, ms=8, label=run["name"])
    if 2 in result_a["stats_mean"]:
        v0 = result_a["stats_mean"][2]["var"]
        K_nums = [k if k is not None else 4096 for k in Ks]
        ref = [v0 * 2 / k if k != 4096 else None for k in K_nums]
        ax.plot(K_xpos[:-1], ref[:-1], "--k", lw=1, alpha=0.4, label="∝1/K (independent ideal)")
    ax.set_xticks(K_xpos)
    ax.set_xticklabels(K_xticks)
    ax.set_xlabel("K (window size)")
    ax.set_ylabel("Var( mean_{d<K} reverse_kl[t+d] )")
    ax.set_yscale("log")
    ax.set_title("Variance of K-step MEAN — shrinks with K (averaging)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 3. Mean of Σ_K (= K * mean_K). Should grow ~linear with K
    ax = axes[1, 0]
    for run, color, mk in [(result_a, "tab:blue", "o"), (result_b, "tab:orange", "s")]:
        v = [run["stats_sum"][k]["mean"] if k in run["stats_sum"] else None for k in Ks]
        ax.plot(K_xpos, v, marker=mk, color=color, lw=2, ms=8, label=run["name"])
    ax.set_xticks(K_xpos)
    ax.set_xticklabels(K_xticks)
    ax.set_xlabel("K (window size)")
    ax.set_ylabel("E[ Σ_{d<K} reverse_kl[t+d] ]")
    ax.set_yscale("log")
    ax.set_title("Mean of K-step SUM — magnitude grows with K → coef must scale")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 4. CV (coefficient of variation) for sum: std / mean — RL signal-to-noise
    ax = axes[1, 1]
    for run, color, mk in [(result_a, "tab:blue", "o"), (result_b, "tab:orange", "s")]:
        cv = [
            run["stats_sum"][k]["std"] / max(run["stats_sum"][k]["mean"], 1e-9)
            if k in run["stats_sum"]
            else None
            for k in Ks
        ]
        ax.plot(K_xpos, cv, marker=mk, color=color, lw=2, ms=8, label=run["name"])
    ax.set_xticks(K_xpos)
    ax.set_xticklabels(K_xticks)
    ax.set_xlabel("K (window size)")
    ax.set_ylabel("CV = std(Σ_K) / mean(Σ_K)")
    ax.set_title("Coefficient of variation for SUM — relative noise of advantage")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("K-step KL: SUM (REINFORCE-style) vs MEAN (averaging)\nbias-variance tradeoff visualized", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def print_table(result_a: dict, result_b: dict):
    print("\n=== SUM K-step KL stats (midpoint rollouts) ===")
    print(f"  {'K':>5s} | {'A.mean':>9s} {'A.var':>9s} {'A.std':>8s} | {'B.mean':>9s} {'B.var':>9s} {'B.std':>8s}")
    print("  ------+-------------------------------+---------------------------------")
    for k in result_a["Ks"]:
        a = result_a["stats_sum"].get(k)
        b = result_b["stats_sum"].get(k)
        klab = str(k) if k is not None else "full"
        if not a or not b:
            continue
        print(f"  {klab:>5s} | {a['mean']:>9.4f} {a['var']:>9.4f} {a['std']:>8.4f} | {b['mean']:>9.4f} {b['var']:>9.4f} {b['std']:>8.4f}")

    print("\n=== MEAN K-step KL stats (var should shrink) ===")
    print(f"  {'K':>5s} | {'A.var':>9s} | {'B.var':>9s}")
    for k in result_a["Ks"]:
        a = result_a["stats_mean"].get(k)
        b = result_b["stats_mean"].get(k)
        klab = str(k) if k is not None else "full"
        if not a or not b:
            continue
        print(f"  {klab:>5s} | {a['var']:>9.4f} | {b['var']:>9.4f}")


def main():
    Ks = [1, 2, 4, 8, 16, 32, 64, None]
    fig_dir = Path("kl_analysis/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    rids_a = list_rollout_ids(Path("kl_analysis/dumps_A"))
    rids_b = list_rollout_ids(Path("kl_analysis/dumps_B"))
    mid_a = rids_a[len(rids_a) // 2]
    mid_b = rids_b[len(rids_b) // 2]

    result_a = analyze(Path("kl_analysis/dumps_A"), "opd-4b-A", Ks, mid_a)
    result_b = analyze(Path("kl_analysis/dumps_B"), "opd-4b-B", Ks, mid_b)

    plot(result_a, result_b, fig_dir / "kstep_sum_vs_mean.png")
    print_table(result_a, result_b)


if __name__ == "__main__":
    main()
