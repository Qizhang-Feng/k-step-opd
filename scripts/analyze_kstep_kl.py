#!/usr/bin/env python3
"""K-step cumulative KL analysis for opd-4b-A and opd-4b-B.

Key questions:
  Q1. How does instant KL distribution evolve over training? (mean / p90 / p99 / max)
  Q2. K-step cumulative KL: does it carry independent signal beyond instant?
      - Pearson r(instant_kl[t], cumulative_kl[t..t+K-1])
      - Variance reduction: var(cumulative_kl_K) / var(instant_kl)
  Q3. Where do KL outliers concentrate in the response? (position bins)
  Q4. opd-4b-A vs opd-4b-B at matched training fraction:
      - Same sampling style? Same magnitude?
      - Does opd-4b-B's "more aggressive" sampling actually drive more KL signal?

Output:
  kl_analysis/figures/kstep_kl_<run>.png       per-run 4-panel figure
  kl_analysis/figures/kstep_kl_compare.png     A vs B side-by-side
  kl_analysis/summary_<run>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def parse_filename(name: str) -> tuple[int, int]:
    stem = name.replace(".jsonl", "")
    parts = stem.split("_")
    rid = int(parts[0][1:])
    rk = int(parts[1].replace("rank", ""))
    return rid, rk


def list_rollout_ids(dump_dir: Path) -> list[int]:
    files = list(dump_dir.glob("r*_rank*.jsonl"))
    return sorted({parse_filename(p.name)[0] for p in files})


def load_samples_for_rollout(dump_dir: Path, rollout_id: int) -> list[dict]:
    samples = []
    for rk in range(4):
        f = dump_dir / f"r{rollout_id}_rank{rk}.jsonl"
        if not f.exists():
            continue
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
    return samples


# ---------------------------------------------------------------------------
# K-step cumulative reverse KL
# ---------------------------------------------------------------------------


def cumulative_window_mean(kl: np.ndarray, K: int | None) -> np.ndarray:
    """For each t, mean of kl[t..t+K-1] (clamped at end). K=None → suffix mean."""
    n = len(kl)
    if n == 0:
        return kl
    csum = np.concatenate([[0.0], np.cumsum(kl)])  # length n+1
    if K is None:
        denom = (n - np.arange(n)).astype(np.float64)
        return (csum[n] - csum[:n]) / denom
    end = np.minimum(np.arange(n) + K, n)
    start = np.arange(n)
    denom = (end - start).astype(np.float64)
    return (csum[end] - csum[start]) / denom


def percentile_summary(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_run(dump_dir: Path, name: str) -> dict:
    rids = list_rollout_ids(dump_dir)
    print(f"\n[{name}] {len(rids)} rollout dumps: r{rids[0]}..r{rids[-1]}")

    Ks = [2, 4, 8, 16, 32, None]  # None = full suffix
    K_labels = {None: "full"} | {k: str(k) for k in Ks if k is not None}

    # --- Pass 1: per-rollout instant KL stats ---
    per_rollout = []
    sample_counts = []
    for rid in rids:
        samples = load_samples_for_rollout(dump_dir, rid)
        sample_counts.append(len(samples))
        kls = [np.asarray(s["reverse_kl"], dtype=np.float32) for s in samples if s["reverse_kl"]]
        if not kls:
            continue
        flat = np.concatenate(kls)
        per_rollout.append({"rollout_id": rid, **percentile_summary(flat)})
    print(f"[{name}] sample counts: min={min(sample_counts)} max={max(sample_counts)}")

    # --- Pass 2: K-step cumulative analysis (per-sample averaging) ---
    pearson_per_K = {k: [] for k in Ks}
    snr_per_K = {k: [] for k in Ks}
    cumulative_dist_per_K = {k: [] for k in Ks}  # for the midpoint rollout

    midpoint_rid = rids[len(rids) // 2]
    print(f"[{name}] midpoint rollout for distribution: r{midpoint_rid}")

    for rid in rids:
        samples = load_samples_for_rollout(dump_dir, rid)
        for s in samples:
            r = np.asarray(s["reverse_kl"], dtype=np.float32)
            if r.size < 16:
                continue
            instant_var = float(r.var())
            for K in Ks:
                fkl = cumulative_window_mean(r, K)
                if r.std() > 1e-9 and fkl.std() > 1e-9:
                    pearson_per_K[K].append(float(np.corrcoef(r, fkl)[0, 1]))
                snr_per_K[K].append(float(fkl.var()) / max(instant_var, 1e-12))
                if rid == midpoint_rid:
                    cumulative_dist_per_K[K].append(fkl)

    # --- Pass 3: KL by relative position (using midpoint rollout) ---
    n_bins = 20
    pos_sums = np.zeros(n_bins)
    pos_counts = np.zeros(n_bins, dtype=np.int64)
    samples_mid = load_samples_for_rollout(dump_dir, midpoint_rid)
    for s in samples_mid:
        r = np.asarray(s["reverse_kl"], dtype=np.float32)
        n = r.size
        if n < n_bins:
            continue
        idx = np.minimum((np.arange(n) * n_bins) // n, n_bins - 1)
        for b in range(n_bins):
            mask = idx == b
            if mask.any():
                pos_sums[b] += r[mask].sum()
                pos_counts[b] += int(mask.sum())
    pos_mean = pos_sums / np.maximum(pos_counts, 1)

    return {
        "name": name,
        "rids": rids,
        "per_rollout": per_rollout,
        "Ks": Ks,
        "K_labels": K_labels,
        "pearson_per_K": pearson_per_K,
        "snr_per_K": snr_per_K,
        "midpoint_rid": midpoint_rid,
        "pos_mean": pos_mean,
        "n_bins": n_bins,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_run(result: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    name = result["name"]
    rids = np.array([r["rollout_id"] for r in result["per_rollout"]])

    # 1. Instant KL trajectory (mean / p90 / p99 / max)
    ax = axes[0, 0]
    means = np.array([r["mean"] for r in result["per_rollout"]])
    p90s = np.array([r["p90"] for r in result["per_rollout"]])
    p99s = np.array([r["p99"] for r in result["per_rollout"]])
    maxs = np.array([r["max"] for r in result["per_rollout"]])
    ax.plot(rids, means, "-o", color="tab:orange", lw=2, label="mean")
    ax.plot(rids, p90s, "-s", color="tab:cyan", lw=1.5, alpha=0.8, label="p90")
    ax.plot(rids, p99s, "-^", color="navy", lw=1.5, alpha=0.8, label="p99")
    ax.plot(rids, maxs, "--x", color="red", lw=1, alpha=0.6, label="max")
    ax.set_xlabel("rollout id")
    ax.set_ylabel("instant reverse KL")
    ax.set_yscale("log")
    ax.set_title(f"[{name}] instant per-token KL — training trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. Pearson r(instant, cumulative_K)
    ax = axes[0, 1]
    Ks = result["Ks"]
    labels = [result["K_labels"][k] for k in Ks]
    means_r = [
        np.mean(result["pearson_per_K"][k]) if result["pearson_per_K"][k] else 0.0 for k in Ks
    ]
    p25_r = [
        np.percentile(result["pearson_per_K"][k], 25) if result["pearson_per_K"][k] else 0.0
        for k in Ks
    ]
    p75_r = [
        np.percentile(result["pearson_per_K"][k], 75) if result["pearson_per_K"][k] else 0.0
        for k in Ks
    ]
    x = np.arange(len(Ks))
    ax.bar(
        x,
        means_r,
        yerr=[np.array(means_r) - p25_r, np.array(p75_r) - means_r],
        capsize=4,
        color="tab:purple",
    )
    ax.axhline(0.95, color="red", lw=1, ls="--", alpha=0.5, label="r=0.95 (essentially redundant)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("K-step window")
    ax.set_ylabel("Pearson r (instant_kl, K-step cumulative)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax.set_title(f"[{name}] How redundant is K-step KL vs instant?")

    # 3. Variance reduction by K
    ax = axes[1, 0]
    var_ratio_mean = [
        np.mean(result["snr_per_K"][k]) if result["snr_per_K"][k] else 0.0 for k in Ks
    ]
    ax.bar(x, var_ratio_mean, color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("K-step window")
    ax.set_ylabel("var(K-step) / var(instant)")
    ax.set_yscale("log")
    ax.set_title(f"[{name}] Variance reduction by K (lower = noise smoothed)")
    ax.grid(True, alpha=0.3, axis="y")
    # Add labels
    for i, v in enumerate(var_ratio_mean):
        ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=8)

    # 4. KL by relative position (in midpoint rollout)
    ax = axes[1, 1]
    bin_centers = (np.arange(result["n_bins"]) + 0.5) / result["n_bins"]
    ax.plot(bin_centers, result["pos_mean"], "-o", color="tab:red", lw=2)
    ax.set_xlabel("relative position in response (0=start, 1=end)")
    ax.set_ylabel("mean instant KL")
    ax.set_title(f"[{name}] KL by position (rollout r{result['midpoint_rid']})")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{name} — K-step cumulative KL analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{name}] saved: {out_path}")


def plot_comparison(result_a: dict, result_b: dict, out_path: Path) -> None:
    """Side-by-side: K-step Pearson r and variance ratio for A vs B."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Top-left: instant KL mean over training (compare both)
    ax = axes[0, 0]
    for result, color, label in [(result_a, "tab:blue", "opd-4b-A"), (result_b, "tab:orange", "opd-4b-B")]:
        rids = [r["rollout_id"] for r in result["per_rollout"]]
        means = [r["mean"] for r in result["per_rollout"]]
        # normalize x to fraction of training
        max_rid = max(rids)
        x_norm = [r / max_rid for r in rids]
        ax.plot(x_norm, means, "-o", color=color, lw=2, label=f"{label} (max r{max_rid})")
    ax.set_xlabel("training fraction (rollout_id / total)")
    ax.set_ylabel("mean instant reverse KL")
    ax.set_yscale("log")
    ax.set_title("Instant KL trajectory — A vs B (matched training fraction)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Top-right: Pearson r per K, side-by-side
    ax = axes[0, 1]
    Ks = result_a["Ks"]
    labels = [result_a["K_labels"][k] for k in Ks]
    x = np.arange(len(Ks))
    width = 0.35
    means_a = [
        np.mean(result_a["pearson_per_K"][k]) if result_a["pearson_per_K"][k] else 0.0 for k in Ks
    ]
    means_b = [
        np.mean(result_b["pearson_per_K"][k]) if result_b["pearson_per_K"][k] else 0.0 for k in Ks
    ]
    ax.bar(x - width / 2, means_a, width, color="tab:blue", label="opd-4b-A")
    ax.bar(x + width / 2, means_b, width, color="tab:orange", label="opd-4b-B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("K-step window")
    ax.set_ylabel("Pearson r (instant, K-step)")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-correlation Pearson r — A vs B")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    # Bottom-left: Variance ratio per K, side-by-side
    ax = axes[1, 0]
    var_a = [np.mean(result_a["snr_per_K"][k]) if result_a["snr_per_K"][k] else 0.0 for k in Ks]
    var_b = [np.mean(result_b["snr_per_K"][k]) if result_b["snr_per_K"][k] else 0.0 for k in Ks]
    ax.bar(x - width / 2, var_a, width, color="tab:blue", label="opd-4b-A")
    ax.bar(x + width / 2, var_b, width, color="tab:orange", label="opd-4b-B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("K-step window")
    ax.set_ylabel("var(K-step) / var(instant)")
    ax.set_yscale("log")
    ax.set_title("Variance reduction — A vs B")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    # Bottom-right: KL by position (both)
    ax = axes[1, 1]
    bin_centers = (np.arange(result_a["n_bins"]) + 0.5) / result_a["n_bins"]
    ax.plot(bin_centers, result_a["pos_mean"], "-o", color="tab:blue", lw=2, label=f"opd-4b-A (r{result_a['midpoint_rid']})")
    ax.plot(bin_centers, result_b["pos_mean"], "-o", color="tab:orange", lw=2, label=f"opd-4b-B (r{result_b['midpoint_rid']})")
    ax.set_xlabel("relative position in response (0=start, 1=end)")
    ax.set_ylabel("mean instant KL")
    ax.set_title("KL by position — A vs B (midpoint rollout)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("K-step KL analysis — opd-4b-A vs opd-4b-B", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"compare saved: {out_path}")


def write_summary(result: dict, out_path: Path) -> None:
    """Write a JSON summary (no numpy arrays)."""
    Ks = result["Ks"]
    K_labels = result["K_labels"]

    summary = {
        "name": result["name"],
        "n_rollouts": len(result["rids"]),
        "rollout_ids": result["rids"],
        "per_rollout_instant_kl": result["per_rollout"],
        "K_pearson_r_mean": {
            K_labels[k]: (float(np.mean(result["pearson_per_K"][k])) if result["pearson_per_K"][k] else None)
            for k in Ks
        },
        "K_var_ratio_mean": {
            K_labels[k]: (float(np.mean(result["snr_per_K"][k])) if result["snr_per_K"][k] else None)
            for k in Ks
        },
        "midpoint_rid": result["midpoint_rid"],
        "pos_mean_kl": result["pos_mean"].tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary saved: {out_path}")


def print_text_table(result: dict) -> None:
    name = result["name"]
    Ks = result["Ks"]
    K_labels = result["K_labels"]
    print(f"\n=== {name} — K-step analysis ===")
    print("  K     | mean Pearson r | var ratio (mean)")
    print("  ------+----------------+-----------------")
    for k in Ks:
        rs = result["pearson_per_K"][k]
        vrs = result["snr_per_K"][k]
        if not rs:
            continue
        print(f"  {K_labels[k]:>5s} |   {np.mean(rs):>7.3f}      |   {np.mean(vrs):>.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump-a", default="kl_analysis/dumps_A")
    p.add_argument("--dump-b", default="kl_analysis/dumps_B")
    p.add_argument("--out-dir", default="kl_analysis")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    result_a = analyze_run(Path(args.dump_a), "opd-4b-A")
    result_b = analyze_run(Path(args.dump_b), "opd-4b-B")

    plot_run(result_a, fig_dir / "kstep_kl_A.png")
    plot_run(result_b, fig_dir / "kstep_kl_B.png")
    plot_comparison(result_a, result_b, fig_dir / "kstep_kl_compare.png")

    write_summary(result_a, out_dir / "summary_A.json")
    write_summary(result_b, out_dir / "summary_B.json")

    print_text_table(result_a)
    print_text_table(result_b)

    # Per-rollout 简表
    print("\n=== Per-rollout instant KL trajectory ===")
    print(
        f"  {'rid':>4s} | {'A.mean':>7s} {'A.p99':>7s} | {'B.mean':>7s} {'B.p99':>7s}"
    )
    print("  -----+---------------+--------------")
    rid_a = {r["rollout_id"]: r for r in result_a["per_rollout"]}
    rid_b = {r["rollout_id"]: r for r in result_b["per_rollout"]}
    all_rids = sorted(set(rid_a) | set(rid_b))
    for rid in all_rids:
        a = rid_a.get(rid, {"mean": float("nan"), "p99": float("nan")})
        b = rid_b.get(rid, {"mean": float("nan"), "p99": float("nan")})
        print(
            f"  {rid:>4d} | {a['mean']:>7.4f} {a['p99']:>7.4f} | {b['mean']:>7.4f} {b['p99']:>7.4f}"
        )


if __name__ == "__main__":
    main()
