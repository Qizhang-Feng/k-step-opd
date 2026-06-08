#!/usr/bin/env python3
"""K-step cumulative KL analysis over training rollouts.

Extends `analyze_kstep_kl.py` with per-rollout temporal aggregation:
  - For each rollout, compute mean(Pearson r) and mean(var_ratio) across the
    32 dumped sequences for K ∈ {2,4,8,16,32,full}.
  - Plot these stats over rollout id.
  - Also plot per-K cumulative KL distribution (mean / p90 / p99) over time.

Output:
  kl_analysis/figures/kstep_over_time_<run>.png    per-run 4-panel
  kl_analysis/figures/kstep_over_time_compare.png  A vs B (same K=8)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_filename(name: str) -> tuple[int, int]:
    stem = name.replace(".jsonl", "")
    parts = stem.split("_")
    return int(parts[0][1:]), int(parts[1].replace("rank", ""))


def list_rollout_ids(dump_dir: Path) -> list[int]:
    return sorted({parse_filename(p.name)[0] for p in dump_dir.glob("r*_rank*.jsonl")})


def load_samples(dump_dir: Path, rid: int) -> list[dict]:
    samples = []
    for rk in range(4):
        f = dump_dir / f"r{rid}_rank{rk}.jsonl"
        if not f.exists():
            continue
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    return samples


def cumulative_window_mean(kl: np.ndarray, K: int | None) -> np.ndarray:
    n = len(kl)
    if n == 0:
        return kl
    csum = np.concatenate([[0.0], np.cumsum(kl)])
    if K is None:
        denom = (n - np.arange(n)).astype(np.float64)
        return (csum[n] - csum[:n]) / denom
    end = np.minimum(np.arange(n) + K, n)
    start = np.arange(n)
    denom = (end - start).astype(np.float64)
    return (csum[end] - csum[start]) / denom


def per_rollout_kstep_stats(samples: list[dict], Ks: list) -> dict:
    """For one rollout, return per-K stats aggregated over its samples."""
    out = {k: {"pearson": [], "var_ratio": [], "kl_mean": [], "kl_p99": []} for k in Ks}
    for s in samples:
        r = np.asarray(s["reverse_kl"], dtype=np.float32)
        if r.size < 16:
            continue
        instant_var = float(r.var())
        for K in Ks:
            fkl = cumulative_window_mean(r, K)
            if r.std() > 1e-9 and fkl.std() > 1e-9:
                out[K]["pearson"].append(float(np.corrcoef(r, fkl)[0, 1]))
            out[K]["var_ratio"].append(float(fkl.var()) / max(instant_var, 1e-12))
            out[K]["kl_mean"].append(float(fkl.mean()))
            out[K]["kl_p99"].append(float(np.percentile(fkl, 99)))
    return out


def analyze_run(dump_dir: Path, name: str, Ks: list) -> dict:
    rids = list_rollout_ids(dump_dir)
    print(f"[{name}] {len(rids)} rollouts: r{rids[0]}..r{rids[-1]}")

    series = {k: {"rid": [], "pearson_mean": [], "var_ratio_mean": [], "kl_mean_mean": [], "kl_p99_mean": []} for k in Ks}

    for rid in rids:
        samples = load_samples(dump_dir, rid)
        if not samples:
            continue
        stats = per_rollout_kstep_stats(samples, Ks)
        for K in Ks:
            series[K]["rid"].append(rid)
            for s_key, t_key in [("pearson", "pearson_mean"), ("var_ratio", "var_ratio_mean"),
                                 ("kl_mean", "kl_mean_mean"), ("kl_p99", "kl_p99_mean")]:
                vs = stats[K][s_key]
                series[K][t_key].append(float(np.mean(vs)) if vs else float("nan"))

    return {"name": name, "Ks": Ks, "series": series}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

K_COLORS = {
    1: "#444444",
    2: "tab:cyan",
    4: "tab:blue",
    8: "tab:red",
    16: "tab:purple",
    32: "tab:olive",
    None: "tab:brown",
}
K_LABELS = {None: "K=full"} | {k: f"K={k}" for k in [1, 2, 4, 8, 16, 32]}


def plot_run(result: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    name = result["name"]
    Ks = result["Ks"]

    # 1. Cumulative-K KL mean over time (per K)
    ax = axes[0, 0]
    for K in Ks:
        s = result["series"][K]
        ax.plot(s["rid"], s["kl_mean_mean"], "-o", color=K_COLORS[K], lw=1.5, ms=4, alpha=0.85, label=K_LABELS[K])
    ax.set_xlabel("rollout id")
    ax.set_ylabel("mean K-step cumulative KL")
    ax.set_yscale("log")
    ax.set_title(f"[{name}] mean K-step KL over training")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. Cumulative-K KL p99 over time
    ax = axes[0, 1]
    for K in Ks:
        s = result["series"][K]
        ax.plot(s["rid"], s["kl_p99_mean"], "-o", color=K_COLORS[K], lw=1.5, ms=4, alpha=0.85, label=K_LABELS[K])
    ax.set_xlabel("rollout id")
    ax.set_ylabel("mean p99 of K-step cumulative KL")
    ax.set_yscale("log")
    ax.set_title(f"[{name}] p99 K-step KL — outlier control over training")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3. Pearson r(instant, K-step) over time
    ax = axes[1, 0]
    for K in Ks:
        if K is None or K == 1:
            continue  # K=1 r=1 by definition; full r noisy
        s = result["series"][K]
        ax.plot(s["rid"], s["pearson_mean"], "-o", color=K_COLORS[K], lw=1.5, ms=4, alpha=0.85, label=K_LABELS[K])
    ax.set_xlabel("rollout id")
    ax.set_ylabel("Pearson r (instant, K-step)")
    ax.set_ylim(0, 1)
    ax.set_title(f"[{name}] Pearson r (instant, K-step) over training")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 4. var(K-step) / var(instant) over time
    ax = axes[1, 1]
    for K in Ks:
        s = result["series"][K]
        ax.plot(s["rid"], s["var_ratio_mean"], "-o", color=K_COLORS[K], lw=1.5, ms=4, alpha=0.85, label=K_LABELS[K])
    ax.set_xlabel("rollout id")
    ax.set_ylabel("var(K-step) / var(instant)")
    ax.set_yscale("log")
    ax.set_title(f"[{name}] variance reduction by K, over training")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"{name} — K-step KL stats over training rollouts", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{name}] saved {out_path}")


def plot_compare(result_a: dict, result_b: dict, out_path: Path) -> None:
    """K=1 (instant), K=8, K=full mean KL over training, A vs B."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Instant (K=2 as proxy of "near-instant"... actually use K=2 mean) vs K=8 vs K=full mean KL
    # Use K=2 as the "smallest meaningful" since K=1 = instant which equals reverse_kl[t]
    # We have instant from per-rollout summary stats elsewhere; for fairness use K=2 here.
    ax = axes[0, 0]
    for run, color, marker in [(result_a, "tab:blue", "o"), (result_b, "tab:orange", "s")]:
        s2 = run["series"][2]
        s8 = run["series"][8]
        sf = run["series"][None]
        max_rid = max(s2["rid"]) if s2["rid"] else 1
        x = [r / max_rid for r in s2["rid"]]
        ax.plot(x, s2["kl_mean_mean"], "-", color=color, alpha=0.6, lw=1.5, label=f"{run['name']} K=2")
        ax.plot(x, s8["kl_mean_mean"], "--", color=color, lw=2.0, label=f"{run['name']} K=8")
        ax.plot(x, sf["kl_mean_mean"], ":", color=color, lw=2.0, label=f"{run['name']} K=full")
    ax.set_xlabel("training fraction")
    ax.set_ylabel("mean K-step KL")
    ax.set_yscale("log")
    ax.set_title("Mean K-step KL over training — A vs B (K=2/8/full)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. K=8 only — clean comparison
    ax = axes[0, 1]
    for run, color in [(result_a, "tab:blue"), (result_b, "tab:orange")]:
        s = run["series"][8]
        max_rid = max(s["rid"]) if s["rid"] else 1
        x = [r / max_rid for r in s["rid"]]
        ax.plot(x, s["kl_mean_mean"], "-o", color=color, lw=2, label=f"{run['name']} K=8 mean")
        ax.plot(x, s["kl_p99_mean"], "--^", color=color, lw=1.2, alpha=0.7, label=f"{run['name']} K=8 p99")
    ax.set_xlabel("training fraction")
    ax.set_ylabel("K=8 cumulative KL")
    ax.set_yscale("log")
    ax.set_title("K=8 mean & p99 — A vs B")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3. Pearson r — K=8 over training
    ax = axes[1, 0]
    for run, color in [(result_a, "tab:blue"), (result_b, "tab:orange")]:
        s = run["series"][8]
        max_rid = max(s["rid"]) if s["rid"] else 1
        x = [r / max_rid for r in s["rid"]]
        ax.plot(x, s["pearson_mean"], "-o", color=color, lw=2, label=run["name"])
    ax.set_xlabel("training fraction")
    ax.set_ylabel("Pearson r (instant, K=8)")
    ax.set_ylim(0, 1)
    ax.set_title("K=8 Pearson r over training — does the structure change?")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Variance reduction K=8 over training
    ax = axes[1, 1]
    for run, color in [(result_a, "tab:blue"), (result_b, "tab:orange")]:
        s = run["series"][8]
        max_rid = max(s["rid"]) if s["rid"] else 1
        x = [r / max_rid for r in s["rid"]]
        ax.plot(x, s["var_ratio_mean"], "-o", color=color, lw=2, label=run["name"])
    ax.set_xlabel("training fraction")
    ax.set_ylabel("var(K=8) / var(instant)")
    ax.set_yscale("log")
    ax.set_title("K=8 variance reduction over training")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("K-step KL over training — opd-4b-A vs opd-4b-B", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"compare saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump-a", default="kl_analysis/dumps_A")
    p.add_argument("--dump-b", default="kl_analysis/dumps_B")
    p.add_argument("--out-dir", default="kl_analysis")
    args = p.parse_args()

    Ks = [2, 4, 8, 16, 32, None]
    fig_dir = Path(args.out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    result_a = analyze_run(Path(args.dump_a), "opd-4b-A", Ks)
    result_b = analyze_run(Path(args.dump_b), "opd-4b-B", Ks)

    plot_run(result_a, fig_dir / "kstep_over_time_A.png")
    plot_run(result_b, fig_dir / "kstep_over_time_B.png")
    plot_compare(result_a, result_b, fig_dir / "kstep_over_time_compare.png")


if __name__ == "__main__":
    main()
