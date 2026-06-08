#!/usr/bin/env python3
"""Analyze per-token KL dumps from 4B Instant OPD training.

Question 1: How does instant KL evolve over training?
  - mean / median / p90 / p99 / max over the 29 dumps (r10..r290)
  - Does the tail shrink uniformly, or just the body?

Question 2: Does future KL (cumulative) carry signal beyond instant KL?
  - For each token compute future_kl_K[t] = mean over [t..t+K-1] of reverse_kl
  - Compare distribution & per-token correlation between instant_kl[t] and future_kl_K[t]
  - K ∈ {2, 4, 8, 16, full}
  - Signal-to-noise: var(future_kl) / var(instant_kl) (lower = more averaging)
  - Pearson r(instant, future_K) — higher r means future KL is mostly redundant

Question 3: Where in the response are KL outliers?
  - Bin tokens by relative position (10 bins from start to end of response)
  - Plot mean KL by position
  - Hypothesis: terminator tokens (last 5%) and structural tokens (first 1-2%
    where </think>/\\boxed{} could appear in non-degenerate responses) have
    higher KL than body tokens

Note: prompt/response_token_ids are NOT in the dumps (we only logged logprobs).
So token-type analysis (</think>, \\boxed{}) is approximated by relative position.

Output:
  kl_analysis/figures/instant_kl_over_training.png
  kl_analysis/figures/instant_kl_distribution.png
  kl_analysis/figures/future_kl_correlation.png
  kl_analysis/figures/kl_by_position.png
  kl_analysis/figures/snr_by_K.png
  kl_analysis/summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("[warn] matplotlib not available — figures will be skipped")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def parse_filename(name: str) -> tuple[int, int]:
    # e.g. "r150_rank2.jsonl" → (150, 2)
    stem = name.replace(".jsonl", "")
    parts = stem.split("_")
    rid = int(parts[0][1:])
    rk = int(parts[1].replace("rank", ""))
    return rid, rk


def iter_dump_files(dump_dir: Path) -> list[Path]:
    files = sorted(dump_dir.glob("r*_rank*.jsonl"), key=lambda p: parse_filename(p.name))
    return files


def load_samples_for_rollout(dump_dir: Path, rollout_id: int) -> list[dict]:
    samples: list[dict] = []
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


def list_rollout_ids(dump_dir: Path) -> list[int]:
    rids = sorted({parse_filename(p.name)[0] for p in iter_dump_files(dump_dir)})
    return rids


# ---------------------------------------------------------------------------
# Per-sample analysis helpers
# ---------------------------------------------------------------------------


def cumulative_future_kl(reverse_kl: np.ndarray, K: int | None = None) -> np.ndarray:
    """For each token t, compute mean reverse_kl over [t, t+1, ..., t+K-1].

    K=None means full (suffix mean).  Returns array of same length as input.
    """
    n = len(reverse_kl)
    if n == 0:
        return reverse_kl
    csum = np.concatenate([[0.0], np.cumsum(reverse_kl)])  # length n+1
    if K is None:
        # suffix mean: (csum[n] - csum[t]) / (n - t)
        denom = (n - np.arange(n)).astype(np.float64)
        return (csum[n] - csum[:n]) / denom
    # window mean of K (clamped at end)
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
# Main analysis
# ---------------------------------------------------------------------------


def analyze(dump_dir: Path, out_dir: Path, position_bins: int = 20) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    rids = list_rollout_ids(dump_dir)
    print(f"[info] found {len(rids)} rollout dumps: {rids[0]}..{rids[-1]} step")

    # --- pass 1: per-rollout instant KL stats + concat all KL for distribution ---
    per_rollout: list[dict] = []
    all_kl_pieces: list[np.ndarray] = []  # concatenate all instant KL across all dumps
    sample_counts: list[int] = []

    for rid in rids:
        samples = load_samples_for_rollout(dump_dir, rid)
        sample_counts.append(len(samples))
        rkls = []
        for s in samples:
            r = np.asarray(s["reverse_kl"], dtype=np.float32)
            if r.size > 0:
                rkls.append(r)
        if not rkls:
            continue
        flat = np.concatenate(rkls)
        all_kl_pieces.append(flat)
        per_rollout.append({"rollout_id": rid, **percentile_summary(flat)})

    print(f"[info] sample counts per rollout: min={min(sample_counts)} max={max(sample_counts)}")

    # --- pass 2: future KL analysis (sample subset to keep runtime sane) ---
    Ks = [2, 4, 8, 16, 32, None]  # None = full suffix
    K_labels = {None: "full"} | {k: str(k) for k in Ks if k is not None}
    pearson_per_K: dict = {k: [] for k in Ks}  # list of per-sample Pearson r
    snr_per_K: dict = {k: [] for k in Ks}  # var(future)/var(instant) — averaged over samples
    instant_var: list[float] = []
    future_var: dict = {k: [] for k in Ks}
    # also collect overall percentile distribution of future KL
    future_kl_pieces: dict = {k: [] for k in Ks}

    # use middle-of-training rollout (~r150) for distribution comparison
    midpoint_rid = rids[len(rids) // 2]
    print(f"[info] using rollout r{midpoint_rid} for instant-vs-future distribution comparison")

    for rid in rids:
        samples = load_samples_for_rollout(dump_dir, rid)
        for s in samples:
            r = np.asarray(s["reverse_kl"], dtype=np.float32)
            if r.size < 16:
                continue
            instant_var.append(float(r.var()))
            for K in Ks:
                fkl = cumulative_future_kl(r, K)
                # Pearson r between instant and future
                if r.std() > 1e-9 and fkl.std() > 1e-9:
                    pr = float(np.corrcoef(r, fkl)[0, 1])
                    pearson_per_K[K].append(pr)
                future_var[K].append(float(fkl.var()))
                snr_per_K[K].append(float(fkl.var()) / max(float(r.var()), 1e-12))
                if rid == midpoint_rid:
                    future_kl_pieces[K].append(fkl)

    # --- pass 3: KL by relative position (using middle rollout to keep trend clean) ---
    pos_bins = np.zeros(position_bins, dtype=np.float64)
    pos_bin_counts = np.zeros(position_bins, dtype=np.int64)
    samples_mid = load_samples_for_rollout(dump_dir, midpoint_rid)
    for s in samples_mid:
        r = np.asarray(s["reverse_kl"], dtype=np.float32)
        n = r.size
        if n < position_bins:
            continue
        # bucket each token to a bin by relative position
        idx = np.minimum((np.arange(n) * position_bins) // n, position_bins - 1)
        for b in range(position_bins):
            mask = idx == b
            if mask.any():
                pos_bins[b] += float(r[mask].sum())
                pos_bin_counts[b] += int(mask.sum())
    pos_mean_kl = pos_bins / np.maximum(pos_bin_counts, 1)

    # --- summary dict ---
    summary = {
        "dump_dir": str(dump_dir),
        "n_rollouts": len(rids),
        "rollout_ids": rids,
        "per_rollout_instant_kl": per_rollout,
        "future_kl_pearson_r_mean": {
            K_labels[k]: (float(np.mean(v)) if v else None) for k, v in pearson_per_K.items()
        },
        "future_kl_pearson_r_p50": {
            K_labels[k]: (float(np.median(v)) if v else None) for k, v in pearson_per_K.items()
        },
        "future_kl_var_ratio_mean": {
            K_labels[k]: (float(np.mean(v)) if v else None) for k, v in snr_per_K.items()
        },
        "midpoint_rollout_for_distribution": midpoint_rid,
        "pos_mean_kl": pos_mean_kl.tolist(),
        "pos_bin_counts": pos_bin_counts.tolist(),
    }

    # --- figures ---
    if plt is not None:
        # 1. Instant KL distribution over training
        rids_arr = np.array([r["rollout_id"] for r in per_rollout])
        means = np.array([r["mean"] for r in per_rollout])
        p90s = np.array([r["p90"] for r in per_rollout])
        p99s = np.array([r["p99"] for r in per_rollout])
        maxs = np.array([r["max"] for r in per_rollout])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rids_arr, means, "-o", label="mean", lw=2)
        ax.plot(rids_arr, p90s, "-s", label="p90", lw=1.5, alpha=0.8)
        ax.plot(rids_arr, p99s, "-^", label="p99", lw=1.5, alpha=0.8)
        ax.plot(rids_arr, maxs, "--x", label="max", lw=1, alpha=0.6)
        ax.set_xlabel("rollout id")
        ax.set_ylabel("instant reverse KL  (log_S - log_T)")
        ax.set_yscale("log")
        ax.set_title("Instant per-token reverse KL — training trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "instant_kl_over_training.png", dpi=120)
        plt.close(fig)

        # 2. Distribution histogram (early vs late)
        first_rid = rids[0]
        last_rid = rids[-1]
        first_kl = np.concatenate(
            [
                np.asarray(s["reverse_kl"], dtype=np.float32)
                for s in load_samples_for_rollout(dump_dir, first_rid)
                if len(s["reverse_kl"]) > 0
            ]
        )
        last_kl = np.concatenate(
            [
                np.asarray(s["reverse_kl"], dtype=np.float32)
                for s in load_samples_for_rollout(dump_dir, last_rid)
                if len(s["reverse_kl"]) > 0
            ]
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(-0.5, 4.0, 80)
        ax.hist(first_kl, bins=bins, alpha=0.5, label=f"r{first_rid} (early)", density=True)
        ax.hist(last_kl, bins=bins, alpha=0.5, label=f"r{last_rid} (late)", density=True)
        ax.set_xlabel("instant reverse KL")
        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.set_title("Instant KL distribution — early vs late training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "instant_kl_distribution.png", dpi=120)
        plt.close(fig)

        # 3. Future-KL Pearson correlation by K
        labels = [K_labels[k] for k in Ks]
        means_r = [np.mean(pearson_per_K[k]) if pearson_per_K[k] else 0.0 for k in Ks]
        p25_r = [np.percentile(pearson_per_K[k], 25) if pearson_per_K[k] else 0.0 for k in Ks]
        p75_r = [np.percentile(pearson_per_K[k], 75) if pearson_per_K[k] else 0.0 for k in Ks]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(Ks))
        ax.bar(x, means_r, yerr=[np.array(means_r) - p25_r, np.array(p75_r) - means_r], capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("future KL window K")
        ax.set_ylabel("Pearson r (instant_kl[t], future_kl_K[t])")
        ax.set_title("How redundant is future KL vs instant KL?")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0.95, color="red", lw=1, ls="--", label="r=0.95 (essentially redundant)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "future_kl_correlation.png", dpi=120)
        plt.close(fig)

        # 4. Variance ratio (SNR proxy)
        var_ratio_mean = [np.mean(snr_per_K[k]) if snr_per_K[k] else 0.0 for k in Ks]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, var_ratio_mean)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("future KL window K")
        ax.set_ylabel("var(future_kl_K) / var(instant_kl)")
        ax.set_title("Smoothing effect of K-step averaging  (low = noisy → smooth)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(fig_dir / "snr_by_K.png", dpi=120)
        plt.close(fig)

        # 5. Mean KL by relative position
        bin_centers = (np.arange(position_bins) + 0.5) / position_bins
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(bin_centers, pos_mean_kl, "-o", lw=2)
        ax.set_xlabel("relative position in response (0=start, 1=end)")
        ax.set_ylabel("mean instant reverse KL")
        ax.set_title(f"Mean KL by position  (rollout r{midpoint_rid})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "kl_by_position.png", dpi=120)
        plt.close(fig)

    # --- write summary ---
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump-dir", default="kl_analysis/dumps")
    p.add_argument("--out-dir", default="kl_analysis")
    p.add_argument("--position-bins", type=int, default=20)
    args = p.parse_args()

    summary = analyze(Path(args.dump_dir), Path(args.out_dir), position_bins=args.position_bins)

    print("\n=== Per-rollout instant KL trajectory ===")
    for r in summary["per_rollout_instant_kl"]:
        print(
            f"r{r['rollout_id']:>3d}: mean={r['mean']:.4f} p50={r['p50']:.4f} "
            f"p90={r['p90']:.4f} p99={r['p99']:.4f} max={r['max']:.3f} n={r['n']}"
        )

    print("\n=== Pearson r(instant_kl, future_kl_K) — averaged over samples ===")
    print("  K     |   mean r   |   median r")
    for k_label, mr in summary["future_kl_pearson_r_mean"].items():
        med = summary["future_kl_pearson_r_p50"][k_label]
        print(f"  {k_label:>5s} |  {mr:>7.3f}  |  {med:>7.3f}")

    print("\n=== Var ratio var(future_K)/var(instant) — smoothing factor ===")
    print("  K     |  ratio")
    for k_label, vr in summary["future_kl_var_ratio_mean"].items():
        print(f"  {k_label:>5s} |  {vr:>.4f}")

    print(
        f"\n=== Mean KL by relative position (r{summary['midpoint_rollout_for_distribution']}) ==="
    )
    for i, v in enumerate(summary["pos_mean_kl"]):
        c = (i + 0.5) / len(summary["pos_mean_kl"])
        bar = "#" * int(v * 100)
        print(f"  pos {c:.2f}: {v:.4f} {bar}")

    print(f"\n[done] figures in {args.out_dir}/figures/   summary in {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
