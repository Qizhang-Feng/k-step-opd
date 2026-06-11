"""Compute reverse_kl autocorrelation ρ(d) and SNR(K) per run.

Inputs:
    /root/.cache/huggingface/<run>/kl_dump/r{rid}_rank{r}.jsonl
    each line = one sample with `reverse_kl` array of length response_length

Outputs (written under /tmp):
    /tmp/kstep_autocorr_summary.json
    /tmp/kstep_autocorr.png            — ρ(d) curves
    /tmp/kstep_snr.png                 — SNR(K) curves
    /tmp/kstep_evolution.png           — ρ(d=4) and SNR(K=8) over training

Run inside the k-step-opd container on p5-3:
    docker exec -it k-step-opd python3 /tmp/analyze_kstep_autocorr.py
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
K_LIST = [1, 2, 4, 8, 16, 32]


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


def load_reverse_kl_arrays(paths):
    """Return list of 1D numpy arrays (one per sample)."""
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                rk = np.asarray(d.get("reverse_kl", []), dtype=np.float64)
                if rk.size > D_MAX + 4:
                    out.append(rk)
    return out


def autocorr(samples, d_max):
    """Average autocorrelation up to lag d_max across all samples."""
    rho = np.zeros(d_max + 1)
    counts = np.zeros(d_max + 1, dtype=np.int64)
    for x in samples:
        if len(x) < d_max + 4:
            continue
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


def snr_K(samples, K_list):
    """SNR(K) = mean_t( cumK[t] ) / std_t( cumK[t] ), aggregated across samples."""
    out = {}
    for K in K_list:
        means, stds = [], []
        for x in samples:
            if len(x) < K + 1:
                continue
            # sliding mean of K reverse_kl values
            cs = np.cumsum(x)
            cumK = (cs[K:] - cs[:-K]) / K   # length len(x)-K
            means.append(cumK.mean())
            stds.append(cumK.std())
        if not means:
            out[K] = (np.nan, np.nan)
            continue
        m = np.mean(means)
        s = np.mean(stds)
        out[K] = (m, s)
    return out


def main():
    summary = {}
    fig_ac, ax_ac = plt.subplots(figsize=(10, 6))
    fig_snr, ax_snr = plt.subplots(figsize=(10, 6))
    fig_evo, axes_evo = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for label, name, color in RUNS:
        run_dir = os.path.join(BASE, name)
        rollouts = list_rollouts(run_dir)
        if not rollouts:
            print(f"[{label}] no dumps, skip")
            continue
        print(f"[{label}] {len(rollouts)} rollouts with dumps")

        # Pool all samples for global ρ(d), SNR(K)
        all_samples = []
        for rid, paths in rollouts:
            all_samples.extend(load_reverse_kl_arrays(paths))
        rho = autocorr(all_samples, D_MAX)
        snr_map = snr_K(all_samples, K_LIST)

        # Effective K = first lag where ρ < 0.1
        eff_K = D_MAX
        for d in range(1, D_MAX + 1):
            if rho[d] < 0.1:
                eff_K = d
                break

        summary[label] = {
            "n_rollouts": len(rollouts),
            "n_samples": len(all_samples),
            "rho": rho.tolist(),
            "snr": {k: snr_map[k] for k in K_LIST},
            "effective_K_at_rho_0.1": eff_K,
        }

        # Plot ρ(d)
        ax_ac.plot(range(D_MAX + 1), rho, "o-", color=color, lw=1.6, ms=3, label=f"{label}  (eff_K={eff_K})")

        # Plot SNR(K) — use |mean| / std to avoid sign flips at low K
        snrs = [abs(snr_map[K][0]) / max(snr_map[K][1], 1e-9) for K in K_LIST]
        ax_snr.plot(K_LIST, snrs, "o-", color=color, lw=1.6, ms=4, label=label)

        # Evolution plots: ρ(4) and SNR(8) per rollout window
        rho4_per = []
        snr8_per = []
        for rid, paths in rollouts:
            samples = load_reverse_kl_arrays(paths)
            if not samples:
                continue
            r = autocorr(samples, 4)
            rho4_per.append((rid, r[4]))
            s = snr_K(samples, [8])
            m, sd = s[8]
            snr8_per.append((rid, abs(m) / max(sd, 1e-9)))
        if rho4_per:
            xs, ys = zip(*rho4_per)
            axes_evo[0].plot(xs, ys, "o-", color=color, lw=1.4, ms=3, label=label)
        if snr8_per:
            xs, ys = zip(*snr8_per)
            axes_evo[1].plot(xs, ys, "o-", color=color, lw=1.4, ms=3, label=label)

    # Finalize ρ(d) plot
    ax_ac.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax_ac.axhline(0.1, color="red", lw=1, ls="--", alpha=0.5, label="ρ=0.1 (effective horizon)")
    ax_ac.axhline(0.3, color="orange", lw=1, ls=":", alpha=0.5, label="ρ=0.3")
    ax_ac.set_xlabel("lag d (tokens)")
    ax_ac.set_ylabel("autocorrelation ρ(d) of reverse_kl")
    ax_ac.set_title("Reverse-KL autocorrelation across runs\n(measures how far future reverse_kl predicts current reverse_kl)")
    ax_ac.legend(fontsize=9, loc="upper right")
    ax_ac.grid(alpha=0.3)
    fig_ac.tight_layout()
    fig_ac.savefig("/tmp/kstep_autocorr.png", dpi=130)
    print("saved /tmp/kstep_autocorr.png")

    # Finalize SNR(K) plot
    ax_snr.set_xscale("log", base=2)
    ax_snr.set_xlabel("K (cumulative window size)")
    ax_snr.set_ylabel("|mean| / std  of cumulative KL  (SNR proxy)")
    ax_snr.set_title("Cumulative KL SNR vs K  —  argmax ≈ sweet-spot K")
    ax_snr.legend(fontsize=9)
    ax_snr.grid(alpha=0.3, which="both")
    fig_snr.tight_layout()
    fig_snr.savefig("/tmp/kstep_snr.png", dpi=130)
    print("saved /tmp/kstep_snr.png")

    # Finalize evolution plot
    axes_evo[0].set_title("ρ(d=4) over training")
    axes_evo[0].set_ylabel("autocorr at lag 4")
    axes_evo[0].axhline(0.1, color="red", lw=0.7, ls="--", alpha=0.5)
    axes_evo[0].grid(alpha=0.3)
    axes_evo[0].legend(fontsize=8)
    axes_evo[1].set_title("SNR(K=8) over training")
    axes_evo[1].set_xlabel("rollout id")
    axes_evo[1].set_ylabel("|mean|/std at K=8")
    axes_evo[1].grid(alpha=0.3)
    fig_evo.tight_layout()
    fig_evo.savefig("/tmp/kstep_evolution.png", dpi=130)
    print("saved /tmp/kstep_evolution.png")

    # Save summary
    with open("/tmp/kstep_autocorr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved /tmp/kstep_autocorr_summary.json")

    # Quick numeric report
    print()
    print(f"{'run':30s} {'n_samp':>7s} {'ρ(1)':>8s} {'ρ(2)':>8s} {'ρ(4)':>8s} {'ρ(8)':>8s} {'eff_K(0.1)':>11s}")
    for label, _name, _ in RUNS:
        if label not in summary:
            continue
        s = summary[label]
        rho = s["rho"]
        print(f"{label:30s} {s['n_samples']:>7d} "
              f"{rho[1]:>8.3f} {rho[2]:>8.3f} {rho[4]:>8.3f} {rho[8]:>8.3f} "
              f"{s['effective_K_at_rho_0.1']:>11d}")
    print()
    print(f"{'run':30s} {'SNR(K=1)':>10s} {'SNR(K=4)':>10s} {'SNR(K=8)':>10s} {'SNR(K=16)':>10s} {'SNR(K=32)':>10s}")
    for label, _name, _ in RUNS:
        if label not in summary:
            continue
        snr = summary[label]["snr"]
        vals = []
        for K in [1, 4, 8, 16, 32]:
            m, sd = snr[K]
            vals.append(abs(m) / max(sd, 1e-9) if not np.isnan(m) else float("nan"))
        print(f"{label:30s} " + " ".join(f"{v:>10.3f}" for v in vals))


if __name__ == "__main__":
    main()
