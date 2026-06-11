"""Per-phase reverse_kl temporal analysis.

Splits training into 3 phases (early/mid/late) and re-computes:
  - sparsity (frac exact 0)
  - active fraction (|x| > 0.05)
  - K-window catch rate
  - |reverse_kl| autocorr ρ(d)

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
    ("R4 (mean K=4)",            "opd-4b-R4-meanK4",         "#2ca02c"),
    ("R3b (mean K=8 + mask)",    "opd-4b-R3b-meanK8-mask",   "#ff7f0e"),
    ("R5 (mean K=8 + soft)",     "opd-4b-R5-meanK8-softmask","#9467bd"),
]
PHASES = [("early", 0, 100), ("mid", 100, 200), ("late", 200, 1000)]
THR = 0.05
K_LIST = [1, 2, 4, 8, 16, 32, 64]
D_MAX = 16


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


def load_arrays(paths, max_samples=64):
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                rk = np.asarray(d.get("reverse_kl", []), dtype=np.float64)
                if rk.size > 32:
                    out.append(rk)
                if len(out) >= max_samples:
                    return out
    return out


def autocorr_abs(samples, d_max):
    rho = np.zeros(d_max + 1)
    counts = np.zeros(d_max + 1, dtype=np.int64)
    for x in samples:
        if len(x) < d_max + 4:
            continue
        x = np.abs(x)
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


def k_window_active(samples, K_list, threshold):
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
    summary = defaultdict(dict)

    for label, name, _color in RUNS:
        run_dir = os.path.join(BASE, name)
        rollouts = list_rollouts(run_dir)
        if not rollouts:
            continue

        for phase, lo, hi in PHASES:
            phase_samples = []
            for rid, paths in rollouts:
                if lo <= rid < hi:
                    phase_samples.extend(load_arrays(paths, max_samples=8))

            if not phase_samples:
                summary[label][phase] = None
                continue

            all_vals = np.concatenate([np.abs(x) for x in phase_samples])
            sparsity = (all_vals == 0).mean()
            active_frac = (all_vals > THR).mean()
            mean_abs = all_vals.mean()
            kw = k_window_active(phase_samples, K_LIST, THR)
            rho = autocorr_abs(phase_samples, D_MAX)

            summary[label][phase] = {
                "n_samples": len(phase_samples),
                "sparsity": float(sparsity),
                "active_frac": float(active_frac),
                "mean_abs": float(mean_abs),
                "rho_abs_lag1": float(rho[1]),
                "rho_abs_lag4": float(rho[4]),
                "rho_abs_lag8": float(rho[8]),
                "kw": kw,
            }

    print(f"{'run':25s} {'phase':>6s} {'#samp':>6s} {'frac=0':>7s} "
          f"{'active%':>8s} {'mean|x|':>8s} {'ρ|x|(1)':>8s} {'ρ|x|(8)':>8s} "
          f"{'P(K=1)':>8s} {'P(K=4)':>8s} {'P(K=8)':>8s} {'P(K=32)':>8s}")
    for label, _name, _ in RUNS:
        for phase, _lo, _hi in PHASES:
            s = summary[label].get(phase)
            if s is None:
                print(f"{label:25s} {phase:>6s}  (no data)")
                continue
            kw = s["kw"]
            print(f"{label:25s} {phase:>6s} {s['n_samples']:>6d} "
                  f"{s['sparsity']:>7.3f} {s['active_frac']:>8.3f} "
                  f"{s['mean_abs']:>8.4f} "
                  f"{s['rho_abs_lag1']:>8.3f} {s['rho_abs_lag8']:>8.3f} "
                  f"{kw[1]:>8.3f} {kw[4]:>8.3f} {kw[8]:>8.3f} {kw[32]:>8.3f}")
        print()

    with open("/tmp/kstep_per_phase.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved /tmp/kstep_per_phase.json")

    # Plot: K-window catch rate per phase (one row per phase, 4 colors per run)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (phase, _lo, _hi) in zip(axes, PHASES):
        for label, _name, color in RUNS:
            s = summary[label].get(phase)
            if s is None:
                continue
            ks = sorted(s["kw"].keys())
            ax.plot(ks, [s["kw"][k] for k in ks], "o-", color=color, lw=1.6, ms=4, label=label)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{phase} phase  (rollouts {[(p, lo, hi) for p, lo, hi in PHASES if p == phase][0][1]}-{[(p, lo, hi) for p, lo, hi in PHASES if p == phase][0][2]})")
        ax.set_xlabel("K")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(0, 1.02)
        if phase == "early":
            ax.set_ylabel("P(K-window contains ≥1 active token)")
            ax.legend(fontsize=8)
    fig.suptitle(f"K-window catch rate by training phase  (active = |reverse_kl| > {THR})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("/tmp/kstep_window_per_phase.png", dpi=130)
    print("saved /tmp/kstep_window_per_phase.png")

    # Plot: |reverse_kl| autocorr per phase
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (phase, _lo, _hi) in zip(axes2, PHASES):
        for label, _name, color in RUNS:
            s = summary[label].get(phase)
            if s is None:
                continue
            # We didn't save full rho array; just plot lag1/4/8
            # Reload to get full curve
            run_dir = os.path.join(BASE, [n for l, n, _ in RUNS if l == label][0])
            rollouts = list_rollouts(run_dir)
            samples = []
            for rid, paths in rollouts:
                if _lo <= rid < _hi:
                    samples.extend(load_arrays(paths, max_samples=8))
            if not samples:
                continue
            rho = autocorr_abs(samples, D_MAX)
            ax.plot(range(D_MAX + 1), rho, "o-", color=color, lw=1.4, ms=3, label=label)
        ax.set_title(f"{phase} phase")
        ax.set_xlabel("lag d")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axhline(0.1, color="red", lw=0.7, ls="--", alpha=0.5)
        ax.grid(alpha=0.3)
        if phase == "early":
            ax.set_ylabel("autocorr of |reverse_kl|")
            ax.legend(fontsize=8)
    fig2.suptitle("|reverse_kl| autocorrelation by training phase", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig("/tmp/kstep_autocorr_per_phase.png", dpi=130)
    print("saved /tmp/kstep_autocorr_per_phase.png")


if __name__ == "__main__":
    main()
