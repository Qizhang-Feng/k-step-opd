"""Phase 2.5 KL dump diagnostics.

Run inside the k-step-opd container on p5-3:
    docker exec -it k-step-opd python3 /tmp/analyze_phase25_dumps.py

Outputs:
 - /tmp/phase25_dump_summary.json  : numeric summary per run/rollout
 - /tmp/phase25_dump_diagnostic.png: 4-panel diagnostic plot

Targets:
 - opd-4b-R3-sumK8-mask        (only r-1, 8 samples; no rollout_log_probs)
 - opd-4b-R3b-meanK8-mask      (full trajectory, no rollout_log_probs)
 - opd-4b-R5-meanK8-softmask   (full trajectory, WITH rollout_log_probs)

Diagnostics:
 1. reverse_kl distribution per rollout (R3 vs R3b vs R5) — answers
    "did R3 (sum+hardmask) collapse signal to ~0?"
 2. IS ratio distribution per rollout (R5 only) — answers
    "is c=10 ever crossed? does soft mask fire?"
 3. soft_weight = min(c/IS, 1) per rollout (R5 only)
 4. frac of tokens downweighted (IS > c) per rollout (R5 only)
"""
import json
import os
import sys
from glob import glob
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = "/root/.cache/huggingface"
RUNS = [
    ("R3",  "opd-4b-R3-sumK8-mask",       False, "#1f77b4"),
    ("R3b", "opd-4b-R3b-meanK8-mask",     False, "#ff7f0e"),
    ("R5",  "opd-4b-R5-meanK8-softmask",  True,  "#2ca02c"),
]
DUALCLIP_C = 10.0


def list_rollout_files(run_dir):
    """Group files by rollout_id, return sorted list of (rid, [paths])."""
    pat = os.path.join(run_dir, "kl_dump", "r*_rank*.jsonl")
    files = glob(pat)
    by_rid = {}
    for f in files:
        name = os.path.basename(f)  # r{rid}_rank{rank}.jsonl
        try:
            rid = int(name.split("_")[0][1:])
        except ValueError:
            continue
        by_rid.setdefault(rid, []).append(f)
    return sorted(by_rid.items())


def per_rollout_stats(paths, has_rlp):
    """Aggregate per-token tensors across all sample rows in this rollout."""
    rkl = []
    is_ratio = []
    for path in paths:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                rl = np.asarray(d.get("reverse_kl", []), dtype=np.float64)
                rkl.append(rl)
                if has_rlp:
                    slp = np.asarray(d.get("student_log_probs", []), dtype=np.float64)
                    rlp = np.asarray(d.get("rollout_log_probs", []), dtype=np.float64)
                    n = min(len(slp), len(rlp))
                    if n:
                        # IS ratio = exp(student - rollout); clamp exponent for numerical safety
                        diff = np.clip(slp[:n] - rlp[:n], -50.0, 50.0)
                        is_ratio.append(np.exp(diff))
    rkl = np.concatenate(rkl) if rkl else np.array([])
    is_ratio = np.concatenate(is_ratio) if is_ratio else np.array([])
    out = {
        "rkl_mean":   float(np.mean(rkl))   if len(rkl) else float("nan"),
        "rkl_median": float(np.median(rkl)) if len(rkl) else float("nan"),
        "rkl_std":    float(np.std(rkl))    if len(rkl) else float("nan"),
        "rkl_max":    float(np.max(np.abs(rkl))) if len(rkl) else float("nan"),
        "rkl_p99":    float(np.quantile(np.abs(rkl), 0.99)) if len(rkl) else float("nan"),
        "n_tokens":   int(len(rkl)),
    }
    if len(is_ratio):
        soft_w = np.clip(DUALCLIP_C / np.clip(is_ratio, a_min=1.0, a_max=None), a_min=None, a_max=1.0)
        frac_down = float(np.mean(is_ratio > DUALCLIP_C))
        out.update({
            "is_mean":   float(np.mean(is_ratio)),
            "is_median": float(np.median(is_ratio)),
            "is_p90":    float(np.quantile(is_ratio, 0.90)),
            "is_p99":    float(np.quantile(is_ratio, 0.99)),
            "is_max":    float(np.max(is_ratio)),
            "soft_w_mean": float(np.mean(soft_w)),
            "soft_w_min":  float(np.min(soft_w)),
            "frac_downweighted": frac_down,
        })
    return out


def main():
    summary = {}
    for label, name, has_rlp, _ in RUNS:
        run_dir = os.path.join(BASE, name)
        rollouts = list_rollout_files(run_dir)
        print(f"[{label}] {name}: {len(rollouts)} rollouts with dumps")
        per = []
        for rid, paths in rollouts:
            stats = per_rollout_stats(paths, has_rlp)
            stats["rid"] = rid
            per.append(stats)
        summary[label] = per

    out_json = "/tmp/phase25_dump_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary → {out_json}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: reverse_kl mean per rollout
    ax = axes[0][0]
    for label, name, _has_rlp, color in RUNS:
        per = summary[label]
        if not per:
            continue
        rids = [s["rid"] for s in per]
        means = [s["rkl_mean"] for s in per]
        ax.plot(rids, means, "o-", color=color, lw=1.6, ms=3, alpha=0.85, label=f"{label} mean")
    ax.set_title("Per-rollout reverse_kl mean (from KL dump samples)\nR3 ≈ 0 → signal collapsed; R3b/R5 ~0.10 → preserved")
    ax.set_xlabel("rollout id")
    ax.set_ylabel("reverse_kl mean")
    ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: IS ratio quantiles for R5 only
    ax = axes[0][1]
    r5 = summary["R5"]
    if r5 and "is_p99" in r5[0]:
        rids = [s["rid"] for s in r5]
        ax.semilogy(rids, [s["is_median"] for s in r5], "-", color="#2ca02c", lw=1.5, label="median")
        ax.semilogy(rids, [s["is_p90"]    for s in r5], "-", color="#1f77b4", lw=1.5, label="p90")
        ax.semilogy(rids, [s["is_p99"]    for s in r5], "-", color="#d62728", lw=1.5, label="p99")
        ax.semilogy(rids, [s["is_max"]    for s in r5], "-", color="#888888", lw=1, alpha=0.6, label="max")
        ax.axhline(DUALCLIP_C, color="black", lw=1, ls="--", label=f"c={DUALCLIP_C}")
    ax.set_title("R5: IS ratio = exp(student_lp − rollout_lp) quantiles per rollout\nbar=c=10 (mask threshold)")
    ax.set_xlabel("rollout id")
    ax.set_ylabel("IS ratio (log scale)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Panel 3: frac of tokens downweighted (IS > c) per rollout, R5
    ax = axes[1][0]
    if r5 and "frac_downweighted" in r5[0]:
        rids = [s["rid"] for s in r5]
        ax.plot(rids, [s["frac_downweighted"] for s in r5], "o-", color="#9467bd", lw=1.6, ms=3)
    ax.set_title("R5: fraction of tokens with IS>c (would fire mask)\nif ~0 → soft mask never activates → R5 ≡ R1/R3b")
    ax.set_xlabel("rollout id")
    ax.set_ylabel("frac IS > c")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    # Panel 4: mean soft_weight per rollout, R5
    ax = axes[1][1]
    if r5 and "soft_w_mean" in r5[0]:
        rids = [s["rid"] for s in r5]
        ax.plot(rids, [s["soft_w_mean"] for s in r5], "o-", color="#17becf", lw=1.6, ms=3, label="mean soft_w")
        ax.plot(rids, [s["soft_w_min"]  for s in r5], "o-", color="#bcbd22", lw=1.2, ms=2, alpha=0.7, label="min soft_w")
    ax.set_title("R5: soft mask weight = min(c/IS, 1)\nmean ≈ 1.0 → soft mask is effectively a no-op")
    ax.set_xlabel("rollout id")
    ax.set_ylabel("soft_weight")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", lw=0.5, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Phase 2.5 — KL dump diagnostics: R3 (sum+hardmask) vs R3b (mean+hardmask) vs R5 (mean+softmask)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = "/tmp/phase25_dump_diagnostic.png"
    fig.savefig(out_png, dpi=130)
    print(f"saved figure → {out_png}")

    # ---- Summary table ----
    print()
    print("=" * 95)
    print(f"{'run':5s} {'rollouts':>9s} {'rkl_mean(last)':>16s} {'rkl_p99(last)':>15s} {'is_p99(last)':>14s} {'frac>c(last)':>14s}")
    print("-" * 95)
    for label, _name, _hr, _ in RUNS:
        per = summary[label]
        if not per:
            print(f"{label:5s} {'0':>9s} {'NA':>16s} {'NA':>15s} {'NA':>14s} {'NA':>14s}")
            continue
        last = per[-1]
        rkl_m = last["rkl_mean"]
        rkl_p99 = last.get("rkl_p99", float("nan"))
        is_p99 = last.get("is_p99", float("nan"))
        frac = last.get("frac_downweighted", float("nan"))
        print(
            f"{label:5s} {len(per):>9d} {rkl_m:>16.5f} {rkl_p99:>15.5f} "
            f"{is_p99 if not math.isnan(is_p99) else float('nan'):>14.4f} "
            f"{frac if not math.isnan(frac) else float('nan'):>14.5f}"
        )
    print("=" * 95)
    print()


if __name__ == "__main__":
    main()
