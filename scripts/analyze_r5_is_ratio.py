"""Detailed IS-ratio + soft_weight analysis on R5 KL dumps.

R5 dumps include `rollout_log_probs`, so we can compute:
    is_ratio[t]   = exp(student_log_probs[t] - rollout_log_probs[t])
    soft_weight[t] = min( c / max(is_ratio[t], 1), 1 ),   c = 10
    hard_keep[t]  = 1 if is_ratio[t] <= c else 0

Per-rollout:
    - IS ratio quantiles (median, p90, p99, max)
    - soft_weight quantiles (mean, min)
    - frac of tokens that would have been hard-masked (IS > c)
    - effective contribution loss for soft mask  (1 - mean(soft_weight))

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

R5_DIR = "/root/.cache/huggingface/opd-4b-R5-meanK8-softmask/kl_dump"
DUALCLIP_C = 10.0


def list_rollouts(run_dir):
    pat = os.path.join(run_dir, "r*_rank*.jsonl")
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


def per_rollout(paths):
    is_ratios = []
    for p in paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                slp = np.asarray(d.get("student_log_probs", []), dtype=np.float64)
                rlp = np.asarray(d.get("rollout_log_probs", []), dtype=np.float64)
                n = min(len(slp), len(rlp))
                if n == 0:
                    continue
                diff = np.clip(slp[:n] - rlp[:n], -50.0, 50.0)
                is_ratios.append(np.exp(diff))
    if not is_ratios:
        return None
    is_all = np.concatenate(is_ratios)
    soft_w = np.clip(DUALCLIP_C / np.clip(is_all, 1.0, None), None, 1.0)
    hard_keep = (is_all <= DUALCLIP_C).astype(np.int8)
    return {
        "n_tokens": int(is_all.size),
        "is_median": float(np.median(is_all)),
        "is_p90": float(np.quantile(is_all, 0.90)),
        "is_p99": float(np.quantile(is_all, 0.99)),
        "is_p999": float(np.quantile(is_all, 0.999)),
        "is_max": float(np.max(is_all)),
        "is_mean": float(np.mean(is_all)),
        "soft_w_mean": float(np.mean(soft_w)),
        "soft_w_min": float(np.min(soft_w)),
        "soft_w_p01": float(np.quantile(soft_w, 0.01)),
        "frac_is_gt_2": float((is_all > 2.0).mean()),
        "frac_is_gt_5": float((is_all > 5.0).mean()),
        "frac_is_gt_c": float((is_all > DUALCLIP_C).mean()),
        "frac_soft_active": float((soft_w < 1.0).mean()),
        "mean_signal_loss": float(1.0 - np.mean(soft_w)),
    }


def main():
    rollouts = list_rollouts(R5_DIR)
    print(f"R5 rollouts with dump: {len(rollouts)}")

    summary = []
    for rid, paths in rollouts:
        s = per_rollout(paths)
        if s is None:
            continue
        s["rid"] = rid
        summary.append(s)

    # Print table
    print()
    print(f"{'rid':>4s} {'n_tok':>7s} {'is_med':>8s} {'is_p90':>8s} {'is_p99':>8s} "
          f"{'is_p999':>8s} {'is_max':>8s} {'sw_mean':>8s} {'sw_min':>8s} "
          f"{'%IS>2':>7s} {'%IS>c':>7s} {'%sw<1':>7s}")
    for s in summary[::10]:  # every 10th rollout
        print(f"{s['rid']:>4d} {s['n_tokens']:>7d} "
              f"{s['is_median']:>8.4f} {s['is_p90']:>8.4f} {s['is_p99']:>8.4f} "
              f"{s['is_p999']:>8.4f} {s['is_max']:>8.2f} "
              f"{s['soft_w_mean']:>8.4f} {s['soft_w_min']:>8.4f} "
              f"{s['frac_is_gt_2']*100:>6.3f}% {s['frac_is_gt_c']*100:>6.3f}% "
              f"{s['frac_soft_active']*100:>6.3f}%")

    # Aggregated across all rollouts
    print()
    print("=== Aggregated across all R5 rollouts ===")
    agg = lambda key: float(np.mean([s[key] for s in summary]))
    print(f"  P(IS > 1.5)  = {float(np.mean([s['frac_is_gt_2'] for s in summary])):.4%}  ← rough")
    print(f"  P(IS > 2)    = {agg('frac_is_gt_2'):.4%}")
    print(f"  P(IS > 5)    = {agg('frac_is_gt_5'):.4%}")
    print(f"  P(IS > c=10) = {agg('frac_is_gt_c'):.4%}  ← would-be hard-mask fraction")
    print(f"  P(soft_w<1)  = {agg('frac_soft_active'):.4%}  ← soft mask fired (= P(IS > 1) effectively)")
    print(f"  Median IS    = {agg('is_median'):.4f}")
    print(f"  Mean IS      = {agg('is_mean'):.4f}")
    print(f"  Mean soft_w  = {agg('soft_w_mean'):.6f}")
    print(f"  Mean signal_loss = {agg('mean_signal_loss'):.6f}  ← 1 - mean(soft_w), magnitude of soft-mask attenuation")
    print(f"  Worst soft_w_min over all rollouts = {min(s['soft_w_min'] for s in summary):.4f}")
    print(f"  Max IS over all rollouts = {max(s['is_max'] for s in summary):.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    rids = [s["rid"] for s in summary]
    axes[0][0].plot(rids, [s["is_median"] for s in summary], "o-", color="#2ca02c", lw=1.4, ms=3, label="median")
    axes[0][0].plot(rids, [s["is_p90"] for s in summary], "o-", color="#1f77b4", lw=1.4, ms=3, label="p90")
    axes[0][0].plot(rids, [s["is_p99"] for s in summary], "o-", color="#d62728", lw=1.4, ms=3, label="p99")
    axes[0][0].plot(rids, [s["is_p999"] for s in summary], "o-", color="#9467bd", lw=1.4, ms=3, label="p99.9")
    axes[0][0].plot(rids, [s["is_max"] for s in summary], "-", color="#888888", lw=1, alpha=0.6, label="max")
    axes[0][0].axhline(DUALCLIP_C, color="black", lw=0.8, ls="--", label=f"c={DUALCLIP_C}")
    axes[0][0].axhline(1.0, color="grey", lw=0.5, ls=":", alpha=0.7)
    axes[0][0].set_yscale("log")
    axes[0][0].set_xlabel("rollout id")
    axes[0][0].set_ylabel("IS ratio (log)")
    axes[0][0].set_title("R5: IS ratio quantiles per rollout")
    axes[0][0].legend(fontsize=8)
    axes[0][0].grid(alpha=0.3, which="both")

    axes[0][1].plot(rids, [s["frac_soft_active"] * 100 for s in summary], "o-", color="#9467bd", lw=1.5, ms=3, label="P(soft_w<1) = P(IS>1)")
    axes[0][1].plot(rids, [s["frac_is_gt_2"] * 100 for s in summary], "o-", color="#ff7f0e", lw=1.5, ms=3, label="P(IS>2)")
    axes[0][1].plot(rids, [s["frac_is_gt_5"] * 100 for s in summary], "o-", color="#d62728", lw=1.5, ms=3, label="P(IS>5)")
    axes[0][1].plot(rids, [s["frac_is_gt_c"] * 100 for s in summary], "o-", color="black", lw=1.5, ms=3, label=f"P(IS>c={DUALCLIP_C})")
    axes[0][1].set_xlabel("rollout id")
    axes[0][1].set_ylabel("fraction of tokens (%)")
    axes[0][1].set_title("R5: fraction of tokens above various IS thresholds")
    axes[0][1].legend(fontsize=8)
    axes[0][1].grid(alpha=0.3)

    axes[1][0].plot(rids, [s["soft_w_mean"] for s in summary], "o-", color="#17becf", lw=1.5, ms=3, label="mean soft_w")
    axes[1][0].plot(rids, [s["soft_w_min"] for s in summary], "o-", color="#bcbd22", lw=1.5, ms=3, alpha=0.7, label="min soft_w")
    axes[1][0].axhline(1.0, color="grey", lw=0.5, ls="--", alpha=0.7, label="no attenuation")
    axes[1][0].set_xlabel("rollout id")
    axes[1][0].set_ylabel("soft_weight = min(c/IS, 1)")
    axes[1][0].set_title("R5: soft mask weight per rollout")
    axes[1][0].set_ylim(0, 1.05)
    axes[1][0].legend(fontsize=8)
    axes[1][0].grid(alpha=0.3)

    axes[1][1].plot(rids, [s["mean_signal_loss"] * 100 for s in summary], "o-", color="#e377c2", lw=1.5, ms=3)
    axes[1][1].set_xlabel("rollout id")
    axes[1][1].set_ylabel("1 - mean(soft_w)  (%)")
    axes[1][1].set_title("R5: average signal attenuation from soft mask\n(0% = identical to R1)")
    axes[1][1].set_ylim(bottom=0)
    axes[1][1].grid(alpha=0.3)

    fig.suptitle("R5 (mean K=8 + soft mask c=10): IS ratio diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = "/tmp/r5_is_ratio_diagnostic.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")

    with open("/tmp/r5_is_ratio_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved /tmp/r5_is_ratio_summary.json")


if __name__ == "__main__":
    main()
