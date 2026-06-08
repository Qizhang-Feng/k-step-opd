#!/usr/bin/env python3
"""Quick text summary of K-step KL temporal evolution."""
import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, "scripts")
from analyze_kstep_kl_over_time import analyze_run

Ks = [2, 4, 8, 16, 32, None]
result_a = analyze_run(Path("kl_analysis/dumps_A"), "opd-4b-A", Ks)
result_b = analyze_run(Path("kl_analysis/dumps_B"), "opd-4b-B", Ks)

for run in [result_a, result_b]:
    print(f"\n=== {run['name']} K=8 trajectory ===")
    s = run["series"][8]
    rids = s["rid"]
    means = s["kl_mean_mean"]
    p99s = s["kl_p99_mean"]
    pears = s["pearson_mean"]
    vrs = s["var_ratio_mean"]

    n = len(rids)
    if n == 0: continue
    # show first, mid, last 3 points
    show = sorted(set([0, 1, n // 2 - 1, n // 2, n // 2 + 1, n - 2, n - 1]))
    show = [i for i in show if 0 <= i < n]
    print(f"  {'rid':>5s}  {'K8_mean':>9s}  {'K8_p99':>8s}  {'Pearson':>8s}  {'var_ratio':>10s}")
    for i in show:
        print(f"  {rids[i]:>5d}  {means[i]:>9.4f}  {p99s[i]:>8.4f}  {pears[i]:>8.3f}  {vrs[i]:>10.4f}")

    # Trend metrics
    print(f"\n  Mean K=8 KL trend over training:")
    print(f"    early avg (first 5): {np.mean(means[:5]):.4f}")
    print(f"    mid avg (middle 5):  {np.mean(means[n//2-2:n//2+3]):.4f}")
    print(f"    late avg (last 5):   {np.mean(means[-5:]):.4f}")
    print(f"    overall change: {means[-1] - means[0]:+.4f} ({(means[-1] - means[0]) / means[0] * 100:+.1f}%)")
