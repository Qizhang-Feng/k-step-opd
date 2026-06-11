"""Cross-correlate Phase 2.5 training-time instant_kl trajectories with n=16 eval AIME scores.

Reads:
    kl_analysis/phase25/kl_*.csv             (per-rollout instant_kl + accum_kl)
    kl_analysis/phase25/eval/aime20{24,25}_*.json  (n=16 evals at iter 99/199/299)
"""
import csv
import json
import os
from glob import glob

EVAL_DIR = "kl_analysis/phase25/eval"
KL_DIR = "kl_analysis/phase25"

# (label, kl_csv, eval_prefix)
RUNS = [
    ("baseline (SFT, no OPD)",        None,        "baseline"),
    ("opd-4b-A (instant) iter299",    None,        "opd-iter299"),
    ("opd-4b-B (instant) iter99/299", "kl_B.csv",  "opd-lightning"),  # iter99 + iter299
    ("R1 (mean K=8, no mask)",        "kl_R1.csv", "R1-meanK8"),
    ("R3 (sum K=8, mask, kl=0.125)",  "kl_R3.csv", "R3-sumK8-mask"),
    ("R3b (mean K=8, mask, kl=1.0)",  "kl_R3b.csv","R3b-meanK8-mask"),
    ("R4 (mean K=4, no mask)",        "kl_R4.csv", "R4-meanK4"),
]

# Map iter -> rollout idx (slime saves at multiples of save_interval=50, iter_99 ≈ rollout 100)
ITER_TO_ROLLOUT = {99: 100, 199: 200, 299: 300}


def load_kl_csv(path):
    if not path:
        return None
    rows = []
    with open(os.path.join(KL_DIR, path)) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_eval(prefix, bench, it):
    if prefix is None:
        return None
    # Naming conventions:
    #  - baseline: aime20XX_baseline.json (no iter)
    #  - opd-iter299: aime20XX_opd-iter299.json (iter is in prefix)
    #  - opd-lightning: aime20XX_opd-lightning-iter{99,299}.json
    #  - R*: aime20XX_R*-iter{99,199,299}.json
    candidates = []
    if it:
        candidates.append(os.path.join(EVAL_DIR, f"aime{bench}_{prefix}-iter{it}.json"))
        candidates.append(os.path.join(EVAL_DIR, f"aime{bench}_{prefix}-iter{it:03d}.json"))
    candidates.append(os.path.join(EVAL_DIR, f"aime{bench}_{prefix}.json"))
    for c in candidates:
        if os.path.exists(c):
            with open(c) as f:
                return json.load(f)
    return None


def kl_at_rollout(rows, rollout):
    """Mean instant_kl over rollouts [r-5, r+5)."""
    if not rows:
        return None
    lo, hi = max(0, rollout - 5), min(len(rows), rollout + 5)
    sub = [float(rows[i]["instant_kl"]) for i in range(lo, hi) if rows[i]["instant_kl"]]
    return sum(sub) / len(sub) if sub else None


print()
print("=" * 130)
print(
    f"{'run':40s} {'iter':>5s} | "
    f"{'instant_kl@iter':>15s} {'AIME-24':>9s} {'AIME-25':>9s} {'p_any-24':>9s} {'p_any-25':>9s} {'len':>7s} {'trunc':>7s}"
)
print("-" * 130)

# Static baseline (no training)
b24 = load_eval("baseline", 2024, None)
b25 = load_eval("baseline", 2025, None)
if b24:
    print(
        f"{'baseline (SFT v2-ckpt700)':40s} {'—':>5s} | "
        f"{'—':>15s} {b24['avg_pass_at_1'] * 100:>8.2f}% "
        f"{b25['avg_pass_at_1'] * 100:>8.2f}% "
        f"{b24['pass_at_any'] * 100:>8.2f}% "
        f"{b25['pass_at_any'] * 100:>8.2f}% "
        f"{b24['avg_response_length']:>7.0f} "
        f"{b24['truncation_rate']:>6.0%}"
    )

# Iter-by-iter for each run
for label, kl_csv, prefix in RUNS:
    if kl_csv is None and prefix is None:
        continue
    rows = load_kl_csv(kl_csv)
    for it in [99, 199, 299]:
        kl = kl_at_rollout(rows, ITER_TO_ROLLOUT[it]) if rows else None
        a24 = load_eval(prefix, 2024, it)
        a25 = load_eval(prefix, 2025, it)
        if a24 is None and a25 is None and kl is None:
            continue
        kl_str = f"{kl:>14.4f}" if kl is not None else " " * 14 + "—"
        a24_str = f"{a24['avg_pass_at_1'] * 100:>8.2f}%" if a24 else "    —    "
        a25_str = f"{a25['avg_pass_at_1'] * 100:>8.2f}%" if a25 else "    —    "
        pa24 = f"{a24['pass_at_any'] * 100:>8.2f}%" if a24 else "    —    "
        pa25 = f"{a25['pass_at_any'] * 100:>8.2f}%" if a25 else "    —    "
        ln = f"{a24['avg_response_length']:>7.0f}" if a24 else "      —"
        tr = f"{a24['truncation_rate']:>6.0%}" if a24 else "      —"
        print(f"{label:40s} {it:>5d} | {kl_str:>15s} {a24_str:>9s} {a25_str:>9s} {pa24:>9s} {pa25:>9s} {ln:>7s} {tr:>7s}")

print("=" * 130)

# Pearson correlation across all (run, iter) cells where both metrics present
pairs = []
for label, kl_csv, prefix in RUNS:
    if kl_csv is None or prefix is None:
        continue
    rows = load_kl_csv(kl_csv)
    if not rows:
        continue
    for it in [99, 199, 299]:
        kl = kl_at_rollout(rows, ITER_TO_ROLLOUT[it])
        a24 = load_eval(prefix, 2024, it)
        a25 = load_eval(prefix, 2025, it)
        if kl is None or a24 is None or a25 is None:
            continue
        pairs.append((kl, a24["avg_pass_at_1"], a25["avg_pass_at_1"], a24["avg_response_length"], a24["truncation_rate"]))

if pairs:
    import math
    print()
    print(f"Cross-run Pearson r (n={len(pairs)} (iter,run) cells with paired KL+eval):")
    def pearson(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        return num / den if den else float("nan")

    kls = [p[0] for p in pairs]
    a24s = [p[1] for p in pairs]
    a25s = [p[2] for p in pairs]
    lens = [p[3] for p in pairs]
    trs = [p[4] for p in pairs]
    print(f"   instant_kl ↔ AIME-24 avg_pass_at_1 :  r = {pearson(kls, a24s):+.3f}")
    print(f"   instant_kl ↔ AIME-25 avg_pass_at_1 :  r = {pearson(kls, a25s):+.3f}")
    print(f"   instant_kl ↔ avg_response_length    :  r = {pearson(kls, lens):+.3f}")
    print(f"   instant_kl ↔ truncation_rate        :  r = {pearson(kls, trs):+.3f}")
