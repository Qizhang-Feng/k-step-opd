"""Diversity / mode-collapse diagnostics for Phase 2.5 evals.

For each (run, iter) cell we compute:
    pass@1 (single-sample, deterministic-ish, t=0.6 here)
    avg_pass@1 = mean over 16 samples per problem (overall capability)
    pass@any  = ≥1 of 16 correct per problem (coverage / diversity)
    gap = pass@any − avg_pass@1                   (= room left to grow if diversity stays)
    coverage_efficiency = avg_pass@1 / pass@any   (fraction of "covered" problems consistently solved)

Mode-collapse signature:
    avg_pass@1 ↑  AND  pass@any ↓ (or static)
    coverage_efficiency ↑ but absolute pass@any not gaining
"""
import json
import os

EVAL_DIR = "kl_analysis/phase25/eval"

CASES = [
    # (label, prefix, [iters or None])
    ("baseline (SFT v2-ckpt700)",       "baseline",        [None]),
    ("opd-4b-A iter299",                "opd-iter299",     [None]),
    ("opd-4b-B iter99",                 "B",               [99]),
    ("opd-4b-B iter299",                "opd-lightning",   [299]),
    ("R1 (mean K=8 no mask)",           "R1-meanK8",       [99, 199, 299]),
    ("R3 (sum K=8 + mask)",             "R3-sumK8-mask",   [99, 199, 299]),
    ("R3b (mean K=8 + mask)",           "R3b-meanK8-mask", [99, 199, 299]),
    ("R4 (mean K=4 no mask)",           "R4-meanK4",       [99, 199, 299]),
]


def load(prefix, bench, it):
    cands = []
    if it is not None:
        cands.append(os.path.join(EVAL_DIR, f"aime{bench}_{prefix}-iter{it}.json"))
    cands.append(os.path.join(EVAL_DIR, f"aime{bench}_{prefix}.json"))
    for c in cands:
        if os.path.exists(c):
            with open(c) as f:
                return json.load(f)
    return None


def fmt(x):
    return f"{x*100:>6.2f}%" if x is not None else "   —  "


print()
print("=" * 130)
print(
    f"{'run':28s} {'iter':>5s} | "
    f"{'avg@1-24':>9s} {'pass@1-24':>10s} {'pass@any-24':>12s} {'gap-24':>8s} "
    f"{'avg@1-25':>9s} {'pass@any-25':>12s} {'avg_len':>8s}"
)
print("-" * 130)

for label, prefix, iters in CASES:
    for it in iters:
        d24 = load(prefix, 2024, it)
        d25 = load(prefix, 2025, it)
        if d24 is None and d25 is None:
            continue
        it_str = "—" if it is None else str(it)
        avg24 = d24.get("avg_pass_at_1") if d24 else None
        # pass_at_1 = single-sample (the s=0 sample) — deterministic-ish
        p1_24 = d24.get("pass_at_1") if d24 else None
        pany24 = d24.get("pass_at_any") if d24 else None
        gap24 = (pany24 - avg24) if (avg24 and pany24) else None
        avg25 = d25.get("avg_pass_at_1") if d25 else None
        pany25 = d25.get("pass_at_any") if d25 else None
        avg_len = d24.get("avg_response_length") if d24 else None
        len_str = f"{avg_len:>8.0f}" if avg_len else "       —"
        print(
            f"{label:28s} {it_str:>5s} | "
            f"{fmt(avg24)} "
            f"{fmt(p1_24)}   "
            f"{fmt(pany24)}   "
            f"{fmt(gap24) if gap24 is not None else '   —  ':>8s} "
            f"{fmt(avg25)} "
            f"{fmt(pany25)}   "
            f"{len_str:>8s}"
        )
print("=" * 130)


# Summary: did any run show mode collapse (avg↑ but pany↓ vs baseline)?
def metric(prefix, bench, it, key):
    d = load(prefix, bench, it)
    return d.get(key) if d else None


base24_avg = metric("baseline", 2024, None, "avg_pass_at_1")
base24_pany = metric("baseline", 2024, None, "pass_at_any")
base25_avg = metric("baseline", 2025, None, "avg_pass_at_1")
base25_pany = metric("baseline", 2025, None, "pass_at_any")

print()
print("=== Mode-collapse signature: Δ avg@1 vs Δ pass@any (relative to SFT baseline) ===")
print(f"baseline AIME-24:  avg@1 {base24_avg*100:.2f}%   pass@any {base24_pany*100:.2f}%")
print(f"baseline AIME-25:  avg@1 {base25_avg*100:.2f}%   pass@any {base25_pany*100:.2f}%")
print()
print(f"{'run':28s} {'iter':>5s} | {'Δavg-24':>8s} {'Δpany-24':>9s} {'Δavg-25':>8s} {'Δpany-25':>9s}  {'collapse?':>10s}")
for label, prefix, iters in CASES:
    if prefix == "baseline":
        continue
    for it in iters:
        a24 = metric(prefix, 2024, it, "avg_pass_at_1")
        p24 = metric(prefix, 2024, it, "pass_at_any")
        a25 = metric(prefix, 2025, it, "avg_pass_at_1")
        p25 = metric(prefix, 2025, it, "pass_at_any")
        if a24 is None or p24 is None:
            continue
        da24 = (a24 - base24_avg) * 100
        dp24 = (p24 - base24_pany) * 100
        da25 = (a25 - base25_avg) * 100 if (a25 is not None) else None
        dp25 = (p25 - base25_pany) * 100 if (p25 is not None) else None
        # Mode collapse: avg up but pany down on either bench
        signs = []
        if da24 > 0 and dp24 < -1:
            signs.append("24:↑/↓")
        if da25 is not None and da25 > 0 and dp25 is not None and dp25 < -1:
            signs.append("25:↑/↓")
        flag = " ".join(signs) if signs else ""
        it_str = "—" if it is None else str(it)
        d25a = f"{da25:+8.2f}" if da25 is not None else "       —"
        d25p = f"{dp25:+8.2f}" if dp25 is not None else "       —"
        print(f"{label:28s} {it_str:>5s} | {da24:+8.2f} {dp24:+8.2f}  {d25a:>8s} {d25p:>8s}  {flag:>10s}")
