"""Pretty-print the JSON output of analyze_phase25_dumps.py.

Run inside k-step-opd container after analyze_phase25_dumps.py has produced
/tmp/phase25_dump_summary.json.
"""
import json

d = json.load(open("/tmp/phase25_dump_summary.json"))

print("=== R5 IS ratio distribution across dumped rollouts ===")
hdr = ["rid", "is_med", "is_p90", "is_p99", "is_max", "frac>c", "sw_min", "rkl_p99"]
print(" ".join(f"{h:>8s}" for h in hdr))
for s in d["R5"]:
    print(
        f"{s['rid']:>8d} "
        f"{s['is_median']:>8.4f} "
        f"{s['is_p90']:>8.4f} "
        f"{s['is_p99']:>8.4f} "
        f"{s['is_max']:>8.2f} "
        f"{s['frac_downweighted']:>8.5f} "
        f"{s['soft_w_min']:>8.4f} "
        f"{s['rkl_p99']:>8.4f}"
    )

print()
print("=== reverse_kl mean trajectory: R3b vs R5 (only rollouts present in both) ===")
print(f"{'rid':>5s} {'R3b':>10s} {'R5':>10s} {'delta':>10s}")
r3b_by_rid = {s["rid"]: s for s in d["R3b"]}
r5_by_rid = {s["rid"]: s for s in d["R5"]}
common = sorted(set(r3b_by_rid) & set(r5_by_rid))
for rid in common:
    r3b = r3b_by_rid[rid]["rkl_mean"]
    r5 = r5_by_rid[rid]["rkl_mean"]
    print(f"{rid:>5d} {r3b:>10.4f} {r5:>10.4f} {r5 - r3b:>10.4f}")

print()
print("=== R3 single dump (r-1, end-of-training snapshot) ===")
for s in d["R3"]:
    print(s)

print()
print("=== R3b end-of-training stats (last 3 dumps) ===")
for s in d["R3b"][-3:]:
    print(
        f"rid={s['rid']:>4d} rkl_mean={s['rkl_mean']:.5f} "
        f"rkl_p99={s['rkl_p99']:.4f} rkl_max={s['rkl_max']:.4f}"
    )

print()
print("=== R5 end-of-training stats (last 3 dumps) ===")
for s in d["R5"][-3:]:
    print(
        f"rid={s['rid']:>4d} rkl_mean={s['rkl_mean']:.5f} "
        f"rkl_p99={s['rkl_p99']:.4f} rkl_max={s['rkl_max']:.4f}"
    )
