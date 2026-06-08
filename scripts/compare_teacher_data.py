#!/usr/bin/env python3
"""Compare sample style between 79K (v2) and 73K (new) teacher data."""
import json

FILES = [
    ("79K v2  (temp=0.6, top_p=0.95)", "/opt/dlami/nvme/qzf/data/teacher_sft_filtered.jsonl"),
    ("73K new (temp=0.7, top_p=0.9)",  "/opt/dlami/nvme/qzf/data/teacher_extra_100k_filtered.jsonl"),
]

print("\n=== Aggregate stats (all samples) ===")
print(f"{'name':<40s}  {'n':>6s}  {'mean_ratio':>10s}  {'p10':>6s}  {'p50':>6s}  {'p90':>6s}  {'mean_len':>8s}  {'<0.15':>6s}  {'<0.10':>6s}")
for name, path in FILES:
    ratios = []
    lens = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                resp = d["messages"][1]["content"]
            except Exception:
                continue
            words = resp.split()
            n_words = len(words)
            if n_words < 100:
                continue
            n_unique = len(set(words))
            ratios.append(n_unique / n_words)
            lens.append(len(resp))
    ratios.sort()
    n = len(ratios)
    mean_r = sum(ratios) / max(n, 1)
    p10 = ratios[int(n*0.1)] if n else 0
    p50 = ratios[n // 2] if n else 0
    p90 = ratios[int(n*0.9)] if n else 0
    mean_len = sum(lens) / max(n, 1)
    n_low_015 = sum(1 for r in ratios if r < 0.15)
    n_low_010 = sum(1 for r in ratios if r < 0.10)
    print(f"{name:<40s}  {n:>6d}  {mean_r:>10.3f}  {p10:>6.3f}  {p50:>6.3f}  {p90:>6.3f}  {mean_len:>8.0f}  {n_low_015:>6d}  {n_low_010:>6d}")
