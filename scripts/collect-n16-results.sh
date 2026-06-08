#!/bin/bash
# Pull n=16 eval results from p5-2, p5-3, p5-4 to local kl_analysis/n16_results/
# and print one-line summary per (model, dataset).
set -e

OUT=kl_analysis/n16_results
mkdir -p "$OUT"

for HOST in p5-3 p5-4 p5-2; do
    echo "=== fetching from $HOST ==="
    rsync -avz "$HOST:/opt/dlami/nvme/qzf/k-step-opd/eval_results_n16/" "$OUT/" 2>&1 | tail -3 || true
done

echo
echo "=== n=16 SUMMARY ==="
python3 - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("kl_analysis/n16_results/*.json")):
    if f.endswith(".partial"):
        continue
    with open(f) as fp:
        d = json.load(fp)
    name = os.path.basename(f).replace(".json","")
    ds = d.get("dataset_name","?")
    avg = d.get("avg_pass_at_1", 0) * 100
    fb = d.get("avg_first_boxed_pass_at_1", 0) * 100
    pa = d.get("pass_at_any", 0) * 100
    ln = d.get("avg_response_length", 0)
    n = d.get("n_problems", 0)
    rows.append((name, ds, n, avg, fb, pa, ln))

# Group by model
from collections import defaultdict
by_model = defaultdict(dict)
for name, ds, n, avg, fb, pa, ln in rows:
    # filename like aime2024_v2-ckpt700-baseline.json
    parts = name.split("_", 1)
    if len(parts) == 2:
        model = parts[1]
    else:
        model = name
    by_model[model][ds] = (n, avg, fb, pa, ln)

print(f"{'Model':35s} {'AIME-2024':>20s} {'AIME-2025':>20s} {'avg_len':>10s}")
print("-"*92)
for model in sorted(by_model):
    r = by_model[model]
    a24 = r.get("aime2024", (0,0,0,0,0))
    a25 = r.get("aime2025", (0,0,0,0,0))
    print(f"{model:35s}  avg={a24[1]:5.1f}% any={a24[3]:5.1f}%  avg={a25[1]:5.1f}% any={a25[3]:5.1f}%   {a24[4]:.0f}")
PY
