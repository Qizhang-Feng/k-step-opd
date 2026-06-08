#!/usr/bin/env python3
"""Check duplicate prompts in raw OpenThoughts3 source files."""
import os
import json
from collections import Counter
import pyarrow.parquet as pq

DATA_DIR = "/workspace/data/OpenThoughts3-1.2M/data"
EXISTING_FILE = "/workspace/data/sft_math_100k_v2.jsonl"

# Count prompt occurrences in 100K v2 (the precursor to 79K)
print("=== sft_math_100k_v2.jsonl (precursor to 79K) ===")
prompt_counts = Counter()
with open(EXISTING_FILE) as f:
    for line in f:
        try:
            d = json.loads(line)
            p = d["messages"][0]["content"]
            prompt_counts[p] += 1
        except Exception:
            continue
total = sum(prompt_counts.values())
n_unique = len(prompt_counts)
n_dup_2 = sum(1 for c in prompt_counts.values() if c >= 2)
n_dup_5 = sum(1 for c in prompt_counts.values() if c >= 5)
print(f"  total samples: {total}")
print(f"  unique prompts: {n_unique}")
print(f"  prompts with >=2 responses: {n_dup_2}")
print(f"  prompts with >=5 responses: {n_dup_5}")
print(f"  max responses per prompt: {max(prompt_counts.values())}")
print(f"  histogram:")
hist = Counter(prompt_counts.values())
for k in sorted(hist.keys()):
    print(f"    {k} responses: {hist[k]} prompts")

# Spot-check raw OT3 source: do the parquet files have duplicate prompts in 'conversations'?
print("\n\n=== Raw OpenThoughts3-1.2M source (1 parquet file) ===")
parquet_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])
print(f"Total parquet files: {len(parquet_files)}")
print(f"Checking first file: {parquet_files[0]}")

table = pq.read_table(parquet_files[0])
rows = table.to_pylist()
print(f"  Rows in file: {len(rows)}")

ot3_prompt_counts = Counter()
for row in rows:
    convs = row.get("conversations", [])
    if convs and len(convs) >= 2:
        p = convs[0].get("value", "")
        if p:
            ot3_prompt_counts[p] += 1

n_unique_ot3 = len(ot3_prompt_counts)
n_dup_ot3_2 = sum(1 for c in ot3_prompt_counts.values() if c >= 2)
print(f"  unique prompts: {n_unique_ot3}")
print(f"  prompts with >=2 responses: {n_dup_ot3_2}")
print(f"  max responses per prompt: {max(ot3_prompt_counts.values())}")
print(f"  histogram (top 5):")
hist_ot3 = Counter(ot3_prompt_counts.values())
for k in sorted(hist_ot3.keys())[:8]:
    print(f"    {k} responses: {hist_ot3[k]} prompts")
