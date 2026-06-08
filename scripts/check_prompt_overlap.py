#!/usr/bin/env python3
"""Check prompt overlap between 79K v2 and 73K new shards."""
import json
import hashlib

FILES = {
    "79K v2 (teacher_sft_filtered.jsonl)":  "/workspace/data/teacher_sft_filtered.jsonl",
    "73K new (teacher_extra_100k_filtered.jsonl)": "/workspace/data/teacher_extra_100k_filtered.jsonl",
    "152K mixed (teacher_sft_179k_thinkfilter.jsonl)": "/workspace/data/teacher_sft_179k_thinkfilter.jsonl",
}

prompt_sets = {}
for name, path in FILES.items():
    s = set()
    n = 0
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                p = d["messages"][0]["content"]
                # hash to save memory
                h = hashlib.md5(p.encode()).hexdigest()
                s.add(h)
                n += 1
            except Exception:
                continue
    print(f"{name}: {n} samples, {len(s)} unique prompts")
    prompt_sets[name] = s

# Pairwise overlap
print()
keys = list(prompt_sets.keys())
v2_set = prompt_sets["79K v2 (teacher_sft_filtered.jsonl)"]
new_set = prompt_sets["73K new (teacher_extra_100k_filtered.jsonl)"]
mix_set = prompt_sets["152K mixed (teacher_sft_179k_thinkfilter.jsonl)"]
print(f"79K v2 ∩ 73K new (prompt overlap): {len(v2_set & new_set)}")
print(f"79K v2 ⊆ 152K mixed: {len(v2_set & mix_set)}/{len(v2_set)}")
print(f"73K new ⊆ 152K mixed: {len(new_set & mix_set)}/{len(new_set)}")
print(f"79K v2 ∪ 73K new: {len(v2_set | new_set)}")
print(f"152K mixed total unique: {len(mix_set)}")
print(f"152K \\ (79K v2 ∪ 73K new): {len(mix_set - v2_set - new_set)}")
