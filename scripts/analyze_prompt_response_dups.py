#!/usr/bin/env python3
"""Drill into prompt-response pattern.
For 152K, find a prompt that has both v2-style and new-style responses, and compare them.
"""
import json
import hashlib
from collections import defaultdict

# Build prompt -> list of responses for each dataset
def load_prompt_responses(path):
    pr = defaultdict(list)
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                p = d["messages"][0]["content"]
                r = d["messages"][1]["content"]
                pr[p].append(r)
            except Exception:
                continue
    return pr

print("Loading 152K mixed...")
pr_mixed = load_prompt_responses("/workspace/data/teacher_sft_179k_thinkfilter.jsonl")

# Distribution of n_responses per prompt
from collections import Counter
n_dist = Counter(len(rs) for rs in pr_mixed.values())
print(f"\nPrompts in 152K with N responses (top 10):")
for k in sorted(n_dist.keys())[:15]:
    print(f"  {k} responses: {n_dist[k]} prompts")
print(f"  ...")
print(f"  max: {max(n_dist.keys())} responses")

# Find prompts with most responses
print("\n\nTop 5 prompts by response count:")
sorted_prompts = sorted(pr_mixed.items(), key=lambda x: len(x[1]), reverse=True)[:5]
for p, rs in sorted_prompts:
    print(f"\n--- prompt ({len(rs)} responses) ---")
    print(f"  prompt[:200]: {p[:200]!r}")
    print(f"  response lengths: {sorted([len(r) for r in rs])}")
    # word ratio per response
    word_ratios = []
    for r in rs:
        ws = r.split()
        word_ratios.append(len(set(ws)) / max(len(ws), 1))
    print(f"  unique-word ratios: {[f'{x:.3f}' for x in sorted(word_ratios)]}")
