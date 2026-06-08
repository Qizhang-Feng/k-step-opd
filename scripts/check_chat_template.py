#!/usr/bin/env python3
"""Check chat template behavior:
1. What does Qwen3-8B's official chat template produce in thinking mode?
2. How do our generated responses compare?
"""
import json

# Compare our manual prompt format vs Qwen3 official
TEST_PROMPT = "What is 2+3?"

# Our manual format (used in both 79K and 73K generation)
manual = (
    f"<|im_start|>user\n"
    f"Question: {TEST_PROMPT}\n"
    f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
print("=== Our manual prompt ===")
print(repr(manual))
print()

# Try official Qwen3 chat template
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    msgs = [{"role": "user", "content": TEST_PROMPT}]

    print("=== Qwen3 chat template (default, enable_thinking=True) ===")
    official = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(repr(official))
    print()

    print("=== Qwen3 chat template (enable_thinking=False) ===")
    no_think = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(repr(no_think))
    print()

    # Compare token-level
    manual_ids = tok.encode(manual, add_special_tokens=False)
    official_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    print(f"Manual prompt length: {len(manual_ids)} tokens")
    print(f"Official prompt length: {len(official_ids)} tokens")
    print()
    print(f"Manual last 5 token ids: {manual_ids[-5:]}")
    print(f"Manual last 5 tokens: {[tok.decode([t]) for t in manual_ids[-5:]]}")
    print(f"Official last 5 token ids: {official_ids[-5:]}")
    print(f"Official last 5 tokens: {[tok.decode([t]) for t in official_ids[-5:]]}")
except Exception as e:
    print(f"Tokenizer not available: {e}")

# Check: do generated responses start with <think>?
print("\n\n=== Response start patterns in generated data ===")
FILES = [
    ("79K v2",   "/workspace/data/teacher_sft_filtered.jsonl"),
    ("73K new",  "/workspace/data/teacher_extra_100k_filtered.jsonl"),
]
for name, path in FILES:
    print(f"\n--- {name} ---")
    counts = {"<think>": 0, "<\\think>": 0, "</think>": 0, "Okay": 0, "First": 0, "Let me": 0, "Other": 0}
    n_total = 0
    bad_starts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            d = json.loads(line)
            resp = d["messages"][1]["content"]
            n_total += 1
            head = resp[:50]
            matched = False
            for k in counts:
                if head.startswith(k):
                    counts[k] += 1
                    matched = True
                    break
            if not matched:
                counts["Other"] += 1
                if len(bad_starts) < 5:
                    bad_starts.append(head[:60])
    print(f"  Sample 1000 first-line patterns:")
    for k, v in counts.items():
        if v > 0:
            print(f"    {k!r:>20s}: {v}")
    if bad_starts:
        print(f"  Sample 'Other' starts:")
        for s in bad_starts:
            print(f"    {s!r}")
