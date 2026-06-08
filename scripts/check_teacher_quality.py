import json, sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/teacher_extra_shard0.jsonl"

with open(filepath) as f:
    lines = f.readlines()

print(f"Total: {len(lines)}")

has_think_open = 0
has_think_close = 0
has_boxed = 0
lengths = []

n = min(len(lines), 1000)
for line in lines[:n]:
    d = json.loads(line)
    r = d["messages"][1]["content"]
    lengths.append(len(r))
    if "<think>" in r: has_think_open += 1
    if "</think>" in r: has_think_close += 1
    if "\\boxed" in r: has_boxed += 1

print(f"\nFirst {n} samples:")
print(f"  Has <think>: {has_think_open}/{n} ({has_think_open*100//n}%)")
print(f"  Has </think>: {has_think_close}/{n} ({has_think_close*100//n}%)")
print(f"  Has boxed: {has_boxed}/{n} ({has_boxed*100//n}%)")
lengths.sort()
print(f"  Avg len: {sum(lengths)//n} chars")
print(f"  p50 len: {lengths[n//2]}")
print(f"  p90 len: {lengths[int(n*0.9)]}")
print(f"  max len: {lengths[-1]}")

# Show samples
for idx in [0, 500]:
    if idx >= len(lines): break
    d = json.loads(lines[idx])
    prompt = d["messages"][0]["content"]
    r = d["messages"][1]["content"]
    print(f"\n=== Sample {idx} ===")
    print(f"Prompt: {prompt[:200]}")
    print(f"Response ({len(r)} chars):")
    print(f"  Start: {r[:250]}")
    print(f"  End: {r[-150:]}")
    print(f"  has_think: {'</think>' in r}, has_boxed: {'\\\\boxed' in r or '\\boxed' in r}")
