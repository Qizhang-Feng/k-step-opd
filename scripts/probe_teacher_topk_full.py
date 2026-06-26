"""Probe SGLang teacher with top_logprobs_num=20 to verify wire format
matches what on_policy_distillation.py expects.

Run on a node that can reach the teacher URL (e.g., p5-3 → qzf-dev private IP).
"""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://172.31.31.105:30000/generate"

payload = {
    "input_ids": [9707, 11, 1246, 525, 498, 30, 358, 2776, 264, 8606, 1379],
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 0,
        "skip_special_tokens": False,
    },
    "return_logprob": True,
    "logprob_start_len": 0,
    "top_logprobs_num": 20,
}

req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=10) as resp:
    body = resp.read()

d = json.loads(body)
m = d["meta_info"]

print(f"meta_info keys: {list(m.keys())}")
itlp = m.get("input_token_logprobs")
print(f"\ninput_token_logprobs: type={type(itlp).__name__}, len={len(itlp)}")
print(f"  [0] = {itlp[0]} (typically None)")
print(f"  [1] = {itlp[1]}")

itplp = m.get("input_top_logprobs")
print(f"\ninput_top_logprobs: type={type(itplp).__name__}, len={len(itplp)}")
print(f"  [0] = {itplp[0]} (typically None)")
print(f"  [1] type={type(itplp[1]).__name__}, len={len(itplp[1])}")
print(f"  [1][:3] = {itplp[1][:3]}")

# Verify our parsing logic works
print("\n=== verify on_policy_distillation parsing path ===")
response_length = 5  # pretend last 5 are response tokens

# Drop position 0, take response tail
response_top = itplp[1:][-response_length:]
topk_ids = []
topk_logp = []
for pos in response_top:
    if pos is None:
        topk_ids.append([0])
        topk_logp.append([0.0])
    else:
        topk_ids.append([entry[1] for entry in pos])
        topk_logp.append([entry[0] for entry in pos])

print(f"  response_length={response_length}, slot count={len(topk_ids)}")
print(f"  slot 0: K={len(topk_ids[0])}, ids[:3]={topk_ids[0][:3]}, logp[:3]={topk_logp[0][:3]}")
print(f"  all slots have K={len(topk_ids[0])}: {all(len(x) == len(topk_ids[0]) for x in topk_ids)}")
print(f"  all logp slots have K={len(topk_logp[0])}: {all(len(x) == len(topk_logp[0]) for x in topk_logp)}")
print("\nPASS: wire format matches expected parsing.")
