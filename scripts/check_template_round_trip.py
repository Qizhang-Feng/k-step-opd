#!/usr/bin/env python3
"""Check what training actually sees: raw response in jsonl vs after applying chat template."""
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/root/.cache/huggingface/Qwen3-4B-Base", trust_remote_code=True)

FILES = [
    ("79K v2",   "/workspace/data/teacher_sft_filtered.jsonl"),
    ("73K new",  "/workspace/data/teacher_extra_100k_filtered.jsonl"),
]

for name, path in FILES:
    print(f"\n{'='*70}")
    print(f"=== {name} ===")
    print(f"{'='*70}")
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 1:
                break
            d = json.loads(line)
            msgs = d["messages"]
            raw_resp = msgs[1]["content"]

            # What does chat template produce?
            applied = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

            # Get the assistant portion of the templated string
            assistant_marker = '<|im_start|>assistant\n'
            if assistant_marker in applied:
                applied_assistant = applied.split(assistant_marker, 1)[1]
            else:
                applied_assistant = ""

            print(f"\n--- RAW response in jsonl (head 300) ---")
            print(repr(raw_resp[:300]))
            print(f"\n--- TEMPLATED assistant portion (head 300) ---")
            print(repr(applied_assistant[:300]))

            # Check if there's a transformation
            print(f"\n--- DIFFERENCES ---")
            if raw_resp.startswith("<think>"):
                # When template sees <think>...</think>, it splits + reformats
                print(f"  raw starts with: <think>")
                if '</think>' in raw_resp:
                    pre, post = raw_resp.split('</think>', 1)
                    print(f"  raw pre-</think> head: {pre[:100]!r}")
                    print(f"  raw post-</think> head: {post[:100]!r}")
            print(f"  raw length: {len(raw_resp)}")
            print(f"  templated assistant length: {len(applied_assistant)}")

            # Check if templated has the full \n\n separator after </think>
            if '</think>\n\n' in applied_assistant:
                print(f"  ✓ templated has '</think>\\n\\n' (proper separator)")
            else:
                print(f"  ✗ templated MISSING proper separator")

            # Find first diff position
            for j in range(min(len(raw_resp), len(applied_assistant))):
                if raw_resp[j] != applied_assistant[j]:
                    print(f"  First char diff at pos {j}: raw={raw_resp[j-5:j+5]!r} vs templated={applied_assistant[j-5:j+5]!r}")
                    break
