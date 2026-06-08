"""Filter sft_math_100k_v2.jsonl to only keep samples that:
1. Have </think> in response
2. Have \boxed{} in response
3. Total tokenized length <= 8192 tokens

Also adds "Question: ... Please reason step by step..." instruction to user prompt.
"""
import json
import sys
from transformers import AutoTokenizer

INPUT = "/workspace/data/sft_math_100k_v2.jsonl"
OUTPUT = "/workspace/data/sft_math_8k_complete.jsonl"
MODEL = "/root/.cache/huggingface/Qwen3-4B-Base"
MAX_LEN = 8192

print(f"Loading tokenizer from {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)

total = 0
kept = 0
no_think = 0
no_boxed = 0
too_long = 0

with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        total += 1
        d = json.loads(line)
        user_msg = d["messages"][0]["content"]
        asst_msg = d["messages"][1]["content"]

        # Check format
        if "</think>" not in asst_msg:
            no_think += 1
            continue
        if "\\boxed{" not in asst_msg and "\\boxed " not in asst_msg:
            no_boxed += 1
            continue

        # Add instruction to user prompt
        enhanced_user = f"Question: {user_msg}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

        # Tokenize full conversation to check length
        full_text = f"{enhanced_user}\n{asst_msg}"
        tokens = tok.encode(full_text)
        if len(tokens) > MAX_LEN:
            too_long += 1
            continue

        # Save with enhanced prompt
        record = {
            "messages": [
                {"role": "user", "content": enhanced_user},
                {"role": "assistant", "content": asst_msg},
            ]
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept += 1

        if total % 10000 == 0:
            print(f"  [{total}] kept={kept} no_think={no_think} no_boxed={no_boxed} too_long={too_long}")

print(f"\nDone!")
print(f"  Total: {total}")
print(f"  Kept: {kept} ({kept*100//total}%)")
print(f"  No </think>: {no_think}")
print(f"  No boxed: {no_boxed}")
print(f"  Too long (>{MAX_LEN} tokens): {too_long}")
print(f"  Output: {OUTPUT}")
