"""Convert OpenThoughts3 math data to ms-swift messages format."""
import json
import sys

input_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/openthoughts3_math_100k.jsonl"
output_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/data/sft_math_100k_messages.jsonl"

count = 0
with open(input_path) as fin, open(output_path, "w") as fout:
    for line in fin:
        d = json.loads(line)
        prompt = d["prompt"]
        response = d["response"]
        
        # ms-swift messages format
        record = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"Converted {count} samples: {input_path} → {output_path}")
