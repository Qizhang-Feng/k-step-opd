"""Convert MATH-500 jsonl to slime eval format."""
import json

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: \\boxed{{$Answer}} "
    "where $Answer is the answer to the problem.\n\n{problem}\n\n"
    "Remember to put your answer on its own line after \"Answer:\"."
)

records = []
with open("/workspace/data/math-500/test.jsonl") as f:
    for line in f:
        d = json.loads(line)
        records.append({
            "prompt": PROMPT_TEMPLATE.format(problem=d["problem"]),
            "label": d["answer"],
            "reward_model": "math",
        })

out_path = "/workspace/data/math-500/math-500.jsonl"
with open(out_path, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"Wrote {len(records)} problems to {out_path}")
