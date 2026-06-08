"""Convert AIME-2024 parquet to slime eval jsonl format."""
import json
import pandas as pd

df = pd.read_parquet("/workspace/data/aime-2024/aime_2024_problems.parquet")

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: \\boxed{{$Answer}} "
    "where $Answer is the answer to the problem.\n\n{problem}\n\n"
    "Remember to put your answer on its own line after \"Answer:\"."
)

records = []
for _, row in df.iterrows():
    records.append({
        "prompt": PROMPT_TEMPLATE.format(problem=row["Problem"]),
        "label": str(row["Answer"]),
        "reward_model": "math",
    })

out_path = "/workspace/data/aime-2024/aime-2024.jsonl"
with open(out_path, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"Wrote {len(records)} problems to {out_path}")
