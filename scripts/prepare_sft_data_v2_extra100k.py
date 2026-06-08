"""
Prepare extra 100K SFT data from OpenThoughts3-1.2M.
Filters: math domain, has </think>, word count <= 14000, no exact (prompt+response) duplicate with existing 100K.

Usage (inside k-step-opd container on p5-5):
  python scripts/prepare_sft_data_v2_extra100k.py
"""

import json
import os
import hashlib
from tqdm import tqdm
import pyarrow.parquet as pq

EXISTING_FILE = "/workspace/data/sft_math_100k_v2.jsonl"
OUTPUT_FILE = "/workspace/data/sft_math_extra_100k_v2.jsonl"
COMBINED_FILE = "/workspace/data/sft_math_200k_v2.jsonl"
DATA_DIR = "/workspace/data/OpenThoughts3-1.2M/data"
MAX_WORDS = 14000
TARGET = 100000


def main():
    # Load existing (prompt, response) pairs to avoid exact duplicates
    print(f"Loading existing samples from {EXISTING_FILE}...")
    existing_hashes = set()
    with open(EXISTING_FILE) as f:
        for line in f:
            d = json.loads(line)
            key = d["messages"][0]["content"][:200] + d["messages"][1]["content"][:200]
            existing_hashes.add(hashlib.md5(key.encode()).hexdigest())
    print(f"  Loaded {len(existing_hashes)} existing hashes to exclude")

    # Find parquet files
    parquet_files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".parquet")
    ])
    print(f"Found {len(parquet_files)} parquet files")

    count = 0
    stats = {
        "total_seen": 0, "not_math": 0, "no_convs": 0,
        "too_long": 0, "duplicate": 0, "no_think_close": 0, "kept": 0,
    }

    pbar = tqdm(total=TARGET, desc="Filtering extra 100K", unit="samples")

    with open(OUTPUT_FILE, "w") as fout:
        for pf in parquet_files:
            if count >= TARGET:
                break

            table = pq.read_table(pf)
            rows = table.to_pylist()
            stats["total_seen"] += len(rows)

            for row in rows:
                if count >= TARGET:
                    break

                domain = row.get("domain", "") or ""
                if "math" not in domain.lower():
                    stats["not_math"] += 1
                    continue

                convs = row.get("conversations", [])
                if not convs or len(convs) < 2:
                    stats["no_convs"] += 1
                    continue
                prompt = convs[0].get("value", "") if convs[0].get("from") == "human" else ""
                response = convs[1].get("value", "") if convs[1].get("from") == "gpt" else ""

                if not prompt or not response:
                    stats["no_convs"] += 1
                    continue

                # Must have </think>
                if "</think>" not in response:
                    stats["no_think_close"] += 1
                    continue

                # Word count filter
                total_text = prompt + response
                if len(total_text.split()) > MAX_WORDS:
                    stats["too_long"] += 1
                    continue

                # Dedup by (prompt[:200] + response[:200]) hash
                key = prompt[:200] + response[:200]
                h = hashlib.md5(key.encode()).hexdigest()
                if h in existing_hashes:
                    stats["duplicate"] += 1
                    continue
                existing_hashes.add(h)

                # Write
                record = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                pbar.update(1)

    pbar.close()
    stats["kept"] = count

    print(f"\nExtra done: {count} samples saved to {OUTPUT_FILE}")
    print(f"\nStats:")
    for k, v in sorted(stats.items()):
        print(f"  {k:20s}: {v:>7d}")

    # Combine into 200K
    print(f"\nCombining into {COMBINED_FILE}...")
    with open(COMBINED_FILE, "w") as fout:
        with open(EXISTING_FILE) as f1:
            for line in f1:
                fout.write(line)
        with open(OUTPUT_FILE) as f2:
            for line in f2:
                fout.write(line)

    total = sum(1 for _ in open(COMBINED_FILE))
    print(f"Combined: {total} samples in {COMBINED_FILE}")


if __name__ == "__main__":
    main()
