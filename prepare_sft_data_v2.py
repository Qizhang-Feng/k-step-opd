"""
Prepare SFT data from local OpenThoughts3-1.2M with strict filtering.
Reads from local parquet files (no network). Uses multiprocessing + tqdm.

Filters:
  1. Math domain only
  2. Has \boxed{} in response
  3. <think> and </think> both present and complete
  4. Tokenized length (user + assistant) <= MAX_TOKENS
  5. \boxed{} appears within the token limit
  6. No severe repetition

Usage:
  python prepare_sft_data_v2.py [--max-tokens 16384] [--target 100000] [--workers 8]
"""

import json
import argparse
import os
from multiprocessing import Pool
from functools import partial
from transformers import AutoTokenizer
from tqdm import tqdm
import pyarrow.parquet as pq

# Global tokenizer (initialized per worker)
_tokenizer = None

def init_worker(model_path):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def has_repetition(text, chunk_size=100, threshold=5):
    if len(text) < chunk_size * threshold:
        return False
    for start in range(1000, min(len(text) - chunk_size, 5000), 500):
        chunk = text[start:start + chunk_size]
        if text.count(chunk) >= threshold:
            return True
    return False

def check_think_tags(response):
    has_open = "<think>" in response
    has_close = "</think>" in response
    if not has_open and not has_close:
        return True
    if has_open and not has_close:
        return False
    if not has_open and has_close:
        return True
    open_pos = response.index("<think>")
    close_pos = response.rindex("</think>")
    return close_pos > open_pos

def process_row(row, max_tokens):
    """Process a single row dict. Returns (record, token_len, reason)."""
    global _tokenizer

    source = row.get("source", "") or ""
    domain = row.get("domain", "") or ""
    if "math" not in domain.lower() and "math" not in source.lower():
        return None, 0, "not_math"

    convs = row.get("conversations", [])
    if not convs or len(convs) < 2:
        return None, 0, "no_convs"
    prompt = convs[0].get("value", "") if convs[0].get("from") == "human" else ""
    response = convs[1].get("value", "") if convs[1].get("from") == "gpt" else ""

    if not prompt or not response:
        return None, 0, "no_convs"

    if "\\boxed" not in response:
        return None, 0, "no_boxed"

    if not check_think_tags(response):
        return None, 0, "think_incomplete"

    if has_repetition(response):
        return None, 0, "repetition"

    # Quick char pre-filter
    if len(prompt) + len(response) > max_tokens * 6:
        return None, 0, "too_long"

    # Tokenize
    full_text = prompt + response
    tokens = _tokenizer.encode(full_text)
    token_len = len(tokens)

    if token_len > max_tokens:
        return None, 0, "too_long"

    # Check boxed within limit
    boxed_pos = response.rfind("\\boxed{")
    if boxed_pos == -1:
        return None, 0, "no_boxed"

    depth = 0
    end_pos = boxed_pos + len("\\boxed{")
    for i in range(end_pos, min(len(response), end_pos + 500)):
        if response[i] == '{':
            depth += 1
        elif response[i] == '}':
            if depth == 0:
                end_pos = i + 1
                break
            depth -= 1

    text_to_boxed = prompt + response[:end_pos]
    tokens_to_boxed = _tokenizer.encode(text_to_boxed)
    if len(tokens_to_boxed) > max_tokens:
        return None, 0, "boxed_truncated"

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return messages, token_len, "kept"


def process_batch(rows, max_tokens):
    """Process a batch of rows."""
    results = []
    for row in rows:
        messages, token_len, reason = process_row(row, max_tokens)
        results.append((messages, token_len, reason))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--target", type=int, default=100000)
    parser.add_argument("--output", type=str, default="/workspace/data/sft_math_100k_v2.jsonl")
    parser.add_argument("--model", type=str, default="/root/.cache/huggingface/Qwen3-8B-Base")
    parser.add_argument("--data-dir", type=str, default="/workspace/data/OpenThoughts3-1.2M/data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    print(f"Config: max_tokens={args.max_tokens}, target={args.target}, workers={args.workers}")
    print(f"Output: {args.output}")
    print(f"Data dir: {args.data_dir}")
    print(f"Model: {args.model}")

    # Find parquet files
    parquet_files = sorted([
        os.path.join(args.data_dir, f) 
        for f in os.listdir(args.data_dir) 
        if f.endswith(".parquet")
    ])
    print(f"Found {len(parquet_files)} parquet files")

    stats = {
        "total_seen": 0,
        "not_math": 0,
        "no_convs": 0,
        "no_boxed": 0,
        "think_incomplete": 0,
        "too_long": 0,
        "boxed_truncated": 0,
        "repetition": 0,
        "kept": 0,
    }
    token_lengths = []
    count = 0

    pbar = tqdm(total=args.target, desc="Filtering", unit="samples")

    with Pool(args.workers, initializer=init_worker, initargs=(args.model,)) as pool:
        with open(args.output, "w") as f:
            for pf in parquet_files:
                if count >= args.target:
                    break

                # Read parquet file
                table = pq.read_table(pf)
                rows = table.to_pylist()
                stats["total_seen"] += len(rows)

                # Split into batches for workers
                batches = [rows[i:i+args.batch_size] for i in range(0, len(rows), args.batch_size)]
                fn = partial(process_batch, max_tokens=args.max_tokens)
                
                for batch_results in pool.imap_unordered(fn, batches):
                    for messages, token_len, reason in batch_results:
                        if messages is not None:
                            record = {"messages": messages}
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            token_lengths.append(token_len)
                            count += 1
                            pbar.update(1)
                        else:
                            stats[reason] = stats.get(reason, 0) + 1
                    
                    if count >= args.target:
                        break

    pbar.close()
    stats["kept"] = count

    print(f"\n{'='*60}")
    print(f"Done: {count} samples saved to {args.output}")
    print(f"{'='*60}")
    print(f"\nFilter stats (total seen: {stats['total_seen']}):")
    for k, v in sorted(stats.items()):
        if k != "total_seen":
            print(f"  {k:20s}: {v:>7d}")

    if token_lengths:
        token_lengths.sort()
        n = len(token_lengths)
        print(f"\nToken length distribution ({n} samples):")
        print(f"  min : {token_lengths[0]:>6d}")
        print(f"  p25 : {token_lengths[n//4]:>6d}")
        print(f"  p50 : {token_lengths[n//2]:>6d}")
        print(f"  p75 : {token_lengths[int(n*0.75)]:>6d}")
        print(f"  p90 : {token_lengths[int(n*0.9)]:>6d}")
        print(f"  p95 : {token_lengths[int(n*0.95)]:>6d}")
        print(f"  p99 : {token_lengths[int(n*0.99)]:>6d}")
        print(f"  max : {token_lengths[-1]:>6d}")
        print(f"  mean: {sum(token_lengths)/n:>6.0f}")

if __name__ == "__main__":
    main()
