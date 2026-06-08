#!/usr/bin/env python3
"""
Generate SFT data using Qwen3-8B teacher via SGLang.
Ensures teacher consistency for Lightning OPD.

Usage (inside container):
    python3 /workspace/k-step-opd/scripts/generate-teacher-sft-data.py \
        --server-url http://127.0.0.1:30010 \
        --prompt-source openthoughts3 \
        --num-samples 100000 \
        --output /workspace/data/sft_teacher_qwen3_8b_100k.jsonl
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def extract_prompts_from_openthoughts3(num_samples=100000, seed=42):
    """Download and extract prompts from OpenThoughts3-1.2M."""
    from datasets import load_dataset
    print(f"Loading OpenThoughts3-1.2M from HuggingFace...")
    ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    print(f"Total samples: {len(ds)}")

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    indices.sort()

    prompts = []
    for idx in indices:
        sample = ds[idx]
        # Extract user prompt from conversations
        if "conversations" in sample:
            for turn in sample["conversations"]:
                role = turn.get("from", turn.get("role", ""))
                content = turn.get("value", turn.get("content", ""))
                if role in ("human", "user") and content:
                    prompts.append(content)
                    break
        elif "prompt" in sample:
            if isinstance(sample["prompt"], str):
                prompts.append(sample["prompt"])
            elif isinstance(sample["prompt"], list):
                for m in sample["prompt"]:
                    if m.get("role") == "user":
                        prompts.append(m["content"])
                        break

    print(f"Extracted {len(prompts)} prompts")
    return prompts


def extract_prompts_from_jsonl(path, num_samples=100000):
    """Extract prompts from existing SFT jsonl file."""
    prompts = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            messages = item.get("messages", [])
            for m in messages:
                if m["role"] == "user":
                    prompts.append(m["content"])
                    break
    random.shuffle(prompts)
    return prompts[:num_samples]


def generate_one(server_url, prompt, max_tokens=16384, temperature=0.7, top_p=0.9):
    """Generate a single response from the teacher."""
    # Format as chat
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    resp = requests.post(f"{server_url}/generate", json=payload, timeout=600)
    resp.raise_for_status()
    result = resp.json()
    response_text = result.get("text", "")

    # Ensure think tag
    if "</think>" in response_text and not response_text.strip().startswith("<think>"):
        response_text = "<think>\n" + response_text

    return response_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:30010")
    parser.add_argument("--prompt-source", type=str, default="openthoughts3",
                        choices=["openthoughts3", "jsonl"],
                        help="Source of prompts")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to jsonl file (when prompt-source=jsonl)")
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load prompts
    if args.prompt_source == "openthoughts3":
        prompts = extract_prompts_from_openthoughts3(args.num_samples, args.seed)
    else:
        prompts = extract_prompts_from_jsonl(args.prompt_file, args.num_samples)

    print(f"\n=== Generating {len(prompts)} responses ===")
    print(f"  Server: {args.server_url}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {args.output}")

    # Generate
    completed = 0
    skipped = 0
    start_time = time.time()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for idx, prompt in enumerate(prompts):
                future = executor.submit(
                    generate_one, args.server_url, prompt,
                    args.max_tokens, args.temperature, args.top_p
                )
                futures[future] = (idx, prompt)

            for future in as_completed(futures):
                idx, prompt = futures[future]
                try:
                    response_text = future.result()
                    if response_text and len(response_text) > 100:
                        record = {
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": response_text},
                            ]
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()
                        completed += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 5:
                        print(f"  Error: {e}")

                total = completed + skipped
                if total % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = total / elapsed * 3600
                    eta = (len(prompts) - total) / (total / elapsed) / 3600 if total > 0 else 0
                    print(f"  [{total}/{len(prompts)}] done={completed} skip={skipped} "
                          f"rate={rate:.0f}/hr ETA={eta:.1f}h")

    elapsed = time.time() - start_time
    print(f"\n=== Done! ===")
    print(f"  Completed: {completed}")
    print(f"  Skipped: {skipped}")
    print(f"  Time: {elapsed/3600:.1f}h")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
