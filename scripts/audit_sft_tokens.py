#!/usr/bin/env python3
"""Fast token-level audit: tokenize each sample ONCE and search special token IDs.

For Qwen3, </think> = id 151668, <think> = id 151667, <|im_end|> = id 151645.
\boxed{ does not have a single token id (BPE), so we still string-search the decoded
prefix only at the boundary check (cheap because we already have token IDs).

Run:
  ssh p5-4 "docker exec k-step-opd-sft python3 /workspace/k-step-opd/scripts/audit_sft_tokens.py"
"""
import json
import os
import sys
from collections import Counter

DATA_DIR = "/workspace/data"
FILES = [
    ("79K_v2_success",  f"{DATA_DIR}/teacher_sft_filtered.jsonl"),
    ("73K_new_shard",   f"{DATA_DIR}/teacher_extra_100k_filtered.jsonl"),
]
MAX_LENGTH = 16384
SAMPLE_N = 5000  # uniform sample per file for tokenization
TOKENIZER_PATHS = [
    "/root/.cache/huggingface/Qwen3-4B-Base",
    "/opt/dlami/nvme/qzf/models/Qwen3-4B-Base",
]


def get_tokenizer():
    from transformers import AutoTokenizer
    for tp in TOKENIZER_PATHS:
        if os.path.exists(tp):
            return AutoTokenizer.from_pretrained(tp)
    raise RuntimeError("tokenizer not found")


def find_token_id(tokenizer, surface):
    """Find the unique token id for a surface form, if it's a single special token."""
    ids = tokenizer.convert_tokens_to_ids(surface)
    if ids is None or ids == tokenizer.unk_token_id:
        return None
    return ids


def audit(label, path, tokenizer, close_id, im_end_id):
    print(f"\n{'='*70}")
    print(f"=== {label}: {path} ===")
    print(f"{'='*70}")
    if not os.path.exists(path):
        print("MISSING")
        return

    # count total lines for sampling
    with open(path) as f:
        total = sum(1 for _ in f)
    step = max(total // SAMPLE_N, 1)
    sample_idxs = set(range(0, total, step))
    print(f"total samples: {total}, sampling every {step} → ~{len(sample_idxs)}")

    # accumulators
    n_total = 0
    n_no_close_raw = 0
    n_no_box_raw = 0
    tok_lens = []
    n_over_max = 0
    n_close_in = 0
    n_close_outside = 0
    n_close_missing_in_tokens = 0  # has </think> in raw text but no close_id in tokens
    n_box_in = 0
    n_box_outside = 0
    char_lens_resp = []

    with open(path) as f:
        for i, line in enumerate(f):
            try:
                ex = json.loads(line)
            except Exception:
                continue
            n_total += 1
            msgs = ex.get("messages", [])
            if not msgs:
                continue
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            char_lens_resp.append(len(assistant))

            has_close_raw = "</think>" in assistant
            has_box_raw = "\\boxed{" in assistant
            if not has_close_raw:
                n_no_close_raw += 1
            if not has_box_raw:
                n_no_box_raw += 1

            if i not in sample_idxs:
                continue

            # tokenize ONCE with apply_chat_template
            try:
                ids = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False, return_tensors=None
                )
                # Some tokenizers return BatchEncoding even when tokenize=True; force list of ints.
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                elif not isinstance(ids, list) or (len(ids) and not isinstance(ids[0], int)):
                    # if it's a list-of-lists or similar, flatten/coerce
                    ids = list(ids)
                # final sanity: ids must be a flat list of ints
                if len(ids) and not isinstance(ids[0], int):
                    # nested case
                    ids = list(ids[0]) if hasattr(ids[0], "__iter__") else list(ids)
            except Exception as e:
                continue
            tok_lens.append(len(ids))
            if len(ids) > MAX_LENGTH:
                n_over_max += 1

            # find LAST </think> token
            if close_id is not None:
                last_close = -1
                for idx in range(len(ids) - 1, -1, -1):
                    if ids[idx] == close_id:
                        last_close = idx
                        break
                if last_close == -1:
                    if has_close_raw:
                        n_close_missing_in_tokens += 1
                else:
                    if last_close < MAX_LENGTH:
                        n_close_in += 1
                    else:
                        n_close_outside += 1

            # for \boxed{ — find via decoded string at token boundary
            # cheap heuristic: decode whole, but we already have it as text. Instead, find
            # last "\boxed{" in the assistant text and convert text position -> approx token.
            if has_box_raw:
                # Approximate: tokenize prefix up to last \boxed{
                box_pos = assistant.rfind("\\boxed{")
                # The chat template prefix length in tokens is roughly the system+user portion.
                # Easier: tokenize just the assistant prefix up to box_pos.
                try:
                    prefix_ids = tokenizer(assistant[:box_pos], add_special_tokens=False).input_ids
                except Exception:
                    prefix_ids = []
                # offset = tokens_before_assistant ≈ len(ids) - len(assistant_tokens)
                # we approximate: just check if prefix tokens within assistant alone exceed max
                # Simpler: if total ids > MAX and assistant alone > max chars, likely cut
                # Use full tokenization heuristic:
                approx_box_token_pos = len(ids) - len(tokenizer(assistant[box_pos:], add_special_tokens=False).input_ids)
                if approx_box_token_pos < MAX_LENGTH:
                    n_box_in += 1
                else:
                    n_box_outside += 1

    show = max(n_total, 1)
    sample = max(len(tok_lens), 1)
    print(f"\n--- char-level (full {n_total}) ---")
    print(f"  no </think> raw:   {n_no_close_raw}/{n_total} ({n_no_close_raw/show*100:.2f}%)")
    print(f"  no \\boxed raw:     {n_no_box_raw}/{n_total} ({n_no_box_raw/show*100:.2f}%)")
    if char_lens_resp:
        s = sorted(char_lens_resp)
        n = len(s)
        print(f"  resp chars:        mean={sum(s)//n}  p50={s[n//2]}  p90={s[int(n*0.9)]}  p99={s[int(n*0.99)]}  max={s[-1]}")

    print(f"\n--- token-level (sample {sample}, max_length={MAX_LENGTH}) ---")
    if tok_lens:
        s = sorted(tok_lens)
        n = len(s)
        print(f"  full tokens:       mean={sum(s)//n}  p50={s[n//2]}  p90={s[int(n*0.9)]}  p99={s[int(n*0.99)]}  max={s[-1]}")
    print(f"  over max_length:   {n_over_max}/{sample} ({n_over_max/sample*100:.2f}%)  [WOULD BE DELETED by truncation_strategy=delete]")
    print(f"  </think> in:       {n_close_in}/{sample} ({n_close_in/sample*100:.2f}%)  [token < max_length]")
    print(f"  </think> outside:  {n_close_outside}/{sample} ({n_close_outside/sample*100:.2f}%)  [exists but at token >= max_length]")
    print(f"  </think> raw->no tok: {n_close_missing_in_tokens}/{sample}  [</think> in raw text but no close token id in chat-templated ids]")
    print(f"  \\boxed in:         {n_box_in}/{sample} ({n_box_in/sample*100:.2f}%)")
    print(f"  \\boxed outside:    {n_box_outside}/{sample} ({n_box_outside/sample*100:.2f}%)")


def main():
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    close_id = find_token_id(tokenizer, "</think>")
    open_id = find_token_id(tokenizer, "<think>")
    im_end_id = find_token_id(tokenizer, "<|im_end|>")
    print(f"  </think> id={close_id}, <think> id={open_id}, <|im_end|> id={im_end_id}")
    if close_id is None:
        print("WARNING: </think> not a single token id, will fall back to string search")

    for label, path in FILES:
        audit(label, path, tokenizer, close_id, im_end_id)


if __name__ == "__main__":
    main()
