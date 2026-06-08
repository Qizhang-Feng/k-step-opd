#!/usr/bin/env python3
"""Audit and compare 79K (v2 success) vs 152K (fail) and the 73K new shard.

Run on p5-4 inside k-step-opd-sft container (has Qwen tokenizer):
  ssh p5-4 "docker exec k-step-opd-sft python3 /workspace/k-step-opd/scripts/audit_sft_data.py"
"""
import json
import os
import sys
from collections import Counter

DATA_DIR = "/workspace/data"
FILES = [
    ("79K_v2_success",      f"{DATA_DIR}/teacher_sft_filtered.jsonl"),
    ("73K_new_shard",       f"{DATA_DIR}/teacher_extra_100k_filtered.jsonl"),
    ("152K_fail",           f"{DATA_DIR}/teacher_sft_179k_thinkfilter.jsonl"),
]
MAX_LENGTH = 16384  # same as training


def stats_for(label, path, tokenizer=None, sample_n=None):
    print(f"\n{'='*70}")
    print(f"=== {label}: {path} ===")
    print(f"{'='*70}")
    if not os.path.exists(path):
        print("MISSING")
        return

    n_total = 0
    n_no_close = 0
    n_no_box = 0
    n_close_at_end = 0  # </think> within last 1000 chars
    n_box_at_end = 0
    char_lens = []
    response_char_lens = []
    n_garbage = 0
    n_thai = 0
    n_repeat_token = 0
    box_positions = []  # fraction position
    close_positions = []
    tok_lens_resp = []
    n_over_max = 0
    n_close_outside_max = 0
    n_box_outside_max = 0

    # sample for token-level audit
    sample_indices = None
    if sample_n is not None:
        # uniform sample
        with open(path) as f:
            total = sum(1 for _ in f)
        step = max(total // sample_n, 1)
        sample_indices = set(range(0, total, step))

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
            full = "\n".join(m["content"] for m in msgs)

            char_lens.append(len(full))
            response_char_lens.append(len(assistant))

            has_close = "</think>" in assistant
            has_box = "\\boxed{" in assistant

            if not has_close:
                n_no_close += 1
            if not has_box:
                n_no_box += 1

            if has_close:
                pos = assistant.rfind("</think>") / max(len(assistant), 1)
                close_positions.append(pos)
                if (len(assistant) - assistant.rfind("</think>")) < 1000:
                    n_close_at_end += 1
            if has_box:
                pos = assistant.rfind("\\boxed{") / max(len(assistant), 1)
                box_positions.append(pos)
                if (len(assistant) - assistant.rfind("\\boxed{")) < 1000:
                    n_box_at_end += 1

            # garbage detection: look for common degeneration patterns in last 2000 chars
            tail = assistant[-2000:]
            if "printStackTrace" in tail or ",,,,,,,,,," in tail or " \\ \\ \\ \\ \\ " in tail:
                n_garbage += 1
            # Thai chars (failure mode reported in worklog v1/v3)
            if any('\u0E00' <= c <= '\u0E7F' for c in assistant[:500]):
                n_thai += 1
            # repetition: count " 0 0 0 0 0" or runs
            for run in [" 0 0 0 0 0", " . . . . .", " - - - - -"]:
                if run in tail:
                    n_repeat_token += 1
                    break

            # token-level audit on sample
            if tokenizer is not None and sample_indices is not None and i in sample_indices:
                # apply chat template if available
                try:
                    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                except Exception:
                    text = full
                ids = tokenizer(text, add_special_tokens=False).input_ids
                tok_lens_resp.append(len(ids))
                if len(ids) > MAX_LENGTH:
                    n_over_max += 1
                # find token positions of </think> and \boxed{
                if has_close:
                    # decode-based approximate position
                    try:
                        close_idx = text.rfind("</think>")
                        prefix_ids = tokenizer(text[:close_idx], add_special_tokens=False).input_ids
                        if len(prefix_ids) >= MAX_LENGTH:
                            n_close_outside_max += 1
                    except Exception:
                        pass
                if has_box:
                    try:
                        box_idx = text.rfind("\\boxed{")
                        prefix_ids = tokenizer(text[:box_idx], add_special_tokens=False).input_ids
                        if len(prefix_ids) >= MAX_LENGTH:
                            n_box_outside_max += 1
                    except Exception:
                        pass

    n_show = max(n_total, 1)
    print(f"total samples:        {n_total}")
    print(f"no </think>:          {n_no_close}/{n_total} ({n_no_close/n_show*100:.2f}%)")
    print(f"no \\boxed:            {n_no_box}/{n_total} ({n_no_box/n_show*100:.2f}%)")
    print(f"</think> near end:    {n_close_at_end}/{n_total} ({n_close_at_end/n_show*100:.2f}%)  [last 1000 chars]")
    print(f"\\boxed near end:      {n_box_at_end}/{n_total} ({n_box_at_end/n_show*100:.2f}%)")
    print(f"GARBAGE patterns:     {n_garbage}/{n_total} ({n_garbage/n_show*100:.2f}%)  [printStackTrace, ,,,,, , \\ \\ \\ \\ in last 2K chars]")
    print(f"Thai chars head:      {n_thai}/{n_total} ({n_thai/n_show*100:.2f}%)")
    print(f"repetition pattern:   {n_repeat_token}/{n_total} ({n_repeat_token/n_show*100:.2f}%)  [' 0 0 0 0 0' / ' . . . . .' / ' - - - - -']")
    if response_char_lens:
        sorted_l = sorted(response_char_lens)
        n = len(sorted_l)
        print(f"response chars:      mean={sum(sorted_l)/n:.0f}  p50={sorted_l[n//2]}  p90={sorted_l[int(n*0.9)]}  p99={sorted_l[int(n*0.99)]}  max={sorted_l[-1]}")
    if close_positions:
        sorted_c = sorted(close_positions)
        n = len(sorted_c)
        print(f"</think> rel pos:    mean={sum(sorted_c)/n:.3f}  p50={sorted_c[n//2]:.3f}  p10={sorted_c[n//10]:.3f}  p90={sorted_c[int(n*0.9)]:.3f}")
    if box_positions:
        sorted_b = sorted(box_positions)
        n = len(sorted_b)
        print(f"\\boxed rel pos:      mean={sum(sorted_b)/n:.3f}  p50={sorted_b[n//2]:.3f}  p10={sorted_b[n//10]:.3f}  p90={sorted_b[int(n*0.9)]:.3f}")
    if tok_lens_resp:
        sorted_t = sorted(tok_lens_resp)
        n = len(sorted_t)
        print(f"--- TOKEN AUDIT (sample {n}) ---")
        print(f"  full tokens:        mean={sum(sorted_t)/n:.0f}  p50={sorted_t[n//2]}  p90={sorted_t[int(n*0.9)]}  p99={sorted_t[int(n*0.99)]}  max={sorted_t[-1]}")
        print(f"  over max_length={MAX_LENGTH}: {n_over_max}/{n} ({n_over_max/n*100:.1f}%)")
        print(f"  </think> outside max: {n_close_outside_max}/{n}  (= would be cut by truncation)")
        print(f"  \\boxed outside max:   {n_box_outside_max}/{n}")


def main():
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tok_paths = [
            "/root/.cache/huggingface/Qwen3-4B-Base",
            "/opt/dlami/nvme/qzf/models/Qwen3-4B-Base",
        ]
        for tp in tok_paths:
            if os.path.exists(tp):
                tokenizer = AutoTokenizer.from_pretrained(tp)
                print(f"Loaded tokenizer from {tp}")
                break
        if tokenizer is None:
            print("WARNING: tokenizer not found, skipping token-level audit")
    except Exception as e:
        print(f"WARNING: failed to load tokenizer: {e}")

    for label, path in FILES:
        stats_for(label, path, tokenizer=tokenizer, sample_n=5000)


if __name__ == "__main__":
    main()
