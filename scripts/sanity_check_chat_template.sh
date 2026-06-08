#!/bin/bash
# Sanity check: does Qwen3-4B SFT v2 auto-emit <think> when prompted via the
# canonical chat template (without manual <think>\n injection)?
#
# Compares two prompt formats:
#   (A) slime/Lightning-OPD style: tokenizer.apply_chat_template(...) with no manual think
#   (B) our eval_math.py style:    same as (A) but with manual <think>\n appended
#
# For each format, runs a single AIME problem N times and reports:
#   - rate of responses starting with "<think>"
#   - rate with </think>
#   - rate with \boxed{}
#   - first 200 chars of each
set -ex

PORT=30030
MODEL_PATH=/root/.cache/huggingface/sft-qwen3-4b-full-v2-ckpt700
N_SAMPLES=${N_SAMPLES:-8}

pkill -9 -f sglang 2>/dev/null || true
sleep 5

nohup python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port $PORT \
    --dp 8 \
    --mem-fraction-static 0.8 \
    --max-total-tokens 32768 \
    --trust-remote-code \
    > /tmp/sglang-sanity.log 2>&1 &

echo "Waiting for SGLang..."
for i in $(seq 1 240); do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if [ $i -eq 240 ]; then
        echo "ERROR: server failed to start"
        tail -40 /tmp/sglang-sanity.log
        exit 1
    fi
    sleep 1
done

python3 - <<PY
import json, requests
from transformers import AutoTokenizer

PORT = $PORT
N = $N_SAMPLES
tok = AutoTokenizer.from_pretrained("$MODEL_PATH")

# Use one AIME-2024 problem
problems = []
with open("/workspace/data/aime-2024/aime-2024.jsonl") as f:
    for line in f:
        problems.append(json.loads(line))

p = problems[0]
print(f"=== Problem: {p['prompt'][:200]}...")
print(f"=== Label: {p['label']}")
print()

# Format (A): Lightning-OPD style — paper's exact template
#   <|im_start|>user
#   Question: {problem}
#   Please reason step by step, and put your final answer within \\boxed{}.
#   <|im_end|>
#   <|im_start|>assistant
prompt_paper = (
    "<|im_start|>user\n"
    f"Question: {p['prompt']}\n"
    "Please reason step by step, and put your final answer within \\\\boxed{}.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Format (B): apply_chat_template (slime default, enable_thinking=True implicit)
prompt_slime = tok.apply_chat_template(
    [{"role": "user", "content": p["prompt"]}],
    tokenize=False, add_generation_prompt=True,
)

# Format (C): our eval_math.py — same as B but force <think>\\n
prompt_ours = prompt_slime + "<think>\n"

formats = {
    "A_paper":  prompt_paper,
    "B_slime":  prompt_slime,
    "C_ours":   prompt_ours,
}

def gen(prompt, n):
    resp = requests.post(
        f"http://127.0.0.1:{PORT}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 16384,
                "temperature": 0.6,
                "top_p": 0.95,
                "n": n,
            },
        },
        timeout=600,
    )
    resp.raise_for_status()
    out = resp.json()
    return [item["text"] for item in out] if isinstance(out, list) else [out["text"]]

for name, prompt in formats.items():
    print(f"\\n========== Format {name} ==========")
    print(f"Prompt repr (last 80 chars): {repr(prompt[-80:])}")
    responses = gen(prompt, N)
    starts_think = sum(1 for r in responses if r.lstrip().startswith("<think>"))
    has_close = sum(1 for r in responses if "</think>" in r)
    has_boxed = sum(1 for r in responses if "\\\\boxed{" in r)
    avg_len = sum(len(r) for r in responses) / N
    print(f"  starts_with_<think>: {starts_think}/{N}")
    print(f"  has_</think>:        {has_close}/{N}")
    print(f"  has_\\\\boxed{{}}:        {has_boxed}/{N}")
    print(f"  avg_len chars:       {avg_len:.0f}")
    print(f"  sample 0 first 250:  {responses[0][:250]!r}")
PY

pkill -9 -f sglang 2>/dev/null || true
sleep 3
echo "=== sanity check done ==="
