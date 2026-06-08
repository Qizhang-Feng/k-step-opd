#!/bin/bash
# Generate SFT data using teacher model (Qwen3-8B) via SGLang
# This ensures teacher consistency for Lightning OPD
#
# Usage: docker exec k-step-opd bash /workspace/k-step-opd/scripts/generate-sft-data-sglang.sh
set -ex

TEACHER_MODEL=${TEACHER_MODEL:-/root/.cache/huggingface/Qwen3-8B}
PROMPTS=${PROMPTS:-/workspace/data/openthoughts3_prompts_300k.jsonl}
OUTPUT=${OUTPUT:-/workspace/data/sft_teacher_generated.jsonl}
PORT=30010
TP=8
MAX_TOKENS=16384
TEMPERATURE=0.7
TOP_P=0.9
BATCH_SIZE=8

echo "=== Generate SFT Data with Teacher ==="
echo "  Teacher: $TEACHER_MODEL"
echo "  Prompts: $PROMPTS"
echo "  Output:  $OUTPUT"

# Start SGLang server
python3 -m sglang.launch_server \
    --model-path $TEACHER_MODEL \
    --port $PORT \
    --tp $TP \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --max-total-tokens 32768 \
    > /tmp/sglang_gen.log 2>&1 &

echo "Waiting for server..."
for i in $(seq 1 180); do
    if curl -s http://127.0.0.1:$PORT/health_generate > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://127.0.0.1:$PORT/health_generate > /dev/null 2>&1; then
    echo "ERROR: Server failed to start"
    tail -20 /tmp/sglang_gen.log
    exit 1
fi

# Generate responses
python3 -c "
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_URL = 'http://127.0.0.1:${PORT}/generate'
MAX_TOKENS = ${MAX_TOKENS}
TEMPERATURE = ${TEMPERATURE}
TOP_P = ${TOP_P}
BATCH_SIZE = ${BATCH_SIZE}

# Load prompts
prompts = []
with open('${PROMPTS}') as f:
    for line in f:
        prompts.append(json.loads(line))
print(f'Loaded {len(prompts)} prompts')

# Format prompt as chat
def format_prompt(item):
    messages = item.get('prompt', [])
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]
    return messages

def generate_one(idx, item):
    messages = format_prompt(item)
    # Build chat text with template
    text = ''
    for msg in messages:
        if msg['role'] == 'system':
            text += f'<|im_start|>system\n{msg[\"content\"]}<|im_end|>\n'
        elif msg['role'] == 'user':
            text += f'<|im_start|>user\n{msg[\"content\"]}<|im_end|>\n'
    text += '<|im_start|>assistant\n'

    payload = {
        'text': text,
        'sampling_params': {
            'max_new_tokens': MAX_TOKENS,
            'temperature': TEMPERATURE,
            'top_p': TOP_P,
        },
    }
    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        response_text = result.get('text', '')
        # Ensure think tag
        if '</think>' in response_text and not response_text.strip().startswith('<think>'):
            response_text = '<think>\n' + response_text
        return idx, messages, response_text
    except Exception as e:
        print(f'  Error on prompt {idx}: {e}')
        return idx, messages, None

# Generate with parallel workers
output_path = '${OUTPUT}'
completed = 0
skipped = 0
start_time = time.time()

with open(output_path, 'w') as fout:
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = []
        for idx, item in enumerate(prompts):
            futures.append(executor.submit(generate_one, idx, item))

        for future in as_completed(futures):
            idx, messages, response_text = future.result()
            if response_text and len(response_text) > 100:
                record = {
                    'messages': messages + [{'role': 'assistant', 'content': response_text}]
                }
                fout.write(json.dumps(record) + '\n')
                completed += 1
            else:
                skipped += 1

            total = completed + skipped
            if total % 100 == 0:
                elapsed = time.time() - start_time
                rate = total / elapsed * 3600
                print(f'  [{total}/{len(prompts)}] completed={completed} skipped={skipped} rate={rate:.0f}/hr')

print(f'\nDone! {completed} samples saved, {skipped} skipped')
print(f'Output: {output_path}')
print(f'Time: {(time.time()-start_time)/3600:.1f}h')
"

# Cleanup
pkill -f sglang || true
echo "=== Done ==="
