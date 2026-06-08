#!/bin/bash
# Generate teacher-consistent SFT data using Qwen3-8B
# Uses prompts from sft_math_100k_v2.jsonl, generates new responses with Qwen3-8B
#
# Usage:
#   docker exec -d k-step-opd bash -c 'bash /workspace/k-step-opd/scripts/generate-teacher-sft-data-shard.sh <start> <end> <output>'
#
# Example (shard 0-33000):
#   bash generate-teacher-sft-data-shard.sh 0 33000 /workspace/data/teacher_sft_shard0.jsonl

set -ex

START=${1:?Usage: $0 <start_idx> <end_idx> <output_file>}
END=${2:?}
OUTPUT=${3:?}
PORT=${PORT:-30010}
TEACHER_MODEL=${TEACHER_MODEL:-/root/.cache/huggingface/Qwen3-8B}
TP=${TP:-8}
WORKERS=${WORKERS:-4}
MAX_TOKENS=${MAX_TOKENS:-16384}
DATA_FILE=${DATA_FILE:-/workspace/data/math_prompts_100k.jsonl}

echo "=== Teacher SFT Data Generation ==="
echo "  Shard: $START to $END"
echo "  Output: $OUTPUT"
echo "  Teacher: $TEACHER_MODEL (TP=$TP)"
echo "  Max tokens: $MAX_TOKENS"
echo "  Workers: $WORKERS"

# Start teacher server
pkill -9 -f sglang 2>/dev/null || true
sleep 3

python3 -m sglang.launch_server \
    --model-path $TEACHER_MODEL \
    --port $PORT \
    --tp $TP \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --max-running-requests 32 \
    --max-total-tokens 32768 \
    > /tmp/teacher_server.log 2>&1 &

echo "Waiting for teacher server..."
for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:$PORT/health_generate > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: Server failed to start"
        tail -20 /tmp/teacher_server.log
        exit 1
    fi
    sleep 1
done

# Generate
python3 -c "
import json, time, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

START = $START
END = $END
OUTPUT = '$OUTPUT'
SERVER_URL = 'http://127.0.0.1:$PORT'
MAX_TOKENS = $MAX_TOKENS
WORKERS = $WORKERS
DATA_FILE = '$DATA_FILE'

# Load prompts from existing SFT data
prompts = []
with open(DATA_FILE) as f:
    for i, line in enumerate(f):
        if i < START:
            continue
        if i >= END:
            break
        d = json.loads(line)
        prompts.append(d.get('prompt', d.get('messages', [{}])[0].get('content', '')))

print(f'Loaded {len(prompts)} prompts (index {START} to {END})')

def generate_one(prompt):
    text = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
    payload = {
        'text': text,
        'sampling_params': {
            'max_new_tokens': MAX_TOKENS,
            'temperature': 0.7,
            'top_p': 0.9,
        },
    }
    for attempt in range(5):
        try:
            resp = requests.post(f'{SERVER_URL}/generate', json=payload, timeout=600)
            resp.raise_for_status()
            result = resp.json()
            return result.get('text', '')
        except Exception as e:
            if attempt < 4:
                import time as _t
                _t.sleep(2 ** attempt)
            else:
                raise

completed = 0
skipped = 0
t0 = time.time()

os.makedirs(os.path.dirname(OUTPUT) or '.', exist_ok=True)
with open(OUTPUT, 'w') as fout:
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        # Submit in batches to avoid overwhelming the server
        batch_size = WORKERS * 2
        for batch_start in range(0, len(prompts), batch_size):
            batch = prompts[batch_start:batch_start + batch_size]
            futures = {}
            for idx, prompt in enumerate(batch, start=batch_start):
                future = executor.submit(generate_one, prompt)
                futures[future] = (idx, prompt)

            for future in as_completed(futures):
                idx, prompt = futures[future]
                try:
                    response = future.result()
                    if response and len(response) > 100:
                        if '</think>' in response and not response.strip().startswith('<think>'):
                            response = '<think>\n' + response
                        record = {
                            'messages': [
                                {'role': 'user', 'content': prompt},
                                {'role': 'assistant', 'content': response},
                            ]
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                        fout.flush()
                        completed += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 5:
                        print(f'  Error: {e}')

            total = completed + skipped
            if total % 200 == 0 or batch_start + batch_size >= len(prompts):
                elapsed = time.time() - t0
                rate = completed / elapsed * 3600 if elapsed > 0 else 0
                eta = (len(prompts) - total) / (completed / elapsed) / 3600 if completed > 0 else 999
                print(f'  [{total}/{len(prompts)}] done={completed} skip={skipped} rate={rate:.0f}/hr ETA={eta:.1f}h')

elapsed = time.time() - t0
print(f'\nDone! completed={completed} skipped={skipped} time={elapsed/3600:.1f}h')
print(f'Output: {OUTPUT}')
"

# Cleanup
pkill -9 -f sglang 2>/dev/null || true
echo "=== Generation complete ==="
