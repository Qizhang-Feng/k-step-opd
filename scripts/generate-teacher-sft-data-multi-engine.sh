#!/bin/bash
# Generate teacher-consistent SFT data using Qwen3-8B
# TP=2 x 4 engines for maximum throughput on 8 GPUs
#
# Usage:
#   docker exec -d k-step-opd bash /workspace/k-step-opd/scripts/generate-teacher-sft-data-multi-engine.sh <start> <end> <output>

set -ex

START=${1:?Usage: $0 <start_idx> <end_idx> <output_file>}
END=${2:?}
OUTPUT=${3:?}
TEACHER_MODEL=${TEACHER_MODEL:-/root/.cache/huggingface/Qwen3-8B}
MAX_TOKENS=${MAX_TOKENS:-16384}
DATA_FILE=${DATA_FILE:-/workspace/data/math_prompts_100k.jsonl}

NUM_ENGINES=4
TP=2
BASE_PORT=30010

echo "=== Teacher SFT Data Generation (Multi-Engine) ==="
echo "  Shard: $START to $END"
echo "  Output: $OUTPUT"
echo "  Teacher: $TEACHER_MODEL"
echo "  Engines: $NUM_ENGINES x TP=$TP"

# Kill old processes
pkill -9 -f sglang 2>/dev/null || true
sleep 3

# Start 4 engines on GPU pairs: (0,1), (2,3), (4,5), (6,7)
for i in $(seq 0 $((NUM_ENGINES-1))); do
    GPU_START=$((i * TP))
    GPU_END=$((GPU_START + TP - 1))
    GPUS=$(seq -s, $GPU_START $GPU_END)
    PORT=$((BASE_PORT + i))
    echo "Starting engine $i on GPUs $GPUS, port $PORT"
    CUDA_VISIBLE_DEVICES=$GPUS python3 -m sglang.launch_server \
        --model-path $TEACHER_MODEL \
        --port $PORT \
        --tp $TP \
        --trust-remote-code \
        --mem-fraction-static 0.85 \
        --max-running-requests 16 \
        --max-total-tokens 32768 \
        > /tmp/teacher_server_${i}.log 2>&1 &
done

# Wait for all engines
echo "Waiting for all engines..."
ALL_READY=0
for attempt in $(seq 1 180); do
    READY=0
    for i in $(seq 0 $((NUM_ENGINES-1))); do
        PORT=$((BASE_PORT + i))
        if curl -sf http://127.0.0.1:$PORT/health_generate > /dev/null 2>&1; then
            READY=$((READY + 1))
        fi
    done
    if [ $READY -eq $NUM_ENGINES ]; then
        echo "All $NUM_ENGINES engines ready after ${attempt}s"
        ALL_READY=1
        break
    fi
    sleep 1
done

if [ $ALL_READY -eq 0 ]; then
    echo "ERROR: Not all engines started"
    for i in $(seq 0 $((NUM_ENGINES-1))); do
        echo "=== Engine $i ==="
        tail -5 /tmp/teacher_server_${i}.log
    done
    exit 1
fi

# Generate using all engines with round-robin
python3 -c "
import json, time, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

START = $START
END = $END
OUTPUT = '$OUTPUT'
MAX_TOKENS = $MAX_TOKENS
DATA_FILE = '$DATA_FILE'
NUM_ENGINES = $NUM_ENGINES
BASE_PORT = $BASE_PORT

# Load prompts
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

server_urls = [f'http://127.0.0.1:{BASE_PORT + i}' for i in range(NUM_ENGINES)]

def generate_one(args):
    idx, prompt = args
    url = server_urls[idx % NUM_ENGINES]
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
            resp = requests.post(f'{url}/generate', json=payload, timeout=600)
            resp.raise_for_status()
            return resp.json().get('text', '')
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                raise

completed = 0
skipped = 0
t0 = time.time()

os.makedirs(os.path.dirname(OUTPUT) or '.', exist_ok=True)
with open(OUTPUT, 'w') as fout:
    with ThreadPoolExecutor(max_workers=NUM_ENGINES * 4) as executor:
        batch_size = NUM_ENGINES * 8
        for batch_start in range(0, len(prompts), batch_size):
            batch = prompts[batch_start:batch_start + batch_size]
            futures = {}
            for i, prompt in enumerate(batch, start=batch_start):
                future = executor.submit(generate_one, (i, prompt))
                futures[future] = (i, prompt)

            for future in as_completed(futures):
                i, prompt = futures[future]
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
                    if skipped <= 10:
                        print(f'  Error: {e}')

            total = completed + skipped
            if total % 100 == 0 or batch_start + batch_size >= len(prompts):
                elapsed = time.time() - t0
                rate = completed / elapsed * 3600 if elapsed > 0 else 0
                eta = (len(prompts) - total) / (completed / elapsed) / 3600 if completed > 0 else 999
                print(f'  [{total}/{len(prompts)}] done={completed} skip={skipped} rate={rate:.0f}/hr ETA={eta:.1f}h', flush=True)

elapsed = time.time() - t0
print(f'\nDone! completed={completed} skipped={skipped} time={elapsed/3600:.1f}h')
print(f'Output: {OUTPUT}')
"

# Cleanup
pkill -9 -f sglang 2>/dev/null || true
echo "=== Generation complete ==="
