#!/bin/bash
# Generate teacher SFT data: 8 SGLang replicas (TP=1 each) on 8 GPUs
# Async concurrent client for maximum throughput
set -e

START=${1:?Usage: $0 <start> <end> <output_dir>}
END=${2:?}
OUTPUT_DIR=${3:?}
TEACHER_MODEL=${TEACHER_MODEL:-/root/.cache/huggingface/Qwen3-8B}
DATA_FILE=${DATA_FILE:-/data/math_prompts_100k.jsonl}
MAX_TOKENS=${MAX_TOKENS:-16384}
BASE_PORT=30010
CONCURRENCY=24

echo "=== 8-Replica Teacher Generation (Async) ==="
echo "  Range: $START to $END"
echo "  Model: $TEACHER_MODEL"
echo "  Data: $DATA_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Concurrency per GPU: $CONCURRENCY"

# Cleanup on exit
trap 'pkill -9 -f sglang 2>/dev/null || true' EXIT

# Kill old processes
pkill -9 -f sglang 2>/dev/null || true
sleep 3

mkdir -p $OUTPUT_DIR

# Start 8 engines (TP=1 each)
for i in $(seq 0 7); do
    PORT=$((BASE_PORT + i))
    echo "Starting engine $i on GPU $i, port $PORT"
    CUDA_VISIBLE_DEVICES=$i python3 -m sglang.launch_server \
        --model-path $TEACHER_MODEL \
        --host 127.0.0.1 \
        --port $PORT \
        --tp 1 \
        --trust-remote-code \
        --context-length 20480 \
        --mem-fraction-static 0.92 \
        --max-running-requests $CONCURRENCY \
        --chunked-prefill-size 8192 \
        > /tmp/engine_${i}.log 2>&1 &
done

# Wait for all engines
echo "Waiting for all 8 engines..."
for attempt in $(seq 1 300); do
    READY=0
    for i in $(seq 0 7); do
        PORT=$((BASE_PORT + i))
        if curl -sf http://127.0.0.1:$PORT/health_generate > /dev/null 2>&1; then
            READY=$((READY + 1))
        fi
    done
    if [ $READY -eq 8 ]; then
        echo "All 8 engines ready after ${attempt}s"
        break
    fi
    if [ $attempt -eq 300 ]; then
        echo "ERROR: Only $READY/8 engines started"
        for i in $(seq 0 7); do
            echo "=== Engine $i ===" && tail -5 /tmp/engine_${i}.log
        done
        exit 1
    fi
    sleep 1
done

# Split data into 8 shards and generate with async clients
TOTAL=$((END - START))
SHARD_SIZE=$(( (TOTAL + 7) / 8 ))

GEN_PIDS=()

for i in $(seq 0 7); do
    SHARD_START=$((START + i * SHARD_SIZE))
    SHARD_END=$((SHARD_START + SHARD_SIZE))
    if [ $SHARD_END -gt $END ]; then
        SHARD_END=$END
    fi
    PORT=$((BASE_PORT + i))
    OUTPUT_FILE=$OUTPUT_DIR/shard_${i}.jsonl

    echo "Launching shard $i: $SHARD_START-$SHARD_END -> $OUTPUT_FILE (port $PORT)"

    python3 -c "
import asyncio, aiohttp, json, time, os

SHARD_START = $SHARD_START
SHARD_END = $SHARD_END
SERVER_URL = 'http://127.0.0.1:$PORT'
MAX_TOKENS = $MAX_TOKENS
CONCURRENCY = $CONCURRENCY
DATA_FILE = '$DATA_FILE'
OUTPUT_FILE = '$OUTPUT_FILE'

# Load prompts
prompts = []
with open(DATA_FILE) as f:
    for idx, line in enumerate(f):
        if idx < SHARD_START:
            continue
        if idx >= SHARD_END:
            break
        d = json.loads(line)
        prompts.append(d.get('prompt', d.get('messages', [{}])[0].get('content', '')))

print(f'Shard $i: {len(prompts)} prompts ({SHARD_START}-{SHARD_END})', flush=True)

async def generate_one(session, sem, prompt, local_idx):
    text = (
        f'<|im_start|>user\n'
        f'Question: {prompt}\n'
        f'Please reason step by step, and put your final answer within \\\\boxed{{}}.'
        f'<|im_end|>\n'
        f'<|im_start|>assistant\n'
    )
    payload = {
        'text': text,
        'sampling_params': {
            'max_new_tokens': MAX_TOKENS,
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
            'min_p': 0.0,
            'stop': ['<|im_end|>'],
        },
    }
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(
                    f'{SERVER_URL}/generate',
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=1800),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return local_idx, prompt, data.get('text', '')
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
    return local_idx, prompt, ''

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 4)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            generate_one(session, sem, prompt, i)
            for i, prompt in enumerate(prompts)
        ]

        completed = 0
        skipped = 0
        t0 = time.time()

        with open(OUTPUT_FILE, 'w') as fout:
            for fut in asyncio.as_completed(tasks):
                local_idx, prompt, response = await fut

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

                total = completed + skipped
                if total % 50 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed * 3600 if elapsed > 0 else 0
                    print(f'  Shard $i [{total}/{len(prompts)}] done={completed} skip={skipped} rate={rate:.0f}/hr', flush=True)

        elapsed = time.time() - t0
        print(f'Shard $i done: {completed} completed, {skipped} skipped, {elapsed/3600:.1f}h', flush=True)

asyncio.run(main())
" &
    GEN_PIDS+=($!)
done

# Wait only for generation clients (not servers)
echo "Waiting for all 8 generation clients..."
for pid in "${GEN_PIDS[@]}"; do
    wait "$pid"
done

echo "All shards complete. Merging..."
cat $OUTPUT_DIR/shard_*.jsonl > $OUTPUT_DIR/teacher_sft_merged.jsonl
TOTAL_LINES=$(wc -l < $OUTPUT_DIR/teacher_sft_merged.jsonl)
echo "=== Done! Total: $TOTAL_LINES samples in $OUTPUT_DIR/teacher_sft_merged.jsonl ==="
