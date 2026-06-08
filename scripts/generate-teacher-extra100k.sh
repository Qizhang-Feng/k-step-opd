#!/bin/bash
# Generate teacher SFT data on extra 100K prompts (shard)
# Uses Qwen3-8B teacher, 8 sglang engines (DP mode)
# Input: messages-format jsonl (extracts user content as prompt)
set -e

SHARD_FILE=${1:?Usage: $0 <shard_file> <output_file>}
OUTPUT_FILE=${2:?Usage: $0 <shard_file> <output_file>}
TEACHER_MODEL=${TEACHER_MODEL:-/root/.cache/huggingface/Qwen3-8B}
MAX_TOKENS=${MAX_TOKENS:-16384}
PORT=30010
CONCURRENCY=24

echo "=== Teacher Generation (DP-8) ==="
echo "  Input: $SHARD_FILE"
echo "  Output: $OUTPUT_FILE"
echo "  Model: $TEACHER_MODEL"
echo "  Max tokens: $MAX_TOKENS"

# Cleanup on exit
trap 'pkill -9 -f sglang 2>/dev/null || true' EXIT

# Kill old processes
pkill -9 -f sglang 2>/dev/null || true
sleep 3

# Start sglang with DP=8
python3 -m sglang.launch_server \
    --model-path $TEACHER_MODEL \
    --host 127.0.0.1 \
    --port $PORT \
    --dp-size 8 --tp 1 \
    --trust-remote-code \
    --context-length 20480 \
    --mem-fraction-static 0.88 \
    --chunked-prefill-size 8192 \
    > /tmp/sglang_teacher.log 2>&1 &

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:$PORT/health_generate > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: Server failed"
        tail -20 /tmp/sglang_teacher.log
        exit 1
    fi
    sleep 1
done

# Generate
python3 -c "
import asyncio, aiohttp, json, time, os

SERVER_URL = 'http://127.0.0.1:$PORT'
MAX_TOKENS = $MAX_TOKENS
CONCURRENCY = $CONCURRENCY * 8  # total across all DP workers
SHARD_FILE = '$SHARD_FILE'
OUTPUT_FILE = '$OUTPUT_FILE'

# Load prompts from messages format
prompts = []
with open(SHARD_FILE) as f:
    for line in f:
        d = json.loads(line)
        if 'messages' in d:
            prompts.append(d['messages'][0]['content'])
        else:
            prompts.append(d.get('prompt', ''))
print(f'Loaded {len(prompts)} prompts', flush=True)

# Resume
completed_prompts = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                row = json.loads(line)
                completed_prompts.add(row['messages'][0]['content'])
            except:
                pass
    print(f'Resuming: {len(completed_prompts)} already done', flush=True)

todo = [p for p in prompts if p not in completed_prompts]
print(f'Remaining: {len(todo)} prompts', flush=True)

if not todo:
    print('All done!')
    exit(0)

async def generate_one(session, sem, prompt):
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
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 20,
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
                    return prompt, data.get('text', '')
            except Exception as e:
                if attempt == 2:
                    print(f'Error: {e}', flush=True)
                await asyncio.sleep(2 * (attempt + 1))
    return prompt, ''

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 8)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [generate_one(session, sem, p) for p in todo]

        completed = 0
        skipped = 0
        t0 = time.time()

        with open(OUTPUT_FILE, 'a') as fout:
            for fut in asyncio.as_completed(tasks):
                prompt, response = await fut

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
                if total % 200 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed * 3600 if elapsed > 0 else 0
                    print(f'[{total}/{len(todo)}] done={completed} skip={skipped} rate={rate:.0f}/hr', flush=True)

        elapsed = time.time() - t0
        print(f'Done: {completed} completed, {skipped} skipped, {elapsed/60:.1f}min', flush=True)

asyncio.run(main())
"

echo "=== Generation complete ==="
