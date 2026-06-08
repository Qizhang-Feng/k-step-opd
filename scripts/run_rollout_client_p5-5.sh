#!/bin/bash
# Rollout client only - reuses existing sglang server on port 30000
# For p5-5, shard 0/1 (all prompts since only p5-5 is running)
set -e

cd /workspace/k-step-opd

OUTPUT_DIR=/root/.cache/huggingface/rollouts-8b-sft
PROMPTS=/workspace/data/sft_math_extra_100k_v2.jsonl
PORT=30000
MAX_TOKENS=16384
TEMPERATURE=0.7
TOP_P=0.9
CONCURRENCY=128

mkdir -p $OUTPUT_DIR

python3 -c "
import asyncio, aiohttp, json, time, os

PORT = $PORT
MAX_TOKENS = $MAX_TOKENS
CONCURRENCY = $CONCURRENCY
OUTPUT_DIR = '$OUTPUT_DIR'
PROMPTS_FILE = '$PROMPTS'
TEMPERATURE = $TEMPERATURE
TOP_P = $TOP_P

SERVER_URL = f'http://127.0.0.1:{PORT}'
output_path = os.path.join(OUTPUT_DIR, 'rollouts_all.jsonl')

# Load all prompts
all_data = []
with open(PROMPTS_FILE) as f:
    for line in f:
        d = json.loads(line)
        # messages format: extract user content
        if 'messages' in d:
            p = d['messages'][0]['content']
        else:
            p = d.get('prompt', d.get('text', ''))
            if isinstance(p, list):
                p = p[0].get('content', '') if p else ''
        all_data.append(p)

print(f'Total prompts: {len(all_data)}', flush=True)

# Resume: check existing output
completed_prompts = set()
if os.path.exists(output_path):
    with open(output_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                completed_prompts.add(row['messages'][0]['content'])
            except:
                pass
    print(f'Resuming: {len(completed_prompts)} already done', flush=True)

todo = [p for p in all_data if p not in completed_prompts]
print(f'Remaining: {len(todo)} prompts', flush=True)

if not todo:
    print('All done!')
    exit(0)

async def generate_one(session, sem, prompt):
    text = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n'
    payload = {
        'text': text,
        'sampling_params': {
            'max_new_tokens': MAX_TOKENS,
            'temperature': TEMPERATURE,
            'top_p': TOP_P,
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
                    print(f'Error after 3 retries: {e}', flush=True)
                await asyncio.sleep(5 * (attempt + 1))
    return prompt, ''

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 8)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [generate_one(session, sem, p) for p in todo]

        completed = 0
        skipped = 0
        t0 = time.time()

        with open(output_path, 'a') as fout:
            for fut in asyncio.as_completed(tasks):
                prompt, response = await fut

                if response:
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
                if total % 100 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed * 3600 if elapsed > 0 else 0
                    print(f'[{total}/{len(todo)}] done={completed} skip={skipped} rate={rate:.0f}/hr', flush=True)

        elapsed = time.time() - t0
        print(f'Done: {completed} completed, {skipped} skipped, {elapsed/60:.1f}min', flush=True)

asyncio.run(main())
"

echo "=== Rollout collection done ==="
