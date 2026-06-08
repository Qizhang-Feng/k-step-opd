import pyarrow.parquet as pq
import os

DATA_DIR = '/workspace/data/OpenThoughts3-1.2M/data'
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])

found = 0
for f in files:
    if found >= 2:
        break
    table = pq.read_table(os.path.join(DATA_DIR, f))
    rows = table.to_pylist()
    for row in rows:
        if found >= 2:
            break
        if row.get('domain') != 'math':
            continue
        convs = row.get('conversations', [])
        if len(convs) < 2:
            continue
        response = convs[1].get('value', '')
        if '</think>' in response:
            continue  # skip ones WITH </think>
        if not response:
            continue
        found += 1
        prompt = convs[0].get('value', '')
        print(f'=== No </think> sample {found} ===')
        print(f'Source: {row.get("source")}')
        print(f'Prompt ({len(prompt)} chars): {prompt[:200]}')
        print(f'Response ({len(response)} chars, {len(response.split())} words):')
        print(f'  Start: {response[:300]}')
        print(f'  End: {response[-200:]}')
        print(f'  Has <think>: {"<think>" in response}')
        print(f'  Has boxed: {"\\\\boxed" in response}')
        print()
