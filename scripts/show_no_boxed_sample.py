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
        prompt = convs[0].get('value', '')
        response = convs[1].get('value', '')
        if '\\boxed' in response:
            continue
        if not response:
            continue
        found += 1
        print(f'=== Sample {found} (no boxed) ===')
        print(f'Source: {row.get("source")}')
        print(f'Prompt: {prompt[:300]}')
        print()
        print(f'Response start: {response[:400]}')
        print()
        print(f'Response end: {response[-200:]}')
        print(f'Len: {len(response)} chars, {len(response.split())} words')
        print()
