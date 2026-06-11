"""Extract per-rollout metrics from R5 ray-job log on p5-3.

Usage (inside k-step-opd container on p5-3):
    ray job logs raysubmit_bVCHR2K7iiiPG3ph 2>&1 > /tmp/r5_logs.txt
    python3 r5_extract.py
"""
import re

with open('/tmp/r5_logs.txt') as f:
    log = f.read()

# Match a rollout line up to the next newline; the dict pretty-prints across
# brackets but stays on a single logical line so this works.
pattern = re.compile(r"data\.py:211 - rollout (\d+): (\{.*\})")
matches = pattern.findall(log)
print(f'total rollouts logged: {len(matches)}')
print()
print('id  | instant_kl | advantages | truncated | resp_len')
print('-' * 70)
indices = [0, 49, 99, 149] + list(range(max(0, len(matches) - 3), len(matches)))
seen = set()
# Print sample body to understand format
print('--- raw body sample (rollout 1, first 500 chars) ---')
print(matches[0][1][:500])
print('--- end sample ---')
print()
for i in indices:
    if i in seen or i >= len(matches):
        continue
    seen.add(i)
    rid_str, body = matches[i]

    def grab(key):
        # Keys are prefixed with `rollout/`
        m = re.search(rf"'rollout/{re.escape(key)}': ([-\d.eE]+)", body)
        return m.group(1) if m else 'NA'

    print(
        f'{int(rid_str):3d} | {grab("opd_reverse_kl"):>10s} | '
        f'{grab("advantages"):>11s} | {grab("truncated"):>9s} | '
        f'{grab("response_lengths"):>8s}'
    )
