import re, json, sys

logfile = sys.argv[1] if len(sys.argv) > 1 else '/workspace/k-step-opd/sft_full_4b.log'

with open(logfile) as f:
    content = f.read()

pattern = r"'loss': '([^']+)', 'grad_norm': '([^']+)', 'learning_rate': '([^']+)', 'token_acc': '([^']+)', 'epoch': '([^']+)', 'global_step/max_steps': '(\d+)/(\d+)'"
matches = re.findall(pattern, content)

data = []
for m in matches:
    data.append({
        'step': int(m[5]),
        'loss': float(m[0]),
        'grad_norm': float(m[1]),
        'lr': float(m[2]),
        'token_acc': float(m[3]),
    })

print(json.dumps(data))
