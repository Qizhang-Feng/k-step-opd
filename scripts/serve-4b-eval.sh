#!/bin/bash
# Serve 4B full FT checkpoint for quick eval
set -ex
pkill -9 -f sglang 2>/dev/null || true
sleep 2

python -m sglang.launch_server \
    --model-path /root/.cache/huggingface/sft-100k-merged \
    --host 0.0.0.0 --port 30000 \
    --dp-size 8 --tp 1 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 4096 \
    > /tmp/sglang_eval.log 2>&1 &

echo "Waiting for server..."
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:30000/health_generate > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        exit 0
    fi
    sleep 1
done
echo "FAILED"
tail -20 /tmp/sglang_eval.log
exit 1
