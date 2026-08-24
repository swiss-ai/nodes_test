#!/bin/bash
# Runs INSIDE the container (launched by online_serving.sbatch).
# Phase 1: start the OpenAI-compatible server in the background.
# Phase 2: wait until /health answers (weight load + torch.compile).
# Phase 3: run the benchmark client against it.
# Phase 4: shut the server down.
set -euo pipefail

echo "=== [1/3] Starting vLLM server (TP=$TP, DP=$DP, expert-parallel EP=$((TP * DP))) ==="
echo "    server log: $SERVER_LOG"
# With --data-parallel-size N the front-end spawns N engine-core processes
# and load-balances requests across them (single endpoint). The expert
# layers form one EP group of size TP*DP; a DP coordinator keeps the ranks
# in lockstep (idle ranks run dummy passes so the MoE collective never
# blocks). This is the 1/1/4/4 shape from the benchmark matrix.
vllm serve "$MODEL" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --enable-expert-parallel \
    --max-model-len 8192 \
    --host 127.0.0.1 \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# First-ever start takes ~10-20 min (42 GB weight load + torch.compile +
# CUDA graph capture); later starts reuse the compile cache and are faster.
echo "=== [2/3] Waiting for server to become ready ==="
READY=0
for i in $(seq 1 240); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "!!! Server process died. Last 50 lines of $SERVER_LOG:"
        tail -n 50 "$SERVER_LOG"
        exit 1
    fi
    if python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$PORT/health', timeout=3)" 2>/dev/null; then
        READY=1
        echo "Server ready after ~$((i * 5))s."
        break
    fi
    sleep 5
done
if [[ "$READY" != 1 ]]; then
    echo "!!! Server not ready after 20 min. Last 50 lines of $SERVER_LOG:"
    tail -n 50 "$SERVER_LOG"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

echo "=== [3/3] Running benchmark client ==="
vllm bench serve \
    --backend vllm \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model "$MODEL" \
    --trust-remote-code \
    --dataset-name sharegpt \
    --dataset-path "$DATASET" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    ${MAX_CONCURRENCY:+--max-concurrency "$MAX_CONCURRENCY"} \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "online-tp${TP}-dp${DP}-rate${REQUEST_RATE}-n${NUM_PROMPTS}-job${SLURM_JOB_ID}.json"

echo "=== Done — shutting down server ==="
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
