#!/bin/bash
# Production-shaped standard benchmark: vllm serve + vllm bench serve on
# 4 nodes -- TP=4 inside each node (NVLink), DP=4 across nodes, and with
# --enable-expert-parallel the EP group is DP*TP=16 wide, so MoE expert
# dispatch (allgather/reduce-scatter) crosses Slingshot every layer: the
# Apertus2 production comm pattern at 16-GPU scale.  Everything is stock
# vLLM tooling from the untouched v0.27.1 image; the bench workload is the
# built-in seeded random dataset (no downloads, identical across jobs).
#
# Same multi-node deadlock guards as qwen3_standard_eval.sbatch:
# VLLM_ALLREDUCE_USE_SYMM_MEM=0 (EDF) + --disable-custom-all-reduce.
#
# Usage:  LABEL=<tag> sbatch --nodelist=nidA,nidB,nidC,nidD qwen3_standard_bench.sbatch
#         NUM_PROMPTS=300 for a prewarm/smoke run (also warms the compile cache).
#         Layout is overridable: TP_SIZE=1 DP_SIZE=8 sbatch --nodes=2 ... gives the
#         original Apertus2 eval layout (TP=1 x DP=8 x EP=8, ALL comm cross-node).

#SBATCH --job-name=qwen3-std-bench
#SBATCH --time=01:30:00
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --account=infra01
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -euo pipefail
: "${LABEL:?LABEL is required (names the output dir)}"

ROOT=/iopsstor/scratch/cscs/$USER
RUN_ENV=$ROOT/node-health/qwen3-bench/vllm-standard-v0271.toml
VENV=$ROOT/vllm-play/lmeval-std-venv
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
NUM_PROMPTS=${NUM_PROMPTS:-3000}
BENCH_PASSES=${BENCH_PASSES:-2}
IN_LEN=${IN_LEN:-1024}
OUT_LEN=${OUT_LEN:-512}
# OUT_LEN=1 = prefill-only benchmark (the lm-eval loglikelihood shape);
# TPOT/ITL are undefined with a single output token, so drop them there.
if [ "$OUT_LEN" -le 1 ]; then PCT_METRICS=ttft,e2el; else PCT_METRICS=ttft,tpot,itl,e2el; fi
OUT=$ROOT/node-health/results/$LABEL-$SLURM_JOB_ID
WORK=$ROOT/tmp/qwen3std-$SLURM_JOB_ID
NGPUS=$((SLURM_NNODES * 4))
TP_SIZE=${TP_SIZE:-4}
DP_SIZE=${DP_SIZE:-$SLURM_NNODES}
if [ $((TP_SIZE * DP_SIZE)) -ne "$NGPUS" ] || [ $((DP_SIZE % SLURM_NNODES)) -ne 0 ]; then
    echo "FATAL: TP_SIZE=$TP_SIZE x DP_SIZE=$DP_SIZE does not tile $NGPUS GPUs on $SLURM_NNODES nodes" >&2
    exit 1
fi
DP_LOCAL=$((DP_SIZE / SLURM_NNODES))

test -f "$RUN_ENV"
mkdir -p "$OUT" "$WORK"

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
RAY_PORT=$((25000 + SLURM_JOB_ID % 10000))
VLLM_PORT=$((18000 + SLURM_JOB_ID % 10000))

echo "START TIME : $(date --iso-8601=seconds)"
echo "nodes      : $SLURM_JOB_NODELIST (head $HEAD, ray $RAY_PORT, serve $VLLM_PORT)"
echo "model      : $MODEL  TP=$TP_SIZE DP=$DP_SIZE EP=$NGPUS via ray ($NGPUS GPUs, $DP_LOCAL replicas/node)"
echo "bench      : random dataset seed=42, in=$IN_LEN out=$OUT_LEN, $NUM_PROMPTS prompts x $BENCH_PASSES passes"
echo "output     : $OUT"

cat > "$WORK/inner.sh" <<EOF
#!/bin/bash
set -u
export HEAD=$HEAD RAY_PORT=$RAY_PORT VLLM_PORT=$VLLM_PORT OUT=$OUT VENV=$VENV
export MODEL=$MODEL NUM_PROMPTS=$NUM_PROMPTS BENCH_PASSES=$BENCH_PASSES NGPUS=$NGPUS
export TP_SIZE=$TP_SIZE DP_SIZE=$DP_SIZE DP_LOCAL=$DP_LOCAL
export IN_LEN=$IN_LEN OUT_LEN=$OUT_LEN PCT_METRICS=$PCT_METRICS
EOF
cat >> "$WORK/inner.sh" <<'EOF'
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_CACHE_ROOT=/iopsstor/scratch/cscs/mvasilev/vllm-play/vllm_cache_std
DONE=$OUT/.ray-done
RAY=$(command -v ray || true); [ -n "$RAY" ] || RAY=$VENV/bin/ray
# the vllm CLI must run under the venv python: the image's /usr/local/bin/vllm
# uses system python, which cannot import the venv-installed ray
VLLM="$VENV/bin/python -m vllm.entrypoints.cli.main"

if [ "$SLURM_PROCID" -eq 0 ]; then
    "$RAY" start --head --port="$RAY_PORT" --num-gpus=4 \
        --disable-usage-stats --include-dashboard=false || exit 1
    for i in $(seq 120); do
        GPUS=$("$VENV/bin/python" - <<'PY'
import ray
ray.init(address="auto", logging_level="error")
print(int(ray.cluster_resources().get("GPU", 0)))
PY
        )
        [ "${GPUS:-0}" -ge "$NGPUS" ] && break
        sleep 5
    done
    if [ "${GPUS:-0}" -lt "$NGPUS" ]; then
        echo "FATAL: ray cluster never reached $NGPUS GPUs (got ${GPUS:-0})" >&2
        touch "$DONE"; "$RAY" stop; exit 1
    fi
    echo "ray cluster up: $GPUS GPUs"

    # --api-server-count 1: the default spawns several frontend processes,
    # doubling exposure to the ZMQ port-bind race that killed job 3145286
    # ("Address already in use" -- a Ray-held ephemeral port won the race).
    # Bind races are transient, so one retry of the whole serve startup.
    UP=""
    for attempt in 1 2; do
        $VLLM serve "$MODEL" \
            --tensor-parallel-size "$TP_SIZE" \
            --data-parallel-size "$DP_SIZE" \
            --data-parallel-size-local "$DP_LOCAL" \
            --data-parallel-backend ray \
            --enable-expert-parallel \
            --disable-custom-all-reduce \
            --api-server-count 1 \
            --host 127.0.0.1 --port "$VLLM_PORT" \
            >> "$OUT/serve.log" 2>&1 &
        SERVE_PID=$!
        for i in $(seq 180); do    # up to 30 min: first run compiles TP=4 graphs
            if ! kill -0 "$SERVE_PID" 2>/dev/null; then
                echo "serve exited early (attempt $attempt); serve.log tail:" >&2
                tail -15 "$OUT/serve.log" >&2
                break
            fi
            curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null && { UP=1; break; }
            sleep 10
        done
        [ -n "$UP" ] && break
        kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null
        sleep 15
    done
    [ -n "$UP" ] || { echo "FATAL: serve never became healthy after 2 attempts" >&2; touch "$DONE"; "$RAY" stop; exit 1; }
    echo "SERVE HEALTHY after startup (attempt $attempt)"

    RC=0
    for pass in $(seq "$BENCH_PASSES"); do
        echo "===== BENCH PASS $pass ====="
        T0=$(date +%s)
        $VLLM bench serve \
            --model "$MODEL" \
            --host 127.0.0.1 --port "$VLLM_PORT" \
            --dataset-name random \
            --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --seed 42 \
            --max-concurrency 512 \
            --request-rate inf \
            --percentile-metrics "$PCT_METRICS" \
            --save-result --result-dir "$OUT" --result-filename "bench_pass${pass}.json" \
            || RC=1
        echo "BENCH_PASS_${pass}_WALL_SECONDS $(( $(date +%s) - T0 ))"
    done
    kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null
    touch "$DONE"
    "$RAY" stop
    exit $RC
else
    for i in $(seq 150); do
        (exec 3<>"/dev/tcp/$HEAD/$RAY_PORT") 2>/dev/null && break
        sleep 2
    done
    "$RAY" start --address="$HEAD:$RAY_PORT" --num-gpus=4 --disable-usage-stats || exit 1
    while [ ! -f "$DONE" ]; do sleep 5; done
    "$RAY" stop
    exit 0
fi
EOF
chmod +x "$WORK/inner.sh"

set +e
srun -ul --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --environment="$RUN_ENV" "$WORK/inner.sh"
RC=$?
set -e
rm -f "$OUT/.ray-done"
echo "END TIME   : $(date --iso-8601=seconds)  rc=$RC"
exit $RC
