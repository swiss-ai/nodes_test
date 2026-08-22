#!/bin/bash
# ONLINE serving benchmark: starts a real `vllm serve` OpenAI-compatible
# HTTP server, then fires ShareGPT requests at it with `vllm bench serve`
# and measures what a user would feel: TTFT / TPOT / ITL / E2E latency
# plus achieved throughput. See README.md.
#
# Default topology: TP=1, DP=4, EP=4 — same 1/1/4/4 shape as the
# megatron-vs-vllm benchmark matrix and the ladder-eval leg (and training's
# expert sharding). Set TP=4 DP=1 to compare against pure tensor parallel.
#
# Submit:            sbatch online_serving.sbatch
# With knobs:        REQUEST_RATE=8 MAX_CONCURRENCY=64 sbatch online_serving.sbatch
#                    TP=4 DP=1 sbatch online_serving.sbatch
#
#SBATCH --job-name=vllm-bench-online
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --output=/iopsstor/scratch/cscs/mvasilev/node-health/vllm-bench/slurm/%x-%j.log

set -euo pipefail

BENCH_DIR=/iopsstor/scratch/cscs/mvasilev/node-health/vllm-bench
export MODEL=/iopsstor/scratch/cscs/mvasilev/hf-export/scaling-ladder/22b-moe-256e-latent-swa15-nope-muonmd-lr2.421e-3-mlr2.421e-2-latmoe-e256-top8-swa15-w512-nope-fp8-22b-moe-256e-swa15-nope/iter_0036689
export DATASET=$BENCH_DIR/ShareGPT_V3_unfiltered_cleaned_split.json

export NUM_PROMPTS=${NUM_PROMPTS:-500}     # how many ShareGPT conversations to send
export TP=${TP:-1}                         # tensor-parallel size (attention/dense)
export DP=${DP:-4}                         # data-parallel size (engine replicas)
export REQUEST_RATE=${REQUEST_RATE:-inf}   # req/s arrival rate; inf = all at once
export MAX_CONCURRENCY=${MAX_CONCURRENCY:-} # cap on in-flight requests; empty = no cap
export PORT=${PORT:-8000}
export SERVER_LOG=$BENCH_DIR/slurm/server-$SLURM_JOB_ID.log
export RESULT_DIR=$BENCH_DIR/results

[[ -f "$DATASET" ]] || { echo "Dataset missing — run ./download_sharegpt.sh first."; exit 1; }
(( TP * DP == 4 )) || { echo "TP*DP must equal 4 (one node, 4 GPUs); got TP=$TP DP=$DP."; exit 1; }

# Same environment discipline as the offline job (explained there).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_MODULES_CACHE=$BENCH_DIR/hf_home/modules/$SLURM_JOB_ID
export VLLM_CACHE_ROOT=$BENCH_DIR/vllm-cache
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export TOKENIZERS_PARALLELISM=false

# The server+client logic needs shell control flow (background server, wait
# for readiness, run client, shut down), so it lives in online_inner.sh and
# runs INSIDE the container. Exported vars propagate through srun.
srun -ul --environment="$BENCH_DIR/vllm-bench.toml" \
    bash "$BENCH_DIR/online_inner.sh"
