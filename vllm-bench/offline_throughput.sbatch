#!/bin/bash
# OFFLINE throughput benchmark: drives the vLLM engine directly in-process
# (no HTTP server). All prompts are enqueued at once; measures the maximum
# sustainable batch throughput in tokens/s. See README.md.
#
# Topology note: this job uses TP=4 (not the 1/1/4/4 DP shape of the online
# job) because single-process offline vLLM refuses data_parallel_size > 1 —
# offline DP requires the torchrun external-launcher machinery that the
# custom harness in vllm-play/benchmark_apertus2_vllm.py implements. For a
# DP=4/EP=4 throughput number, use the online job with REQUEST_RATE=inf
# (= offline + HTTP overhead) or the vllm-play harness.
#
# Submit:            sbatch offline_throughput.sbatch
# With knobs:        NUM_PROMPTS=200 TP=4 sbatch offline_throughput.sbatch
#
#SBATCH --job-name=vllm-bench-offline
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
MODEL=/iopsstor/scratch/cscs/mvasilev/hf-export/scaling-ladder/22b-moe-256e-latent-swa15-nope-muonmd-lr2.421e-3-mlr2.421e-2-latmoe-e256-top8-swa15-w512-nope-fp8-22b-moe-256e-swa15-nope/iter_0036689
DATASET=$BENCH_DIR/ShareGPT_V3_unfiltered_cleaned_split.json

NUM_PROMPTS=${NUM_PROMPTS:-500}   # how many ShareGPT conversations to run
TP=${TP:-4}                       # tensor-parallel size (4 GPUs on one node)

[[ -f "$DATASET" ]] || { echo "Dataset missing — run ./download_sharegpt.sh first."; exit 1; }

# Everything is local (model dir + dataset json) — forbid hub access so a
# transient network issue can never change what the job does.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Per-job module cache: trust_remote_code copies modeling_apertus2.py here;
# isolating per job avoids concurrent jobs racing on the same module dir.
export HF_MODULES_CACHE=$BENCH_DIR/hf_home/modules/$SLURM_JOB_ID
# torch.compile / cudagraph artifacts land here and are reused by later jobs
# (first run pays a few minutes of compile, later runs start much faster).
export VLLM_CACHE_ROOT=$BENCH_DIR/vllm-cache
# Pin the MoE kernel path to triton (deterministic; matches all prior runs).
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export TOKENIZERS_PARALLELISM=false

# srun starts ONE task inside the container; vLLM itself forks one worker
# per GPU. Exported env vars propagate into the container.
srun -ul --environment="$BENCH_DIR/vllm-bench.toml" \
    vllm bench throughput \
        --model "$MODEL" \
        --trust-remote-code \
        --dtype bfloat16 \
        --tensor-parallel-size "$TP" \
        --enable-expert-parallel \
        --max-model-len 8192 \
        --dataset-name sharegpt \
        --dataset-path "$DATASET" \
        --num-prompts "$NUM_PROMPTS" \
        --output-json "$BENCH_DIR/results/offline-tp${TP}-n${NUM_PROMPTS}-job${SLURM_JOB_ID}.json"
