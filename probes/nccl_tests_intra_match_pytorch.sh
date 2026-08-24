#!/bin/bash
# =============================================================================
# Stock nccl-tests with sampling and rank layout close to the PyTorch P2P probe.
# =============================================================================
#
# Differences from nccl_tests_intra_pairs.sh:
#   - two MPI processes, one process per selected GPU;
#   - one 64 MiB timed operation per reported out-of-place sample (-n 1);
#   - ten printed cycles per pair on one warm communicator (-N 10);
#   - slowest-rank timing is reported (-a 3);
#   - all four GPUs remain allocated exclusively while one pair communicates.
#
# This removes the 20-iteration averaging and matches the PyTorch process
# layout more closely.  One difference remains: sendrecv_perf issues grouped,
# simultaneous send+recv operations; the PyTorch probe uses a serial ping-pong.
#
# RUN: NCCL_BENCH_NODELIST=nidX ./nccl_tests_intra_match_pytorch.sh
#
#SBATCH --job-name=nccl-intra-match
#SBATCH --time=00:15:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --account=infra01
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

if [ -z "${SLURM_JOB_ID:-}" ]; then
    test -n "${NCCL_BENCH_NODELIST:-}" || {
        echo "set NCCL_BENCH_NODELIST=nidX (single node to test)" >&2
        exit 2
    }
    exec sbatch \
        --reservation="${NCCL_BENCH_RESERVATION:-SD-69241-apertus-1-5-0}" \
        --nodelist="$NCCL_BENCH_NODELIST" \
        "$0"
fi

set -euo pipefail
ulimit -c 0

ROOT=/iopsstor/scratch/cscs/mvasilev
RUN_ENV=${NCCL_BENCH_ENV:-$ROOT/vllm-play/megatron-v2601-multinode.toml}
SAMPLES=${NCCL_MATCH_SAMPLES:-10}
WARMUP=${NCCL_MATCH_WARMUP:-10}
SIZE=${NCCL_MATCH_SIZE:-64M}
GPUS_PER_NODE=4
CPUS_PER_TASK=36
WORK=$ROOT/tmp/nccl-intra-match-${SLURM_JOB_ID}

if [ "${SLURM_JOB_NUM_NODES}" -ne 1 ]; then
    echo "matched intra-pair test requires exactly 1 node" >&2
    exit 2
fi
if ! [[ "$SAMPLES" =~ ^[1-9][0-9]*$ && "$WARMUP" =~ ^[1-9][0-9]*$ ]]; then
    echo "NCCL_MATCH_SAMPLES and NCCL_MATCH_WARMUP must be positive integers" >&2
    exit 2
fi

test -f "$RUN_ENV"
mkdir -p "$WORK"
export SAMPLES WARMUP SIZE
export CUDA_CACHE_DISABLE=1
export SLURM_NETWORK=${SLURM_NETWORK:-disable_rdzv_get}

echo "=== node: $SLURM_JOB_NODELIST  EDF: $RUN_ENV ==="
echo "=== 2 MPI ranks, 1 GPU/rank, size: $SIZE, samples: $SAMPLES, iters/sample: 1 ==="
echo "=== each PAIR_SAMPLES line below uses the out-of-place algbw column ==="

for pair in "0 1" "0 2" "0 3" "1 2" "1 3" "2 3"; do
    read -r ga gb <<< "$pair"
    pair_log=$WORK/g${ga}-g${gb}.log
    echo ""
    echo "##### pair g${ga}-g${gb}: $SAMPLES independent printed samples"

    srun -ul --nodes=1 --ntasks=2 --ntasks-per-node=2 \
        --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_TASK" \
        --gpu-bind=none --mpi=pmix --network="$SLURM_NETWORK" \
        --environment="$RUN_ENV" bash -c '
        set -euo pipefail
        ga=$1
        gb=$2
        # nccl-tests validates the number of visible devices against the two
        # local MPI ranks before assigning one GPU to each rank.  Give both
        # ranks the same ordered pair; local rank 0 selects ga and rank 1 gb.
        export CUDA_VISIBLE_DEVICES=$ga,$gb
        exec sendrecv_perf_mpi \
            -b "$SIZE" -e "$SIZE" -n 1 -w "$WARMUP" -N "$SAMPLES" \
            -a 3 -g 1 -c 0
    ' _ "$ga" "$gb" | tee "$pair_log"

    count=$(awk '$2 == 67108864 {n++} END {print n+0}' "$pair_log")
    if [ "$SIZE" = 64M ] && [ "$count" -ne "$SAMPLES" ]; then
        echo "expected $SAMPLES rows for g${ga}-g${gb}, found $count" >&2
        exit 3
    fi
    awk -v pair="g${ga}-g${gb}" '$2 == 67108864 {
        values = values (values ? " " : "") sprintf("%.2f", $8)
    } END {
        print "PAIR_SAMPLES " pair " out-of-place GB/s: " values
    }' "$pair_log"
done

echo ""
echo "NCCL_TESTS_INTRA_MATCH_COMPLETE $SLURM_JOB_NODELIST"
