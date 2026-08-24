#!/bin/bash
# =============================================================================
# Official NVIDIA nccl-tests, single node: sendrecv_perf on each NVLink pair.
# =============================================================================
#
# Standard-tool counterpart of nccl_p2p_matrix_rep.sh, intra-node only: for
# each of the 6 GPU pairs, run the stock sendrecv_perf (NGC image,
# /usr/local/bin) with only that pair visible, so one NVLink link is exercised
# at a time.  The whole pair loop repeats PASSES times, so retests of the same
# pair are separated in time -- that is what gives an intermittent link fault
# a chance to land inside a measurement window.  nccl-tests prints per-size
# 20-iter averages, so compare the same (pair, size) cell across passes: a
# clean node repeats within ~1-2%, an episode shows as one pass 20%+ off.
#
# RUN:  NCCL_BENCH_NODELIST=nidX ./nccl_tests_intra_pairs.sh
#
# This probe intentionally uses one Slurm task with all four GPUs visible:
# sendrecv_perf -g 2 owns both GPUs in each tested pair.  The task still uses
# the Spellbook launch plumbing (explicit resources, PMIx, network, EDF).
#
#SBATCH --job-name=nccl-tests-intra-pairs
#SBATCH --time=00:15:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
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
export PASSES=${NCCL_BENCH_PASSES:-20}
export SIZES="-b 8M -e 256M -f 2"
export ITERS="-n 20 -w 5"
GPUS_PER_NODE=4
CPUS_PER_TASK=72
export CUDA_CACHE_DISABLE=1
export SLURM_NETWORK=${SLURM_NETWORK:-disable_rdzv_get}

if [ "$SLURM_JOB_NUM_NODES" -ne 1 ]; then
    echo "intra-pair test requires exactly 1 node (got ${SLURM_JOB_NUM_NODES})" >&2
    exit 2
fi

test -f "$RUN_ENV"
echo "=== node: $SLURM_JOB_NODELIST  EDF: $RUN_ENV  passes: $PASSES ==="

# One srun step, one container start; the pair loop runs inside it.
srun -ul --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_TASK" \
    --mpi=pmix --network="$SLURM_NETWORK" --environment="$RUN_ENV" bash -c '
    set -euo pipefail
    BIN=$(command -v sendrecv_perf || command -v sendrecv_perf_mpi) || {
        echo "no sendrecv_perf binary in image" >&2; exit 3; }
    echo "=== nvlink status ==="
    nvidia-smi nvlink -s || true
    echo "=== nvlink error counters BEFORE ==="
    nvidia-smi nvlink -e || true
    for pass in $(seq 1 "$PASSES"); do
        for pair in 0,1 0,2 0,3 1,2 1,3 2,3; do
            echo ""
            echo "##### pass $pass/$PASSES  pair g${pair/,/-g}  ${BIN##*/}"
            CUDA_VISIBLE_DEVICES=$pair "$BIN" $SIZES $ITERS -g 2
        done
    done
    echo "=== nvlink error counters AFTER ==="
    nvidia-smi nvlink -e || true
'

echo ""
echo "NCCL_TESTS_INTRA_COMPLETE $SLURM_JOB_NODELIST"
