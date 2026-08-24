#!/bin/bash
# =============================================================================
# Official NVIDIA nccl-tests, cross-node: sendrecv_perf on each GPU pair.
# =============================================================================
#
# Standard-tool counterpart of the cross half of nccl_p2p_matrix_rep.sh:
# for each of the 16 (GPU on node A) x (GPU on node B) pairs, run the stock
# sendrecv_perf with 2 MPI ranks (one per node, one GPU each), so exactly one
# network path is exercised at a time.  The pair loop repeats PASSES times so
# retests of the same pair are separated in time -- an intermittent ingress
# fault either lands inside a 20-iter window (big dip in that average) or
# misses it; many short windows beat one long diluted one.  First size row
# of each run is startup-noisy -- compare from the second size up.
#
# RUN:  NCCL_BENCH_NODELIST=nidA,nidB ./nccl_tests_cross_pairs.sh
#
#SBATCH --job-name=nccl-tests-cross-pairs
#SBATCH --time=00:45:00
#SBATCH --partition=normal
#SBATCH --nodes=2
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
        echo "set NCCL_BENCH_NODELIST=nidA,nidB (exact pair to test)" >&2
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
PASSES=${NCCL_BENCH_PASSES:-8}
SIZES=${NCCL_BENCH_SIZES:-"-b 16M -e 256M -f 4"}
ITERS="-n 20 -w 5"
GPUS_PER_NODE=4
CPUS_PER_TASK=72
export SLURM_NETWORK=${SLURM_NETWORK:-disable_rdzv_get}

test -f "$RUN_ENV"
if [ "$SLURM_JOB_NUM_NODES" -ne 2 ]; then
    echo "cross-pair test requires exactly 2 nodes (got ${SLURM_JOB_NUM_NODES})" >&2
    exit 2
fi
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "=== nodes: A=${NODES[0]} B=${NODES[1]}  EDF: $RUN_ENV  passes: $PASSES ==="

for pass in $(seq 1 "$PASSES"); do
    for ga in 0 1 2 3; do
        for gb in 0 1 2 3; do
            echo ""
            echo "##### pass $pass/$PASSES  cross ${NODES[0]}:g$ga <-> ${NODES[1]}:g$gb  sendrecv"
            srun -ul --nodes=2 --ntasks=2 --ntasks-per-node=1 \
                --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_TASK" \
                --gpu-bind=none --mpi=pmix --network="$SLURM_NETWORK" \
                --environment="$RUN_ENV" \
                bash -c "export CUDA_VISIBLE_DEVICES=\$([ \"\$SLURM_NODEID\" = 0 ] && echo $ga || echo $gb); exec sendrecv_perf_mpi $SIZES $ITERS -g 1"
        done
    done
done

echo ""
echo "NCCL_TESTS_CROSS_COMPLETE $SLURM_JOB_NODELIST"
