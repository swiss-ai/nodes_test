#!/bin/bash
# =============================================================================
# Official NVIDIA nccl-tests, 3 modes: intra node A, intra node B, cross 8-GPU.
# =============================================================================
#
# Same 3-group design as nccl_collective_rep.sh, but using the stock nccl-tests
# binaries shipped in the NGC PyTorch 26.01 image (/usr/local/bin/*_perf_mpi),
# one MPI rank per GPU, launched with srun --mpi=pmix:
#   MODE A: 4 GPUs inside the first node   (acts as the intra-node control)
#   MODE B: 4 GPUs inside the second node
#   MODE C: all 8 GPUs across both nodes   (needs the aws_ofi_nccl hook)
# Collectives: all_reduce, reduce_scatter, scatter.  Sizes 8M..256M (x2).
# nccl-tests only reports per-size AVERAGES, so episodic faults get diluted;
# PASSES full repetitions give an episode more than one chance to land.
#
# RUN:  NCCL_BENCH_NODELIST=nidA,nidB ./nccl_tests_3mode.sh
#
# Allocation follows Spellbook's slurm_tasks.sh.j2 convention (one Slurm task
# per GPU).  Separate srun steps are retained intentionally for intra A,
# intra B, and cross-node measurements.
#
#SBATCH --job-name=nccl-tests-3mode
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
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
PASSES=${NCCL_BENCH_PASSES:-3}
SIZES="-b 8M -e 256M -f 2"
ITERS="-n 20 -w 5"
BINS=${NCCL_BENCH_BINS:-"all_reduce_perf_mpi reduce_scatter_perf_mpi scatter_perf_mpi"}
GPUS_PER_NODE=4
TOTAL_TASKS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))

export GPUS_PER_NODE
export HOSTNAMES="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=${MASTER_PORT:-6800}
export WORLD_SIZE=$TOTAL_TASKS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_CACHE_DISABLE=1
export SLURM_NETWORK=${SLURM_NETWORK:-disable_rdzv_get}

test -f "$RUN_ENV"
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
A=${NODES[0]}
B=${NODES[1]:-}   # 1-node allocation (sbatch --nodes=1): intra mode A only
echo "=== nodes: A=$A B=${B:-none}  EDF: $RUN_ENV  passes: $PASSES ==="

for pass in $(seq 1 "$PASSES"); do
    for bin in $BINS; do
        echo ""
        echo "##### pass $pass/$PASSES  MODE A  intra $A  $bin"
        srun -ul -N1 -n4 -w "$A" --mpi=pmix --network="$SLURM_NETWORK" \
            --environment="$RUN_ENV" \
            "$bin" $SIZES $ITERS -g 1
        test -n "$B" || continue
        echo ""
        echo "##### pass $pass/$PASSES  MODE B  intra $B  $bin"
        srun -ul -N1 -n4 -w "$B" --mpi=pmix --network="$SLURM_NETWORK" \
            --environment="$RUN_ENV" \
            "$bin" $SIZES $ITERS -g 1
        echo ""
        echo "##### pass $pass/$PASSES  MODE C  cross $A+$B  $bin"
        srun -ul -N2 -n8 --ntasks-per-node="$GPUS_PER_NODE" --mpi=pmix \
            --network="$SLURM_NETWORK" --environment="$RUN_ENV" \
            "$bin" $SIZES $ITERS -g 1
    done
done

echo ""
echo "NCCL_TESTS_COMPLETE $SLURM_JOB_NODELIST"
