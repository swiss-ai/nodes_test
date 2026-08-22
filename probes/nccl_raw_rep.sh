#!/bin/bash
# =============================================================================
# Raw NCCL collective probe -- no PyTorch / Python / MPI in the loop.
# =============================================================================
#
# Runs nccl_raw_collrep.cu (see its header) inside the UNTOUCHED standard
# vLLM image (node-health/qwen3-bench/vllm-standard-v0271.toml), linked against the very
# libnccl 2.30.7 the vLLM benchmarks load, with the same aws-ofi/Slingshot
# hook.  Compiles in-container on first use or when the source is newer.
#
# RUN:  NCCL_BENCH_NODELIST=nidA,nidB ./nccl_raw_rep.sh
#       (NCCL_RAW_SWEEPS=40 NCCL_RAW_SIZE_MB=64 to override)
#
#SBATCH --job-name=nccl-rawrep
#SBATCH --time=00:15:00
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
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
        --export=ALL \
        "$0"
fi

set -u
ROOT=/iopsstor/scratch/cscs/mvasilev
RUN_ENV=$ROOT/node-health/qwen3-bench/vllm-standard-v0271.toml
SRC=$ROOT/node-health/probes/nccl_raw_collrep.cu
BIN=$ROOT/node-health/probes/nccl_raw_collrep
NCCL_PIP=/usr/local/lib/python3.12/dist-packages/nvidia/nccl

echo "nodes : $SLURM_JOB_NODELIST"
echo "sweeps: ${NCCL_RAW_SWEEPS:-20} x ${NCCL_RAW_SIZE_MB:-64} MiB"

# compile inside the container if missing/stale (single task, first node)
if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
    echo "compiling $BIN in-container..."
    srun -N1 -n1 --environment="$RUN_ENV" bash -c "
        /usr/local/cuda/bin/nvcc -O2 -o '$BIN' '$SRC' \
            -I'$NCCL_PIP/include' -L'$NCCL_PIP/lib' \
            -l:libnccl.so.2 -Xlinker -rpath='$NCCL_PIP/lib'" || exit 1
    echo "compiled OK"
fi

export NCCL_RAW_ID_FILE=$ROOT/tmp/nccl-raw-$SLURM_JOB_ID.id
rm -f "$NCCL_RAW_ID_FILE"

srun -ul --ntasks=8 --ntasks-per-node=4 --environment="$RUN_ENV" "$BIN"
RC=$?
rm -f "$NCCL_RAW_ID_FILE"
echo "RAW_REP_COMPLETE rc=$RC"
echo "DONE: $SLURM_JOB_NODELIST"
exit $RC
