#!/bin/bash
# =============================================================================
# All-reduce bandwidth/latency sweep between an exact pair of nodes.
# =============================================================================
#
# Companion to check_nccl_transport.sh (same self-submitting pattern, same
# baked-image runtime as the 22B vLLM eval leg), written for the
# TIMING_PLACEMENT_STUDY nid007195 investigation: the pinned pair
# nid[007133,007195] scored 506s eval vs the 404-410s band. This measures
# whether the fabric between a given pair is the problem.
#
# Two phases:
#   A: 8 ranks, 2 nodes x 4 GPU  - the eval topology (NVLink + network mixed)
#   B: 2 ranks, 1 GPU per node   - pure cross-node path, sharpest fabric probe
# plus one GPU clock/temp/power line per node (throttling check, host-side).
#
# Reported per size: max-across-ranks time, algbw, busbw (= algbw*2(n-1)/n,
# nccl-tests convention), so numbers are comparable to all_reduce_perf.
#
# RUN:  NCCL_BENCH_NODELIST=nid005398,nid005424 ./nccl_allreduce_bench.sh
#       (submits itself; log lands as nccl-arbench-<jobid>.log in cwd)
# Optional: NCCL_BENCH_ENV=<edf.toml>  NCCL_BENCH_PYTHON=<python>
#
#SBATCH --job-name=nccl-arbench
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
        "$0"
fi

set -u

ROOT=/iopsstor/scratch/cscs/mvasilev
# The baked eval image: the exact NCCL (2.30.7) + plugin the slow run used.
RUN_ENV=${NCCL_BENCH_ENV:-$ROOT/vllm-play/vllm-v0260-eval-multinode.toml}
PROBE_PYTHON=${NCCL_BENCH_PYTHON:-python3}
WORK=$ROOT/tmp/nccl-arbench-${SLURM_JOB_ID}

test -f "$RUN_ENV"
mkdir -p "$WORK"
echo "=== nodes: $SLURM_JOB_NODELIST  EDF: $RUN_ENV ==="

echo "=== per-node GPU clocks/temp/power (host nvidia-smi) ==="
srun -ul --ntasks=2 --ntasks-per-node=1 bash -c \
    'echo "$(hostname): $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader | paste -sd" | ")"'

cat > "$WORK/bench.py" <<'PYEOF'
import os
import torch
import torch.distributed as dist

os.environ["RANK"] = os.environ["SLURM_PROCID"]
os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
local_rank = int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()

SIZES = [  # bytes, float32 tensors; iters scaled down as size grows
    (1 << 10, 100), (32 << 10, 100), (1 << 20, 50),
    (16 << 20, 20), (128 << 20, 10), (1 << 30, 5),
]
if rank == 0:
    print(f"world={world}  {torch.cuda.get_device_name(0)}  "
          f"nccl={'.'.join(map(str, torch.cuda.nccl.version()))}", flush=True)
    print(f"{'bytes':>12} {'iters':>5} {'time_ms':>9} {'algbw_GB/s':>10} "
          f"{'busbw_GB/s':>10}", flush=True)

for nbytes, iters in SIZES:
    x = torch.ones(nbytes // 4, dtype=torch.float32, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dist.all_reduce(x)
    end.record()
    torch.cuda.synchronize()
    ms = torch.tensor([start.elapsed_time(end) / iters], device="cuda")
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)  # slowest rank is the truth
    if rank == 0:
        t = ms.item() / 1e3
        algbw = nbytes / t / 1e9
        busbw = algbw * 2 * (world - 1) / world
        print(f"{nbytes:>12} {iters:>5} {ms.item():>9.3f} {algbw:>10.2f} "
              f"{busbw:>10.2f}", flush=True)

dist.barrier()
dist.destroy_process_group()
PYEOF

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29513

echo
echo "=== PHASE A: 8 ranks (2 nodes x 4 GPU, eval topology) ==="
srun -ul --ntasks-per-node=4 --environment="$RUN_ENV" \
    env MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    "$PROBE_PYTHON" "$WORK/bench.py"

echo
echo "=== PHASE B: 2 ranks (1 GPU per node, pure cross-node) ==="
# --ntasks=2 is required: srun inside the batch inherits SLURM_NTASKS=8 and
# IGNORES a bare --ntasks-per-node=1 (job 3141116/17 phase B ran 8 ranks).
srun -ul --ntasks=2 --ntasks-per-node=1 --environment="$RUN_ENV" \
    env MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=29514 \
    "$PROBE_PYTHON" "$WORK/bench.py"

echo
echo "DONE: $SLURM_JOB_NODELIST"
