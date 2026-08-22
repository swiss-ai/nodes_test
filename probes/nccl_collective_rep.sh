#!/bin/bash
# =============================================================================
# Repeated collective benchmark: all-reduce / all-gather / reduce-scatter,
# 3 groups (each node's 4 GPUs intra, all 8 ranks multinode), 20 samples.
# =============================================================================
#
# Production-shaped sibling of nccl_p2p_matrix_rep.sh: instead of pairwise
# ping-pong it measures the collectives real workloads run (vLLM EP =
# allgather + reduce-scatter, training = all-reduce), which stress all links
# of a group CONCURRENTLY with NCCL's real algorithms.  64 MiB payload per
# collective, fresh random buffers each sample, NCCL_BENCH_SWEEPS (default
# 20) samples per (group, collective) in one job.  Time is max-across-group
# ranks.  busbw uses nccl-tests conventions:
#   allreduce      busbw = 2(n-1)/n * S / t         (S = 64 MiB buffer)
#   allgather      busbw =  (n-1)/n * S / t         (S = 64 MiB total output)
#   reducescatter  busbw =  (n-1)/n * S / t         (S = 64 MiB total input)
# Prints mean/std/min per metric + all 20 samples for any metric whose min
# falls below 80% of its median (the intermittency fingerprint).
#
# RUN:  NCCL_BENCH_NODELIST=nidA,nidB ./nccl_collective_rep.sh
#
#SBATCH --job-name=nccl-collrep
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
RUN_ENV=${NCCL_BENCH_ENV:-$ROOT/vllm-play/vllm-v0260-eval-multinode.toml}
PROBE_PYTHON=${NCCL_BENCH_PYTHON:-python3}
SWEEPS=${NCCL_BENCH_SWEEPS:-20}
WORK=$ROOT/tmp/nccl-collrep-${SLURM_JOB_ID}

test -f "$RUN_ENV"
mkdir -p "$WORK"
echo "=== nodes: $SLURM_JOB_NODELIST  EDF: $RUN_ENV  sweeps: $SWEEPS ==="

cat > "$WORK/collrep.py" <<'PYEOF'
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist

os.environ["RANK"] = os.environ["SLURM_PROCID"]
os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
local_rank = int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
SWEEPS = int(os.environ["SWEEPS"])
NBYTES = 64 << 20
ITERS = 5

names = [None] * world
dist.all_gather_object(names, f"{socket.gethostname()}:g{local_rank}")
node_a = names[0].split(":")[0]
node_b = names[4].split(":")[0]
if rank == 0:
    print(f"rank map: {names}", flush=True)

GROUPS = [
    (f"intra {node_a}", list(range(4)), dist.new_group(list(range(4)))),
    (f"intra {node_b}", list(range(4, 8)), dist.new_group(list(range(4, 8)))),
    ("all-node (8)", list(range(8)), dist.new_group(list(range(8)))),
]


def timed(fn, group, ranks):
    torch.cuda.synchronize()
    dist.barrier(group=group)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    ms = torch.tensor([(time.perf_counter() - t0) / ITERS], device="cuda")
    dist.all_reduce(ms, op=dist.ReduceOp.MAX, group=group)  # slowest rank
    return ms.item()


samples = {}  # (group name, collective) -> [busbw GB/s]
for sweep in range(SWEEPS):
    for gname, ranks, g in GROUPS:
        dist.barrier()
        if rank not in ranks:
            dist.barrier()
            continue
        n = len(ranks)
        x = torch.randn(NBYTES // 4, dtype=torch.float32, device="cuda")
        ag_in = torch.randn(NBYTES // 4 // n, dtype=torch.float32, device="cuda")
        ag_out = torch.empty(NBYTES // 4 // n * n, dtype=torch.float32, device="cuda")
        rs_out = torch.empty(NBYTES // 4 // n, dtype=torch.float32, device="cuda")
        warm = 5 if sweep == 0 else 1
        cases = [
            ("allreduce", lambda: dist.all_reduce(x, group=g),
             2 * (n - 1) / n * NBYTES),
            ("allgather", lambda: dist.all_gather_into_tensor(ag_out, ag_in, group=g),
             (n - 1) / n * (NBYTES // n * n)),
            ("reducescatter", lambda: dist.reduce_scatter_tensor(rs_out, x, group=g),
             (n - 1) / n * NBYTES),
        ]
        for cname, fn, wire in cases:
            for _ in range(warm):
                fn()
            t = timed(fn, g, ranks)
            if rank == ranks[0]:
                samples.setdefault((gname, cname), []).append(wire / t / 1e9)
        dist.barrier()
    if rank == 0:
        print(f"sweep {sweep + 1}/{SWEEPS} done", flush=True)

gathered = [None] * world
dist.all_gather_object(gathered, samples)
if rank == 0:
    merged = {}
    for part in gathered:
        merged.update(part)
    print(f"\n{'group':<18} {'collective':<14} {'mean':>8} {'std':>7} "
          f"{'min':>8} {'max':>8}  (busbw GB/s, {SWEEPS} samples)", flush=True)
    flagged = []
    for gname, _, _ in GROUPS:
        for cname in ("allreduce", "allgather", "reducescatter"):
            v = merged[(gname, cname)]
            med = statistics.median(v)
            line = (f"{gname:<18} {cname:<14} {statistics.mean(v):>8.2f} "
                    f"{statistics.stdev(v):>7.2f} {min(v):>8.2f} {max(v):>8.2f}")
            if min(v) < 0.8 * med:
                line += "  <-- FLAG"
                flagged.append((gname, cname, v))
            print(line, flush=True)
    print("\nflagged metrics (all samples):")
    for gname, cname, v in flagged or []:
        print(f"  {gname} / {cname}: " + " ".join(f"{x:.1f}" for x in v))
    if not flagged:
        print("  none")
    print("COLLECTIVE_REP_COMPLETE")

dist.barrier()
dist.destroy_process_group()
PYEOF

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29521

srun -ul --ntasks-per-node=4 --environment="$RUN_ENV" \
    env MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" SWEEPS="$SWEEPS" \
    "$PROBE_PYTHON" "$WORK/collrep.py"

echo "DONE: $SLURM_JOB_NODELIST"
