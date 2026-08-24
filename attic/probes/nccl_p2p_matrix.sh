#!/bin/bash
# =============================================================================
# Pairwise GPU-to-GPU send/recv matrix over a 2-node allocation (28 pairs).
# =============================================================================
#
# Sibling of nccl_allreduce_bench.sh, same runtime (baked eval image).
# For every unordered pair of the 8 GPUs (2 nodes x 4), a NCCL ping-pong
# measures one-way latency (8 B, 64 KiB) and bandwidth (64 MiB) while all
# other ranks idle at a barrier.  Layout of the result matrix:
#   ranks 0-3 = node A GPUs 0-3, ranks 4-7 = node B GPUs 0-3.
#   Intra-node cells -> NVLink health; the 4x4 cross-node block -> per-NIC
#   paths (NCCL uses the NIC nearest each GPU).  A slow row/column in the
#   cross block fingers one GPU's NIC; a uniformly slow block fingers the
#   inter-node route.
#
# RUN:  NCCL_BENCH_NODELIST=nidA,nidB ./nccl_p2p_matrix.sh
#       (submits itself; log lands as nccl-p2pmatrix-<jobid>.log in cwd)
#
#SBATCH --job-name=nccl-p2pmatrix
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
WORK=$ROOT/tmp/nccl-p2pmatrix-${SLURM_JOB_ID}

test -f "$RUN_ENV"
mkdir -p "$WORK"
echo "=== nodes: $SLURM_JOB_NODELIST  EDF: $RUN_ENV ==="

cat > "$WORK/p2p.py" <<'PYEOF'
import itertools
import os
import socket
import time

import torch
import torch.distributed as dist

os.environ["RANK"] = os.environ["SLURM_PROCID"]
os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
local_rank = int(os.environ["SLURM_LOCALID"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
host = socket.gethostname()

names = [None] * world
dist.all_gather_object(names, f"{host}:g{local_rank}")
if rank == 0:
    print(f"rank map: {names}", flush=True)

# (bytes, iters); ping-pong, one-way latency = RTT/2
CASES = [(8, 200), (64 << 10, 200), (64 << 20, 10)]
results = {}

for a, b in itertools.combinations(range(world), 2):
    for nbytes, iters in CASES:
        n = max(1, nbytes // 4)
        buf = torch.ones(n, dtype=torch.float32, device="cuda")
        if rank in (a, b):
            peer = b if rank == a else a
            for _ in range(10):  # warm up the pair's channels
                if rank == a:
                    dist.send(buf, peer); dist.recv(buf, peer)
                else:
                    dist.recv(buf, peer); dist.send(buf, peer)
            torch.cuda.synchronize()
        dist.barrier()
        if rank in (a, b):
            peer = b if rank == a else a
            t0 = time.perf_counter()
            for _ in range(iters):
                if rank == a:
                    dist.send(buf, peer); dist.recv(buf, peer)
                else:
                    dist.recv(buf, peer); dist.send(buf, peer)
            torch.cuda.synchronize()
            one_way = (time.perf_counter() - t0) / iters / 2
            if rank == a:
                results[(a, b, nbytes)] = one_way
        dist.barrier()

# rank a measured each of its pairs; funnel everything to rank 0
gathered = [None] * world
dist.all_gather_object(gathered, results)
if rank == 0:
    merged = {}
    for part in gathered:
        merged.update(part)
    for nbytes, _ in CASES:
        print(f"\n=== one-way, {nbytes} bytes "
              f"({'us' if nbytes < 1 << 20 else 'GB/s'}) ===", flush=True)
        print("      " + "".join(f"  r{j}    " for j in range(world)))
        for i in range(world):
            row = [f"r{i}  "]
            for j in range(world):
                if j <= i:
                    row.append("   .    ")
                else:
                    v = merged[(i, j, nbytes)]
                    if nbytes < 1 << 20:
                        row.append(f"{v * 1e6:7.1f} ")
                    else:
                        row.append(f"{nbytes / v / 1e9:7.2f} ")
            print("".join(row), flush=True)
    print("\nrows/cols r0-r3 = first node GPUs 0-3, r4-r7 = second node")
    print("P2P_MATRIX_COMPLETE")

dist.barrier()
dist.destroy_process_group()
PYEOF

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29517

srun -ul --ntasks-per-node=4 --environment="$RUN_ENV" \
    env MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    "$PROBE_PYTHON" "$WORK/p2p.py"

echo "DONE: $SLURM_JOB_NODELIST"
