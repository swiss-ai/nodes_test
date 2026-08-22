#!/bin/bash
# =============================================================================
# Repeated pairwise 64 MiB bandwidth matrix: N full sweeps, mean/std/min.
# =============================================================================
#
# Statistical version of nccl_p2p_matrix.sh: measures all 28 GPU pairs
# (ping-pong, fresh RANDOM 64 MiB buffers each sweep), then repeats the whole
# sweep NCCL_BENCH_SWEEPS times (default 10) inside ONE job, so reps of the
# same pair are separated in time — that is what catches intermittent cells.
# Prints three 8x8 matrices: mean GB/s, std, min.  Flaky link fingerprint:
# std blows up and min collapses while mean may look acceptable.
#
# RUN:  NCCL_BENCH_NODELIST=nidA,nidB ./nccl_p2p_matrix_rep.sh
#
#SBATCH --job-name=nccl-p2prep
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
SWEEPS=${NCCL_BENCH_SWEEPS:-10}
WORK=$ROOT/tmp/nccl-p2prep-${SLURM_JOB_ID}

test -f "$RUN_ENV"
mkdir -p "$WORK"
echo "=== nodes: $SLURM_JOB_NODELIST  EDF: $RUN_ENV  sweeps: $SWEEPS ==="

cat > "$WORK/p2prep.py" <<'PYEOF'
import itertools
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
ITERS = 10

names = [None] * world
dist.all_gather_object(names, f"{socket.gethostname()}:g{local_rank}")
if rank == 0:
    print(f"rank map: {names}", flush=True)
    print(f"{SWEEPS} sweeps x {ITERS} ping-pong iters, "
          f"random {NBYTES >> 20} MiB buffers", flush=True)

pairs = list(itertools.combinations(range(world), 2))
samples = {}  # (a, b) -> [GB/s per sweep], held by rank a

for sweep in range(SWEEPS):
    for a, b in pairs:
        if rank in (a, b):
            peer = b if rank == a else a
            buf = torch.randn(NBYTES // 4, dtype=torch.float32, device="cuda")
            warm = 10 if sweep == 0 else 2  # comm creation only on sweep 0
            for _ in range(warm):
                if rank == a:
                    dist.send(buf, peer); dist.recv(buf, peer)
                else:
                    dist.recv(buf, peer); dist.send(buf, peer)
            torch.cuda.synchronize()
        dist.barrier()
        if rank in (a, b):
            peer = b if rank == a else a
            t0 = time.perf_counter()
            for _ in range(ITERS):
                if rank == a:
                    dist.send(buf, peer); dist.recv(buf, peer)
                else:
                    dist.recv(buf, peer); dist.send(buf, peer)
            torch.cuda.synchronize()
            one_way = (time.perf_counter() - t0) / ITERS / 2
            if rank == a:
                samples.setdefault((a, b), []).append(NBYTES / one_way / 1e9)
        dist.barrier()
    if rank == 0:
        print(f"sweep {sweep + 1}/{SWEEPS} done", flush=True)

gathered = [None] * world
dist.all_gather_object(gathered, samples)
if rank == 0:
    merged = {}
    for part in gathered:
        merged.update(part)

    def show(title, fn):
        print(f"\n=== 64 MiB bandwidth, {title} (GB/s) ===", flush=True)
        print("      " + "".join(f"  r{j}    " for j in range(world)))
        for i in range(world):
            row = [f"r{i}  "]
            for j in range(world):
                row.append("   .    " if j <= i else f"{fn(merged[(i, j)]):7.2f} ")
            print("".join(row), flush=True)

    for s in range(SWEEPS):
        show(f"SWEEP {s + 1}", lambda v, s=s: v[s])
    show("MEAN", statistics.mean)
    show("STD", lambda v: statistics.stdev(v) if len(v) > 1 else 0.0)
    show("MIN", min)
    flagged = [
        (i, j, merged[(i, j)])
        for i, j in pairs
        if min(merged[(i, j)]) < (85 if (i < 4) == (j < 4) else 21)
    ]
    print("\nflagged cells (all samples):")
    for i, j, v in flagged or []:
        print(f"  r{i}-r{j}: " + " ".join(f"{x:.1f}" for x in v))
    if not flagged:
        print("  none")
    print("\nrows/cols r0-r3 = first node GPUs 0-3, r4-r7 = second node")
    print("P2P_REP_COMPLETE")

dist.barrier()
dist.destroy_process_group()
PYEOF

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29519

srun -ul --ntasks-per-node=4 --environment="$RUN_ENV" \
    env MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" SWEEPS="$SWEEPS" \
    "$PROBE_PYTHON" "$WORK/p2prep.py"

echo "DONE: $SLURM_JOB_NODELIST"
