#!/bin/bash
# =============================================================================
# Burst collective benchmark: barrier OUTSIDE the loop — 10 collectives
# enqueued back-to-back, one sync, overall time.  Plus a simultaneous phase
# where both nodes' intra groups burst at the same time.
# =============================================================================
#
# Difference vs nccl_collective_rep.sh: there, every sample is
# barrier -> N iters -> sync (isolated latency).  Here a sample is
# barrier -> enqueue BURST collectives with NO sync between -> one sync ->
# overall time (sustained/pipelined throughput, how real workloads run).
# Collectives on the same group serialize on the NCCL stream (pipelined,
# no host gaps); collectives on DIFFERENT groups run on different NCCL
# streams, so the "simultaneous" phase overlaps intra-A and intra-B bursts
# for a genuine concurrency/contention test.
#
# Reported per (phase, group, collective): overall burst time in ms and
# aggregate busbw = BURST * wire / time, mean/std/min-bw over SWEEPS reps.
#
# RUN:  NCCL_BENCH_NODELIST=nidA,nidB ./nccl_burst_rep.sh
#
#SBATCH --job-name=nccl-burst
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
BURST=${NCCL_BENCH_BURST:-10}
WORK=$ROOT/tmp/nccl-burst-${SLURM_JOB_ID}

test -f "$RUN_ENV"
mkdir -p "$WORK"
echo "=== nodes: $SLURM_JOB_NODELIST  sweeps: $SWEEPS  burst: $BURST ==="

cat > "$WORK/burst.py" <<'PYEOF'
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
BURST = int(os.environ["BURST"])
NBYTES = 64 << 20

names = [None] * world
dist.all_gather_object(names, f"{socket.gethostname()}:g{local_rank}")
node_a = names[0].split(":")[0]
node_b = names[4].split(":")[0]
if rank == 0:
    print(f"rank map: {names}", flush=True)
    print(f"burst = {BURST} x {NBYTES >> 20} MiB back-to-back, one sync; "
          f"{SWEEPS} reps", flush=True)

g_a = dist.new_group(list(range(4)))
g_b = dist.new_group(list(range(4, 8)))
g_w = dist.new_group(list(range(8)))
GROUPS = [(f"intra {node_a}", list(range(4)), g_a),
          (f"intra {node_b}", list(range(4, 8)), g_b),
          ("all-node (8)", list(range(8)), g_w)]


def make_cases(n):
    x = torch.randn(NBYTES // 4, dtype=torch.float32, device="cuda")
    ag_in = torch.randn(NBYTES // 4 // n, dtype=torch.float32, device="cuda")
    ag_out = torch.empty(NBYTES // 4 // n * n, dtype=torch.float32, device="cuda")
    rs_out = torch.empty(NBYTES // 4 // n, dtype=torch.float32, device="cuda")
    return [
        ("allreduce", lambda g: dist.all_reduce(x, group=g),
         2 * (n - 1) / n * NBYTES),
        ("allgather", lambda g: dist.all_gather_into_tensor(ag_out, ag_in, group=g),
         (n - 1) / n * (NBYTES // n * n)),
        ("reducescatter", lambda g: dist.reduce_scatter_tensor(rs_out, x, group=g),
         (n - 1) / n * NBYTES),
    ]


def burst_time(fn, g):
    torch.cuda.synchronize()
    dist.barrier()                      # barrier OUTSIDE the loop, world-wide
    t0 = time.perf_counter()
    for _ in range(BURST):
        fn(g)                           # no sync between collectives
    torch.cuda.synchronize()
    return time.perf_counter() - t0


samples = {}  # (phase, group, collective) -> [(ms, busbw GB/s)]
for sweep in range(SWEEPS):
    # phase 1: sequential bursts per group
    for gname, ranks, g in GROUPS:
        n = len(ranks)
        if rank in ranks:
            for cname, fn, wire in make_cases(n):
                if sweep == 0:
                    for _ in range(3):
                        fn(g)
                    torch.cuda.synchronize()
                t = burst_time(fn, g)
                if rank == ranks[0]:
                    samples.setdefault(("burst", gname, cname), []).append(
                        (t * 1e3, BURST * wire / t / 1e9))
        else:
            for _ in range(3):          # keep world barrier counts aligned
                burst_time(lambda _g: None, None)
    # phase 2: intra-A and intra-B burst SIMULTANEOUSLY (different NCCL
    # streams -> genuine overlap); each node group times its own burst
    g = g_a if rank < 4 else g_b
    gname = f"intra {node_a}" if rank < 4 else f"intra {node_b}"
    n = 4
    for cname, fn, wire in make_cases(n):
        t = burst_time(fn, g)
        if rank in (0, 4):
            samples.setdefault(("simultaneous", gname, cname), []).append(
                (t * 1e3, BURST * wire / t / 1e9))
    if rank == 0:
        print(f"sweep {sweep + 1}/{SWEEPS} done", flush=True)

gathered = [None] * world
dist.all_gather_object(gathered, samples)
if rank == 0:
    merged = {}
    for part in gathered:
        merged.update(part)
    print(f"\n{'phase':<13} {'group':<18} {'collective':<14} "
          f"{'ms mean':>8} {'ms max':>8} {'bw mean':>8} {'bw std':>7} "
          f"{'bw min':>8}", flush=True)
    flagged = []
    for key in sorted(merged):
        phase, gname, cname = key
        ms = [a for a, _ in merged[key]]
        bw = [b for _, b in merged[key]]
        med = statistics.median(bw)
        line = (f"{phase:<13} {gname:<18} {cname:<14} "
                f"{statistics.mean(ms):>8.2f} {max(ms):>8.2f} "
                f"{statistics.mean(bw):>8.2f} {statistics.stdev(bw):>7.2f} "
                f"{min(bw):>8.2f}")
        if min(bw) < 0.8 * med:
            line += "  <-- FLAG"
            flagged.append((key, bw))
        print(line, flush=True)
    print("\nflagged metrics (all busbw samples):")
    for (phase, gname, cname), bw in flagged or []:
        print(f"  {phase}/{gname}/{cname}: "
              + " ".join(f"{x:.1f}" for x in bw))
    if not flagged:
        print("  none")
    print("BURST_REP_COMPLETE")

dist.barrier()
dist.destroy_process_group()
PYEOF

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29523

srun -ul --ntasks-per-node=4 --environment="$RUN_ENV" \
    env MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    SWEEPS="$SWEEPS" BURST="$BURST" \
    "$PROBE_PYTHON" "$WORK/burst.py"

echo "DONE: $SLURM_JOB_NODELIST"
