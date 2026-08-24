#!/bin/bash
# =============================================================================
# Failing-vs-control path localizer: raw CUDA P2P + traced NCCL + CXI rates.
# =============================================================================
#
# This is a diagnostic trace, not a peak-throughput benchmark.  It runs one
# suspect and one control node in the same exclusive allocation and preserves
# enough evidence to distinguish three layers:
#
#   1. raw PyTorch/CUDA peer copies, with all 4 CUDA contexts resident per node
#   2. NCCL point-to-point timings, with all 8 NCCL ranks initialized
#   3. Slingshot/CXI rate, error, and pause telemetry during the NCCL phase
#
# Nsight Systems records CUDA/NVTX/NCCL timelines for every process.  Raw-copy
# and NCCL samples are emitted as JSON lines in the Slurm log for quick parsing.
#
# RUN (defaults shown):
#   NCCL_TRACE_FAILING_NODE=nid006907 \
#   NCCL_TRACE_CONTROL_NODE=nid006856 ./nccl_path_trace.sh
#
# Knobs: NCCL_TRACE_RAW_SWEEPS=10 NCCL_TRACE_NCCL_SWEEPS=10
#        NCCL_TRACE_ITERS=10 NCCL_TRACE_SIZE_MB=64 NCCL_TRACE_DIR=/path
#        NCCL_TRACE_ENABLE_NSYS=0 (timing-only smoke; skips report generation)
#
#SBATCH --job-name=nccl-path-trace
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

FAILING_NODE=${NCCL_TRACE_FAILING_NODE:-nid006907}
CONTROL_NODE=${NCCL_TRACE_CONTROL_NODE:-nid006856}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    test "$FAILING_NODE" != "$CONTROL_NODE" || {
        echo "failing and control nodes must be different" >&2
        exit 2
    }
    exec sbatch \
        --reservation="${NCCL_BENCH_RESERVATION:-SD-69241-apertus-1-5-0}" \
        --nodelist="${NCCL_BENCH_NODELIST:-$FAILING_NODE,$CONTROL_NODE}" \
        "$0"
fi

set -euo pipefail
ulimit -c 0

ROOT=/iopsstor/scratch/cscs/mvasilev
RUN_ENV=${NCCL_BENCH_ENV:-$ROOT/vllm-play/megatron-v2601-multinode.toml}
PROBE_PYTHON=${NCCL_BENCH_PYTHON:-python3}
RAW_SWEEPS=${NCCL_TRACE_RAW_SWEEPS:-10}
NCCL_SWEEPS=${NCCL_TRACE_NCCL_SWEEPS:-10}
ITERS=${NCCL_TRACE_ITERS:-10}
SIZE_MB=${NCCL_TRACE_SIZE_MB:-64}
ENABLE_NSYS=${NCCL_TRACE_ENABLE_NSYS:-1}
GPUS_PER_NODE=4
TOTAL_TASKS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
TRACE_DIR=${NCCL_TRACE_DIR:-$ROOT/tmp/nccl-path-trace-${SLURM_JOB_ID}}
export SLURM_NETWORK=${SLURM_NETWORK:-disable_rdzv_get}

if [ "$SLURM_JOB_NUM_NODES" -ne 2 ] || [ "$TOTAL_TASKS" -ne 8 ]; then
    echo "path trace requires exactly 2 nodes x 4 GPUs" >&2
    exit 2
fi
if ! [[ "$RAW_SWEEPS" =~ ^[1-9][0-9]*$ && "$NCCL_SWEEPS" =~ ^[1-9][0-9]*$ \
        && "$ITERS" =~ ^[1-9][0-9]*$ && "$SIZE_MB" =~ ^[1-9][0-9]*$ ]]; then
    echo "sweeps, iterations, and size must be positive integers" >&2
    exit 2
fi
if [[ "$ENABLE_NSYS" != 0 && "$ENABLE_NSYS" != 1 ]]; then
    echo "NCCL_TRACE_ENABLE_NSYS must be 0 or 1" >&2
    exit 2
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#NODES[@]}" -ne 2 ]; then
    echo "could not resolve exactly two allocated hosts" >&2
    exit 2
fi
if [[ " ${NODES[*]} " != *" $FAILING_NODE "* ]] || \
        [[ " ${NODES[*]} " != *" $CONTROL_NODE "* ]]; then
    echo "allocation ${NODES[*]} does not contain failing=$FAILING_NODE and control=$CONTROL_NODE" >&2
    exit 2
fi

export FAILING_NODE CONTROL_NODE RUN_ENV PROBE_PYTHON
export RAW_SWEEPS NCCL_SWEEPS ITERS SIZE_MB ENABLE_NSYS
export GPUS_PER_NODE TOTAL_TASKS TRACE_DIR
export HOSTNAMES="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
export MASTER_ADDR="${NODES[0]}"
export MASTER_PORT=${MASTER_PORT:-29529}
export WORLD_SIZE=$TOTAL_TASKS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_CACHE_DISABLE=1

test -f "$RUN_ENV"
mkdir -p "$TRACE_DIR"

echo "=== failing: $FAILING_NODE  control: $CONTROL_NODE ==="
echo "=== allocation: $SLURM_JOB_NODELIST  EDF: $RUN_ENV ==="
echo "=== raw sweeps: $RAW_SWEEPS  NCCL sweeps: $NCCL_SWEEPS ==="
echo "=== size: ${SIZE_MB} MiB  iterations/sample: $ITERS ==="
echo "=== Nsight enabled: $ENABLE_NSYS ==="
echo "=== trace artifacts: $TRACE_DIR ==="

cat > "$TRACE_DIR/raw_cuda_p2p.py" <<'PYEOF'
import itertools
import json
import os
import socket
import statistics
import time

import torch


HOST = socket.gethostname()
ROLE = "failing" if HOST == os.environ["FAILING_NODE"] else "control"
SWEEPS = int(os.environ["RAW_SWEEPS"])
ITERS = int(os.environ["ITERS"])
NBYTES = int(os.environ["SIZE_MB"]) << 20
NELEMS = NBYTES // 4
NGPUS = torch.cuda.device_count()

if NGPUS != 4:
    raise RuntimeError(f"raw P2P phase requires 4 visible GPUs, got {NGPUS}")

# Allocating one buffer on every GPU initializes and retains all four CUDA
# contexts.  Only the selected source and destination are active in each timed
# range, so the physical link remains isolated.
buffers = []
for dev in range(NGPUS):
    with torch.cuda.device(dev):
        buffers.append(torch.randn(NELEMS, dtype=torch.float32, device=dev))
for dev in range(NGPUS):
    torch.cuda.synchronize(dev)

print(
    "RAW_CONFIG "
    + json.dumps(
        {
            "host": HOST,
            "role": ROLE,
            "visible_gpus": NGPUS,
            "sweeps": SWEEPS,
            "iters": ITERS,
            "bytes": NBYTES,
        },
        sort_keys=True,
    ),
    flush=True,
)

samples = {}
for sweep in range(1, SWEEPS + 1):
    for a, b in itertools.combinations(range(NGPUS), 2):
        for src, dst in ((a, b), (b, a)):
            warmup = 5 if sweep == 1 else 1
            with torch.cuda.device(dst):
                for _ in range(warmup):
                    buffers[dst].copy_(buffers[src], non_blocking=True)
            torch.cuda.synchronize(src)
            torch.cuda.synchronize(dst)

            label = f"raw:{HOST}:g{src}->g{dst}:s{sweep}"
            torch.cuda.nvtx.range_push(label)
            started = time.perf_counter()
            with torch.cuda.device(dst):
                for _ in range(ITERS):
                    buffers[dst].copy_(buffers[src], non_blocking=True)
            torch.cuda.synchronize(src)
            torch.cuda.synchronize(dst)
            elapsed = time.perf_counter() - started
            torch.cuda.nvtx.range_pop()

            gbps = NBYTES * ITERS / elapsed / 1.0e9
            key = f"g{src}->g{dst}"
            samples.setdefault(key, []).append(gbps)
            print(
                "RAW_SAMPLE "
                + json.dumps(
                    {
                        "host": HOST,
                        "role": ROLE,
                        "sweep": sweep,
                        "path": key,
                        "gbps": round(gbps, 4),
                        "elapsed_s": elapsed,
                        "time_ns": time.time_ns(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

for path in sorted(samples):
    values = samples[path]
    print(
        "RAW_SUMMARY "
        + json.dumps(
            {
                "host": HOST,
                "role": ROLE,
                "path": path,
                "mean_gbps": round(statistics.mean(values), 4),
                "std_gbps": round(statistics.stdev(values), 4)
                if len(values) > 1
                else 0.0,
                "min_gbps": round(min(values), 4),
                "max_gbps": round(max(values), 4),
                "samples": len(values),
            },
            sort_keys=True,
        ),
        flush=True,
    )
print(f"RAW_P2P_COMPLETE {HOST} {ROLE}", flush=True)
PYEOF

cat > "$TRACE_DIR/nccl_p2p_trace.py" <<'PYEOF'
import itertools
import json
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

rank = dist.get_rank()
world = dist.get_world_size()
host = socket.gethostname()
role = "failing" if host == os.environ["FAILING_NODE"] else "control"
sweeps = int(os.environ["NCCL_SWEEPS"])
iters = int(os.environ["ITERS"])
nbytes = int(os.environ["SIZE_MB"]) << 20

labels = [None] * world
dist.all_gather_object(
    labels,
    {
        "rank": rank,
        "host": host,
        "role": role,
        "gpu": local_rank,
    },
)
if rank == 0:
    print("NCCL_RANK_MAP " + json.dumps(labels, sort_keys=True), flush=True)

# Every rank retains a 64 MiB buffer and its NCCL communicator for the entire
# run.  Only the selected two ranks enter each timed send/receive range.
buf = torch.randn(nbytes // 4, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
pairs = list(itertools.combinations(range(world), 2))
local_samples = {}

for sweep in range(1, sweeps + 1):
    for a, b in pairs:
        if rank in (a, b):
            peer = b if rank == a else a
            warmup = 5 if sweep == 1 else 1
            for _ in range(warmup):
                if rank == a:
                    dist.send(buf, peer)
                    dist.recv(buf, peer)
                else:
                    dist.recv(buf, peer)
                    dist.send(buf, peer)
            torch.cuda.synchronize()
        dist.barrier()

        if rank in (a, b):
            peer = b if rank == a else a
            pair_label = (
                f"nccl:r{a}-{b}:"
                f"{labels[a]['host']}:g{labels[a]['gpu']}<->"
                f"{labels[b]['host']}:g{labels[b]['gpu']}:s{sweep}"
            )
            torch.cuda.nvtx.range_push(pair_label)
            started = time.perf_counter()
            for _ in range(iters):
                if rank == a:
                    dist.send(buf, peer)
                    dist.recv(buf, peer)
                else:
                    dist.recv(buf, peer)
                    dist.send(buf, peer)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            torch.cuda.nvtx.range_pop()
            if rank == a:
                # Each iteration contains one transfer in each direction.  The
                # reported value is effective one-way bandwidth, matching the
                # repeated p2p matrix probe.
                one_way_s = elapsed / iters / 2
                gbps = nbytes / one_way_s / 1.0e9
                local_samples.setdefault((a, b), []).append(
                    {
                        "sweep": sweep,
                        "gbps": gbps,
                        "elapsed_s": elapsed,
                        "time_ns": time.time_ns(),
                    }
                )
        dist.barrier()

gathered = [None] * world
dist.all_gather_object(gathered, local_samples)
if rank == 0:
    merged = {}
    for part in gathered:
        merged.update(part)
    for (a, b), rows in sorted(merged.items()):
        path = {
            "rank_a": a,
            "rank_b": b,
            "host_a": labels[a]["host"],
            "host_b": labels[b]["host"],
            "role_a": labels[a]["role"],
            "role_b": labels[b]["role"],
            "gpu_a": labels[a]["gpu"],
            "gpu_b": labels[b]["gpu"],
            "scope": "intra" if labels[a]["host"] == labels[b]["host"] else "cross",
        }
        for row in rows:
            print(
                "NCCL_SAMPLE "
                + json.dumps({**path, **row, "gbps": round(row["gbps"], 4)}, sort_keys=True),
                flush=True,
            )
        values = [row["gbps"] for row in rows]
        print(
            "NCCL_SUMMARY "
            + json.dumps(
                {
                    **path,
                    "mean_gbps": round(statistics.mean(values), 4),
                    "std_gbps": round(statistics.stdev(values), 4)
                    if len(values) > 1
                    else 0.0,
                    "min_gbps": round(min(values), 4),
                    "max_gbps": round(max(values), 4),
                    "samples": len(values),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print("NCCL_PATH_TRACE_COMPLETE", flush=True)

dist.barrier()
dist.destroy_process_group()
PYEOF

cat > "$TRACE_DIR/profile_raw_node.sh" <<'SHEOF'
#!/bin/bash
set -euo pipefail
node=$(hostname)
output="$TRACE_DIR/raw-${node}"
unset DEBUGINFOD_URLS || true
if [ "$ENABLE_NSYS" = 1 ] && command -v nsys >/dev/null 2>&1; then
    exec nsys profile \
        --force-overwrite=true --sample=none --cpuctxsw=none \
        --resolve-symbols=false --show-source-info=false --trace=cuda,nvtx \
        --output="$output" \
        "$PROBE_PYTHON" "$TRACE_DIR/raw_cuda_p2p.py"
fi
echo "WARNING: nsys not found; raw timing will run without a timeline" >&2
exec "$PROBE_PYTHON" "$TRACE_DIR/raw_cuda_p2p.py"
SHEOF

cat > "$TRACE_DIR/profile_nccl_rank.sh" <<'SHEOF'
#!/bin/bash
set -euo pipefail
node=$(hostname)
output="$TRACE_DIR/nccl-${node}-r${SLURM_LOCALID}"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,NET,TUNING
export NCCL_DEBUG_FILE="$TRACE_DIR/nccl-debug-${node}-r${SLURM_LOCALID}-%p.log"
unset DEBUGINFOD_URLS || true

if [ "$ENABLE_NSYS" = 1 ] && command -v nsys >/dev/null 2>&1; then
    trace_args=(--trace=cuda,nvtx)
    if nsys profile --help 2>&1 | grep -q -- '--nccl-trace'; then
        trace_args=(--trace=cuda,nvtx,nccl --nccl-trace=all)
    fi
    exec nsys profile \
        --force-overwrite=true --sample=none --cpuctxsw=none \
        --resolve-symbols=false --show-source-info=false "${trace_args[@]}" \
        --output="$output" \
        "$PROBE_PYTHON" "$TRACE_DIR/nccl_p2p_trace.py"
fi
echo "WARNING: nsys not found; NCCL timing will run without a timeline" >&2
exec "$PROBE_PYTHON" "$TRACE_DIR/nccl_p2p_trace.py"
SHEOF

chmod 700 "$TRACE_DIR/profile_raw_node.sh" "$TRACE_DIR/profile_nccl_rank.sh"

echo ""
echo "=== PHASE 1: raw directed CUDA P2P, all four contexts resident ==="
srun -ul --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=18 \
    --gpus-per-node="$GPUS_PER_NODE" --gpu-bind=none --mpi=pmix \
    --network="$SLURM_NETWORK" --environment="$RUN_ENV" \
    "$TRACE_DIR/profile_raw_node.sh"

echo ""
echo "=== PHASE 2: all-rank NCCL P2P trace + CXI telemetry ==="
CXI_STOP_FILE="$TRACE_DIR/.stop-cxi"
export CXI_STOP_FILE
rm -f "$CXI_STOP_FILE"

srun -u --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --cpus-per-task=1 bash -c '
        output="$TRACE_DIR/cxi-$(hostname).log"
        if ! command -v cxi_stat >/dev/null 2>&1; then
            echo "cxi_stat not available on $(hostname)" > "$output"
            exit 0
        fi
        while [ ! -e "$CXI_STOP_FILE" ]; do
            date --iso-8601=ns
            cxi_stat -r -p 1 || true
        done > "$output" 2>&1
    ' &
CXI_STEP_PID=$!

stop_cxi() {
    touch "$CXI_STOP_FILE"
    wait "$CXI_STEP_PID" || true
}
trap stop_cxi EXIT

set +e
srun -ul --overlap --ntasks="$TOTAL_TASKS" --ntasks-per-node="$GPUS_PER_NODE" \
    --distribution=block:block --mpi=pmix --network="$SLURM_NETWORK" \
    --environment="$RUN_ENV" \
    env MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    "$TRACE_DIR/profile_nccl_rank.sh"
NCCL_RC=$?
set -e

stop_cxi
trap - EXIT
if [ "$NCCL_RC" -ne 0 ]; then
    echo "NCCL trace step failed with exit code $NCCL_RC" >&2
    exit "$NCCL_RC"
fi

echo ""
echo "=== artifacts ==="
find "$TRACE_DIR" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
echo "PATH_TRACE_COMPLETE failing=$FAILING_NODE control=$CONTROL_NODE artifacts=$TRACE_DIR"
