#!/bin/bash
# Standard-stack cluster-health repro: stock lm-eval + untouched
# vllm-openai:v0.27.1 + Qwen/Qwen3-30B-A3B on 2 nodes, TP=8 with
# enable_expert_parallel (vLLM runs the MoE layers at EP=8 -> the expert
# dispatch is the same 8-rank allgather/reduce-scatter the sick nodes
# collapse on).  No DP anywhere: stock lm-eval cannot drive vLLM's
# external-launcher DP, so the standard multinode path is a 2-node Ray
# cluster and one lm_eval process on the head.
#
# Everything (weights, datasets) is prefetched by qwen3_standard_setup.sbatch;
# the job runs HF-offline and the only variable across runs is the node pair.
#
# Two multi-node-TP init deadlocks are defused (both intra-node-only
# optimizations, zero perf cost for a cross-node group): symm-mem allreduce
# (VLLM_ALLREDUCE_USE_SYMM_MEM=0 in the EDF; job 3144552) and custom
# allreduce, whose per-rank MNNVL probe diverged on GH200 -- one rank entered
# _init_mnnvl_buffer while its peers skipped it (disable_custom_all_reduce;
# job 3144711, py-spy stacks).
#
# Usage:  LABEL=<tag> sbatch --nodelist=nidA,nidB qwen3_standard_eval.sbatch
#         LIMIT=32 for a prewarm/smoke run (also warms the vLLM compile cache).

#SBATCH --job-name=qwen3-std-eval
#SBATCH --time=01:30:00
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --account=infra01
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -euo pipefail
: "${LABEL:?LABEL is required (names the output dir)}"

ROOT=/iopsstor/scratch/cscs/$USER
RUN_ENV=$ROOT/node-health/qwen3-bench/vllm-standard-v0271.toml
VENV=$ROOT/vllm-play/lmeval-std-venv
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
# Parallel layout, overridable per submit (e.g. the production-shaped 3-node
# variant: --nodes=3 + PARALLEL_ARGS="tensor_parallel_size=4,pipeline_parallel_size=3")
PARALLEL_ARGS=${PARALLEL_ARGS:-tensor_parallel_size=8}
NGPUS=$((SLURM_NNODES * 4))
TASKS=${TASKS:-piqa,arc_challenge,arc_easy,hellaswag,winogrande,xcopa,xnli,xwinograd,pawsx,mmlu}
OUT=$ROOT/node-health/results/$LABEL-$SLURM_JOB_ID
WORK=$ROOT/tmp/qwen3std-$SLURM_JOB_ID

test -f "$RUN_ENV"
test -x "$VENV/bin/lm_eval"
mkdir -p "$OUT" "$WORK"

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
RAY_PORT=$((25000 + SLURM_JOB_ID % 10000))

echo "START TIME : $(date --iso-8601=seconds)"
echo "nodes      : $SLURM_JOB_NODELIST (head $HEAD:$RAY_PORT)"
echo "model      : $MODEL  $PARALLEL_ARGS +expert-parallel via ray ($NGPUS GPUs)"
echo "tasks      : $TASKS ${LIMIT:+(LIMIT=$LIMIT)}"
echo "output     : $OUT"

cat > "$WORK/inner.sh" <<EOF
#!/bin/bash
set -u
export HEAD=$HEAD RAY_PORT=$RAY_PORT OUT=$OUT VENV=$VENV
export MODEL=$MODEL TASKS=$TASKS LIMIT=${LIMIT:-}
export PARALLEL_ARGS=$PARALLEL_ARGS NGPUS=$NGPUS
EOF
cat >> "$WORK/inner.sh" <<'EOF'
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_CACHE_ROOT=/iopsstor/scratch/cscs/mvasilev/vllm-play/vllm_cache_std
DONE=$OUT/.ray-done
RAY=$(command -v ray || true); [ -n "$RAY" ] || RAY=$VENV/bin/ray

if [ "$SLURM_PROCID" -eq 0 ]; then
    "$RAY" start --head --port="$RAY_PORT" --num-gpus=4 \
        --disable-usage-stats --include-dashboard=false || exit 1
    for i in $(seq 120); do
        GPUS=$("$VENV/bin/python" - <<'PY'
import ray
ray.init(address="auto", logging_level="error")
print(int(ray.cluster_resources().get("GPU", 0)))
PY
        )
        [ "${GPUS:-0}" -ge 8 ] && break
        sleep 5
    done
    if [ "${GPUS:-0}" -lt 8 ]; then
        echo "FATAL: ray cluster never reached 8 GPUs (got ${GPUS:-0})" >&2
        touch "$DONE"; "$RAY" stop; exit 1
    fi
    echo "ray cluster up: $GPUS GPUs"
    T0=$(date +%s)
    "$VENV/bin/lm_eval" --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=8,enable_expert_parallel=True,distributed_executor_backend=ray,disable_custom_all_reduce=True" \
        --tasks "$TASKS" \
        --batch_size auto \
        --output_path "$OUT" \
        ${LIMIT:+--limit "$LIMIT"}
    RC=$?
    T1=$(date +%s)
    echo "LM_EVAL_WALL_SECONDS $((T1 - T0)) rc=$RC"
    touch "$DONE"
    "$RAY" stop
    exit $RC
else
    for i in $(seq 150); do
        (exec 3<>"/dev/tcp/$HEAD/$RAY_PORT") 2>/dev/null && break
        sleep 2
    done
    "$RAY" start --address="$HEAD:$RAY_PORT" --num-gpus=4 --disable-usage-stats || exit 1
    while [ ! -f "$DONE" ]; do sleep 5; done
    "$RAY" stop
    exit 0
fi
EOF
chmod +x "$WORK/inner.sh"

set +e
srun -ul --ntasks=2 --ntasks-per-node=1 --environment="$RUN_ENV" "$WORK/inner.sh"
RC=$?
set -e
rm -f "$OUT/.ray-done"
echo "END TIME   : $(date --iso-8601=seconds)  rc=$RC"
exit $RC
