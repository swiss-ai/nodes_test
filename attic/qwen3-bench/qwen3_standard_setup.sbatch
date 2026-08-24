#!/bin/bash
# One-time setup for the standard-stack repro: stock lm-eval venv on top of
# the untouched vllm-openai:v0.27.1 image, Qwen3-30B-A3B weights, and all
# task datasets prefetched into hf_home_std -- so the timed eval jobs run
# fully offline and measure only speed.
#
# Usage:  sbatch [--dependency=afterok:<import job>] qwen3_standard_setup.sbatch

#SBATCH --job-name=qwen3-std-setup
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=230000
#SBATCH --no-requeue
#SBATCH --account=infra01
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -euo pipefail
echo "START TIME: $(date)"

ROOT=/iopsstor/scratch/cscs/$USER
RUN_ENV=$ROOT/node-health/qwen3-bench/vllm-standard-v0271.toml
VENV=$ROOT/vllm-play/lmeval-std-venv
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
TASKS=${TASKS:-piqa,arc_challenge,arc_easy,hellaswag,winogrande,xcopa,xnli,xwinograd,pawsx,mmlu}
WORK=$ROOT/tmp/qwen3std-setup-$SLURM_JOB_ID

test -f "$RUN_ENV"
mkdir -p "$WORK"

cat > "$WORK/setup.sh" <<EOF
#!/bin/bash
set -euo pipefail
export VENV=$VENV
export MODEL=$MODEL
export TASKS=$TASKS
EOF
cat >> "$WORK/setup.sh" <<'EOF'
# The base image's ENV NCCL_VERSION poisons dependency resolvers
# (build_eval_container lesson); harmless to drop for pip too.
unset NCCL_VERSION || true

# The compute-node NAT IP shares one anonymous HF rate-limit pool (job
# 3144227: 429, 0/500 requests).  Re-read the token before every attempt so
# it can be added to ~/.hf-token while this job runs; the limit window is
# 300 s, hence the 150 s backoff.
retry() {
    local n=0
    while true; do
        export HF_TOKEN=$(cat "$HOME/.hf-token" 2>/dev/null || true)
        [ -n "$HF_TOKEN" ] || unset HF_TOKEN
        "$@" && return 0
        n=$((n + 1))
        [ "$n" -ge 15 ] && { echo "FATAL: 15 attempts failed: $*" >&2; return 1; }
        echo "attempt $n failed (HF rate limit?), retrying in 150 s"
        sleep 150
    done
}

echo "=== container versions (system) ==="
python3 --version
python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
command -v ray && ray --version || echo "no system ray CLI"

echo "=== venv + stock lm-eval ==="
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/pip" install --upgrade pip >/dev/null
"$VENV/bin/pip" install lm_eval
"$VENV/bin/pip" show lm_eval | sed -n '1,2p'
# stock install must NOT have replaced the image's vllm/torch, and lm_eval
# must resolve from the venv, not from ~/.local (PYTHONNOUSERSITE guards it)
"$VENV/bin/python" - <<'PY'
import os, vllm, torch, lm_eval
assert vllm.__version__ == "0.27.1", vllm.__version__
assert lm_eval.__file__.startswith(os.environ["VENV"]), lm_eval.__file__
print("OK: lm_eval", lm_eval.__version__, "| vllm", vllm.__version__,
      "| torch", torch.__version__)
PY

cat > /tmp/dl_model.py <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(os.environ["MODEL"])
print("snapshot at", p)
PY
cat > /tmp/prefetch.py <<'PY'
import os
from lm_eval.tasks import TaskManager, get_task_dict
names = os.environ["TASKS"].split(",")
td = get_task_dict(names, TaskManager())
def leaves(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v)
        else:
            yield k
print("prefetched tasks:", len(list(leaves(td))))
PY

echo "=== model download: $MODEL ==="
retry "$VENV/bin/python" /tmp/dl_model.py

echo "=== dataset prefetch: $TASKS ==="
retry "$VENV/bin/python" /tmp/prefetch.py

echo "=== offline verification ==="
env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$VENV/bin/python" - <<'PY'
import os
from lm_eval.tasks import TaskManager, get_task_dict
from huggingface_hub import snapshot_download
get_task_dict(os.environ["TASKS"].split(","), TaskManager())
snapshot_download(os.environ["MODEL"])
print("OFFLINE OK: tasks + model resolve without network")
PY
echo "SETUP_COMPLETE"
EOF
chmod +x "$WORK/setup.sh"

srun -ul --ntasks=1 --environment="$RUN_ENV" "$WORK/setup.sh"

du -sh "$ROOT/vllm-play/hf_home_std" || true
echo "END TIME: $(date)"
