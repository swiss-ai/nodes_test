#!/bin/bash
# Import the untouched upstream vLLM release image (no Apertus2 overlay, no
# eval venv) for the standard-stack cluster-health reproduction.  Same digest
# the Apertus2 eval image was built FROM (build_eval_container.sbatch), so the
# comparison isolates the cluster, not the software version.
#
# Usage:  sbatch import_vllm_standard.sbatch

#SBATCH --job-name=vllm-std-import
#SBATCH --time=00:45:00
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --account=infra01
#SBATCH --cpus-per-task=72
#SBATCH --mem=230000
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -euo pipefail
echo "START TIME: $(date)"

UPSTREAM_IMAGE=${UPSTREAM_IMAGE:-docker://vllm/vllm-openai:v0.27.1@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c}
SQSH_OUT=${SQSH_OUT:-/iopsstor/scratch/cscs/$USER/.edf_imagestore/vllm+vllm-openai+v0.27.1.aarch64.sqsh}

die() { echo "FATAL: $*" >&2; exit 1; }

test ! -e "$SQSH_OUT" || die "already imported: $SQSH_OUT"
enroot version

# Import under a temporary name, validate the SquashFS, publish with mv.
# Trust the artifact, not enroot's exit code (see build_eval_container.sbatch,
# job 3125570).
TMP_SQSH=$SQSH_OUT.tmp.$SLURM_JOB_ID
test ! -e "$TMP_SQSH" || die "temporary image already exists: $TMP_SQSH"
ENROOT_RC=0
enroot import -o "$TMP_SQSH" "$UPSTREAM_IMAGE" || ENROOT_RC=$?
echo "enroot import exited $ENROOT_RC"
test -s "$TMP_SQSH" || die "enroot import wrote no image"
unsquashfs -s "$TMP_SQSH" || die "invalid SquashFS at $TMP_SQSH"
mv "$TMP_SQSH" "$SQSH_OUT"
ls -la "$SQSH_OUT"
echo "IMPORT OK: $SQSH_OUT"
echo "END TIME: $(date)"
