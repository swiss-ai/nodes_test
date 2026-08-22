#!/bin/bash
# One-time download of the standard vLLM benchmark dataset (~640 MB).
# Run from a login node (compute nodes also have outbound HTTPS, but there
# is no reason to burn a GPU allocation on a download).
set -euo pipefail
cd "$(dirname "$0")"

URL=https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
OUT=ShareGPT_V3_unfiltered_cleaned_split.json

if [[ -f "$OUT" ]]; then
    echo "$OUT already exists ($(du -h "$OUT" | cut -f1)) — nothing to do."
    exit 0
fi

wget -O "$OUT.part" "$URL"
mv "$OUT.part" "$OUT"
echo "Done: $(du -h "$OUT" | cut -f1)"
