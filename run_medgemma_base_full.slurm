#!/bin/bash
#SBATCH --job-name=medgemma_base_full
#SBATCH --output=/data/jperez/projects/utils_medgemma/results/slurm_medgemma_base_full_%j.out
#SBATCH --error=/data/jperez/projects/utils_medgemma/results/slurm_medgemma_base_full_%j.err
#SBATCH --partition=nukwa-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE

set -euo pipefail

PROJECT_DIR=/data/jperez/projects/utils_medgemma
PYTHON=/data/jperez/app_data/miniconda/envs/medgemma/bin/python
cd "$PROJECT_DIR"
mkdir -p results
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export HF_HOME=/data/jperez/app_data/huggingface
export XDG_CACHE_HOME=/data/jperez/app_data/cache
export TORCH_HOME=/data/jperez/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
export MEDGEMMA_RUN_MODE=full
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

"$PYTHON" - <<'PY'
import sys
import torch
from huggingface_hub import get_token
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is not available in this job"
assert get_token(), "Hugging Face login missing; run: hf auth login"
print("CUDA device:", torch.cuda.get_device_name(0))
PY

LIMIT="${MEDGEMMA_LIMIT:-}"
OUTPUT="${MEDGEMMA_OUTPUT:-results/full_conditioned_base_${SLURM_JOB_ID}.json}"

RUN_ARGS=()
if [[ -n "$LIMIT" ]]; then
  RUN_ARGS+=(--limit "$LIMIT")
fi

"$PYTHON" -m scripts.run_conditioned_medgemma \
  "${RUN_ARGS[@]}" \
  --output "$OUTPUT"
