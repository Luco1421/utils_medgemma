#!/bin/bash
#SBATCH --job-name=medgemma_lora_full
#SBATCH --output=/data/jperez/projects/utils_medgemma/results/slurm_medgemma_lora_full_%j.out
#SBATCH --error=/data/jperez/projects/utils_medgemma/results/slurm_medgemma_lora_full_%j.err
#SBATCH --partition=nukwa-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
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

TRAIN_LIMIT="${MEDGEMMA_TRAIN_LIMIT:-}"
EVAL_LIMIT="${MEDGEMMA_LIMIT:-}"
MAX_STEPS="${MEDGEMMA_MAX_STEPS:-}"
GRAD_ACCUM="${MEDGEMMA_GRAD_ACCUM:-}"
ADAPTER_DIR="${MEDGEMMA_ADAPTER_DIR:-checkpoints/full_medgemma_lora_${SLURM_JOB_ID}}"
OUTPUT="${MEDGEMMA_OUTPUT:-results/full_conditioned_lora_${SLURM_JOB_ID}.json}"

TRAIN_ARGS=()
if [[ -n "$TRAIN_LIMIT" ]]; then
  TRAIN_ARGS+=(--train-limit "$TRAIN_LIMIT")
fi
if [[ -n "$MAX_STEPS" ]]; then
  TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$GRAD_ACCUM" ]]; then
  TRAIN_ARGS+=(--gradient-accumulation-steps "$GRAD_ACCUM")
fi

RUN_ARGS=()
if [[ -n "$EVAL_LIMIT" ]]; then
  RUN_ARGS+=(--limit "$EVAL_LIMIT")
fi

"$PYTHON" -m scripts.train_medgemma_lora \
  "${TRAIN_ARGS[@]}" \
  --output-dir "$ADAPTER_DIR"

"$PYTHON" -m scripts.run_conditioned_medgemma \
  "${RUN_ARGS[@]}" \
  --adapter-path "$ADAPTER_DIR" \
  --output "$OUTPUT"
