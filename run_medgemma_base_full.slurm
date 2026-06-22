#!/bin/bash
#SBATCH --job-name=medgemma_base_full
#SBATCH --output=/data/jperez/projects/utils_medgemma/results/baseline/logs/base_%j.out
#SBATCH --error=/data/jperez/projects/utils_medgemma/results/baseline/logs/base_%j.err
#SBATCH --partition=nukwa-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE

set -euo pipefail

PROJECT_DIR=/data/jperez/projects/utils_medgemma
PYTHON=/data/jperez/app_data/miniconda/envs/medgemma/bin/python
cd "$PROJECT_DIR"
mkdir -p results/baseline/logs
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export HF_HOME=/data/jperez/app_data/huggingface
export XDG_CACHE_HOME=/data/jperez/app_data/cache
export TORCH_HOME=/data/jperez/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

"$PYTHON" -m scripts.preflight --require-cuda --require-hf-token

OUTPUT="results/baseline/conditioned_base.json"

"$PYTHON" -m scripts.run_conditioned_medgemma \
  --split-file dataset/split.json \
  --output "$OUTPUT"
