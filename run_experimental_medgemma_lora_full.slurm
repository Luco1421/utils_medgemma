#!/bin/bash
#SBATCH --job-name=experimental_medgemma_lora_full
#SBATCH --output=/data/jperez/projects/utils_medgemma/results/slurm_experimental_medgemma_lora_full_%j.out
#SBATCH --error=/data/jperez/projects/utils_medgemma/results/slurm_experimental_medgemma_lora_full_%j.err
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
mkdir -p results
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export HF_HOME=/data/jperez/app_data/huggingface
export XDG_CACHE_HOME=/data/jperez/app_data/cache
export TORCH_HOME=/data/jperez/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

"$PYTHON" -m scripts.preflight --require-cuda --require-hf-token

ADAPTER_DIR="checkpoints/experimental_full_medgemma_lora_${SLURM_JOB_ID}"
OUTPUT="results/experimental_full_medgemma_lora_${SLURM_JOB_ID}.json"

"$PYTHON" -m scripts.train_experimental_medgemma_lora \
  --split-file dataset/split.json \
  --output-dir "$ADAPTER_DIR"

"$PYTHON" -m scripts.run_conditioned_medgemma \
  --split-file dataset/split.json \
  --adapter-path "$ADAPTER_DIR" \
  --output "$OUTPUT"
