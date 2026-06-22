#!/bin/bash
#SBATCH --job-name=medgemma_lora_sweep
#SBATCH --output=/data/jperez/projects/utils_medgemma/results/sweep/logs/sweep_%A_%a.out
#SBATCH --error=/data/jperez/projects/utils_medgemma/results/sweep/logs/sweep_%A_%a.err
#SBATCH --partition=nukwa-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE
#SBATCH --array=0-5

set -euo pipefail

PROJECT_DIR=/data/jperez/projects/utils_medgemma
PYTHON=/data/jperez/app_data/miniconda/envs/medgemma/bin/python
cd "$PROJECT_DIR"
mkdir -p results/sweep/logs checkpoints/sweep
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export HF_HOME=/data/jperez/app_data/huggingface
export XDG_CACHE_HOME=/data/jperez/app_data/cache
export TORCH_HOME=/data/jperez/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

# Resolve this array task into one grid point (single source of truth).
eval "$("$PYTHON" -m scripts.sweep_grid --index "$SLURM_ARRAY_TASK_ID" --export)"
echo "Sweep idx=${SLURM_ARRAY_TASK_ID} tag=${SWEEP_TAG}"

"$PYTHON" -m scripts.preflight --require-cuda --require-hf-token

ADAPTER_DIR="checkpoints/sweep/idx${SLURM_ARRAY_TASK_ID}_${SWEEP_TAG}"
VAL_OUTPUT="results/sweep/val_idx${SLURM_ARRAY_TASK_ID}.json"

# Full dataset (no sample limits). Schedule expressed in epochs.
"$PYTHON" -m scripts.train_experimental_medgemma_lora \
  --split-file dataset/split.json \
  --prompt-mode "$PROMPT_MODE" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --learning-rate "$LEARNING_RATE" \
  --num-epochs "$NUM_EPOCHS" \
  --output-dir "$ADAPTER_DIR"

# Select hyperparameters on validation so test stays untouched.
"$PYTHON" -m scripts.run_conditioned_medgemma \
  --split-file dataset/split.json \
  --split-name validation \
  --adapter-path "$ADAPTER_DIR" \
  --output "$VAL_OUTPUT"
