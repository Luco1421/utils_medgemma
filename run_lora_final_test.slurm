#!/bin/bash
#SBATCH --job-name=medgemma_lora_final_test
#SBATCH --output=/data/jperez/projects/utils_medgemma/results/final/logs/final_%A_%a.out
#SBATCH --error=/data/jperez/projects/utils_medgemma/results/final/logs/final_%A_%a.err
#SBATCH --partition=nukwa-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE
#SBATCH --array=0-4

set -euo pipefail

# Which sweep winner to confirm on test: overall | generic | conditioned.
# Override at submit time, e.g.:
#   sbatch --export=ALL,SELECTOR=conditioned run_lora_final_test.slurm
SELECTOR="${SELECTOR:-overall}"

PROJECT_DIR=/data/jperez/projects/utils_medgemma
PYTHON=/data/jperez/app_data/miniconda/envs/medgemma/bin/python
cd "$PROJECT_DIR"
mkdir -p results/final/logs checkpoints/final
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export HF_HOME=/data/jperez/app_data/huggingface
export XDG_CACHE_HOME=/data/jperez/app_data/cache
export TORCH_HOME=/data/jperez/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

# Resolve the winning hyperparameters from the validation sweep.
eval "$("$PYTHON" -m scripts.select_config --selector "$SELECTOR" --export)"
SEED=$((42 + SLURM_ARRAY_TASK_ID))
echo "Final test selector=${SELECTOR} tag=${BEST_TAG} seed=${SEED}"

"$PYTHON" -m scripts.preflight --require-cuda --require-hf-token

ADAPTER_DIR="checkpoints/final/${SELECTOR}_seed${SEED}"
OUTPUT="results/final/test_${SELECTOR}_seed${SEED}.json"

"$PYTHON" -m scripts.train_experimental_medgemma_lora \
  --split-file dataset/split.json \
  --prompt-mode "$PROMPT_MODE" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --learning-rate "$LEARNING_RATE" \
  --num-epochs "$NUM_EPOCHS" \
  --seed "$SEED" \
  --output-dir "$ADAPTER_DIR"

"$PYTHON" -m scripts.run_conditioned_medgemma \
  --split-file dataset/split.json \
  --split-name test \
  --adapter-path "$ADAPTER_DIR" \
  --output "$OUTPUT"
