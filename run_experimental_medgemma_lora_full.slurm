#!/bin/bash
#SBATCH --job-name=experimental_medgemma_lora
#SBATCH --output=/data/jperez/projects/utils_medgemma/results/lora/logs/lora_%A_%a.out
#SBATCH --error=/data/jperez/projects/utils_medgemma/results/lora/logs/lora_%A_%a.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --propagate=NONE
#SBATCH --array=0-2

set -euo pipefail

# Experimento extra (no contractual): UNA config justa de LoRA generico,
# repetida con 3 semillas (42,43,44) para reportar media +/- std. Se compara
# contra el job base (sin LoRA) con el Wilcoxon pareado de M8. No hay seleccion
# de hiperparametros, asi que evaluar directo en test es legitimo.
#
# Config justa: rank=8, alpha=16 (alpha=2*rank), lr=1e-4, 3 epocas.

PROJECT_DIR=/data/jperez/projects/utils_medgemma
PYTHON=/data/jperez/app_data/miniconda/envs/medgemma/bin/python
cd "$PROJECT_DIR"
mkdir -p results/lora/logs checkpoints/lora
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export HF_HOME=/data/jperez/app_data/huggingface
export XDG_CACHE_HOME=/data/jperez/app_data/cache
export TORCH_HOME=/data/jperez/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
export WANDB_PROJECT="${WANDB_PROJECT:-utils-medgemma}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/results/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PROJECT_DIR/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PROJECT_DIR/.cache/wandb_config}"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

SEED=$((42 + SLURM_ARRAY_TASK_ID))
echo "LoRA generico seed=${SEED}"

ADAPTER_DIR="checkpoints/lora/seed${SEED}"
OUTPUT="results/lora/test_lora_seed${SEED}.json"

"$PYTHON" -m scripts.train_experimental_medgemma_lora \
  --split-file dataset/split.json \
  --lora-rank 8 \
  --lora-alpha 16 \
  --learning-rate 1e-4 \
  --num-epochs 3 \
  --seed "$SEED" \
  --wandb-run-name "lora_generic_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  --wandb-tags lora generic "seed${SEED}" \
  --output-dir "$ADAPTER_DIR"

"$PYTHON" -m scripts.run_conditioned_medgemma \
  --split-file dataset/split.json \
  --split-name test \
  --adapter-path "$ADAPTER_DIR" \
  --output "$OUTPUT"
