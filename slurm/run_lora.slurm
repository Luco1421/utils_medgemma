#!/bin/bash
#SBATCH --job-name=medgemma_lora
#SBATCH --output=/data/jperez/projects/medgemma_CIARP/results/logs/lora_%A_%a.out
#SBATCH --error=/data/jperez/projects/medgemma_CIARP/results/logs/lora_%A_%a.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# nukwa-00..03 tienen V100 (sm_70); el torch del env (2.11+cu128) ya no soporta Volta
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE
#SBATCH --array=0-4

set -euo pipefail

# LoRA generico, 5 semillas (42,123,456,789,1024) para reportar media +/- std.
#
# ENTRENAMIENTO: siempre con GT (train split, prompt generico, imagen->transcripcion,
# sin mascaras ni condicionamiento). El fine-tuning no ve el pipeline.
# EVALUACION: en las dos fuentes (dataset y pipeline) con el mismo adapter, para
# separar "cuanto aporta el LoRA" de "cuanto degrada el pipeline real".
#
# Config: rank=8, alpha=16 (alpha=2*rank), lr=1e-4, 3 epocas.

PROJECT_DIR=/data/jperez/projects/medgemma_CIARP
PYTHON=/data/jperez/app_data/miniconda/envs/medgemma/bin/python
cd "$PROJECT_DIR"
mkdir -p results/logs checkpoints/lora
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

SEEDS=(42 123 456 789 1024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "LoRA generico seed=${SEED}"
ADAPTER_DIR="checkpoints/lora/seed${SEED}"

# 1) Entrenamiento (GT)
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

# 2) Evaluacion lora x dataset
"$PYTHON" -m scripts.run_conditioned_medgemma \
  --split-file dataset/split.json \
  --split-name test \
  --adapter-path "$ADAPTER_DIR" \
  --output "results/medgemma_lora_dataset_seed${SEED}.json"

# 3) Evaluacion lora x pipeline
"$PYTHON" -m scripts.run_conditioned_medgemma \
  --pipeline-dir dataset/test \
  --pipeline-classifications-json dataset/pipeline_classifications.json \
  --adapter-path "$ADAPTER_DIR" \
  --output "results/medgemma_lora_pipeline_seed${SEED}.json"
