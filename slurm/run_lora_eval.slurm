#!/bin/bash
#SBATCH --job-name=medgemma_lora_ds
#SBATCH --output=/data/jperez/projects/medgemma_CIARP/results/logs/lora_ds_%A_%a.out
#SBATCH --error=/data/jperez/projects/medgemma_CIARP/results/logs/lora_ds_%A_%a.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE
#SBATCH --array=0-4

set -euo pipefail

# Re-evaluacion LoRA en la fuente DATASET (5 semillas) REUSANDO los adapters ya
# entrenados en checkpoints/lora/seed*. NO reentrena: solo regenera las
# descripciones con el overlay del oraculo unificado a 50%. La fuente PIPELINE no
# se re-corre (usa los overlays entregados por el modulo de segmentacion, sin
# cambios). El adapter se entrena sobre imagen cruda + prompt generico, asi que
# el cambio de overlay no lo afecta, solo la evaluacion.

PROJECT_DIR=/data/jperez/projects/medgemma_CIARP
PYTHON=/data/jperez/app_data/miniconda/envs/medgemma/bin/python
cd "$PROJECT_DIR"
mkdir -p results/logs
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export HF_HOME=/data/jperez/app_data/huggingface
export XDG_CACHE_HOME=/data/jperez/app_data/cache
export TORCH_HOME=/data/jperez/app_data/cache/torch
export HF_HUB_DISABLE_XET=1
mkdir -p "$MPLCONFIGDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME"

SEEDS=(42 123 456 789 1024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ADAPTER_DIR="checkpoints/lora/seed${SEED}"
echo "LoRA eval-only (dataset) seed=${SEED} (adapter=${ADAPTER_DIR})"

"$PYTHON" -m scripts.run_conditioned_medgemma \
  --split-file dataset/split.json \
  --split-name test \
  --adapter-path "$ADAPTER_DIR" \
  --output "results/medgemma_lora_dataset_seed${SEED}.json"
