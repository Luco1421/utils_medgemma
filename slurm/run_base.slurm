#!/bin/bash
#SBATCH --job-name=medgemma_base
#SBATCH --output=/data/jperez/projects/medgemma_CIARP/results/logs/base_%j.out
#SBATCH --error=/data/jperez/projects/medgemma_CIARP/results/logs/base_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# nukwa-00..03 tienen V100 (sm_70); el torch del env (2.11+cu128) ya no soporta Volta
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

set -euo pipefail

# Base MedGemma (sin LoRA) evaluado en las DOS fuentes:
#   - dataset:  etiquetas GT + overlay pre-renderizado del disco GT
#   - pipeline: clasificador real + overlay del segmentador (SAM)
# Base es greedy/deterministico (do_sample=False) -> una corrida por fuente.

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

# base x dataset
"$PYTHON" -m scripts.run_conditioned_medgemma \
  --split-file dataset/split.json \
  --split-name test \
  --output results/medgemma_base_dataset.json

# base x pipeline
"$PYTHON" -m scripts.run_conditioned_medgemma \
  --pipeline-dir dataset/test \
  --pipeline-classifications-json dataset/pipeline_classifications.json \
  --output results/medgemma_base_pipeline.json
