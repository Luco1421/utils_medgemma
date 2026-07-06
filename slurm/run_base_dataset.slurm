#!/bin/bash
#SBATCH --job-name=medgemma_base_ds
#SBATCH --output=/data/jperez/projects/medgemma_CIARP/results/logs/base_ds_%j.out
#SBATCH --error=/data/jperez/projects/medgemma_CIARP/results/logs/base_ds_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca
#SBATCH --propagate=NONE

set -euo pipefail

# Solo la fuente DATASET (control de evidencia perfecta). El overlay del oraculo
# se compone desde la mascara GT con make_overlay al 50% (OVERLAY_RED_ALPHA),
# igualando el 50% de los overlays que el modulo de segmentacion entrego para el
# pipeline. La fuente PIPELINE NO se re-corre: consume esos overlays entregados,
# asi que sus resultados no cambian.

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

"$PYTHON" -m scripts.run_conditioned_medgemma \
  --split-file dataset/split.json \
  --split-name test \
  --output results/medgemma_base_dataset.json
