#!/bin/bash
#SBATCH --job-name=medgemma_analysis
#SBATCH --output=/data/jperez/projects/medgemma_CIARP/results/logs/analysis_%j.out
#SBATCH --error=/data/jperez/projects/medgemma_CIARP/results/logs/analysis_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --propagate=NONE

set -euo pipefail

# Analisis post-hoc (extras NO contractuales): Recall@1 + diagnostico por clase
# sobre las generaciones ya evaluadas con M8. Es CPU puro (sin GPU). Corre el
# base-vs-LoRA por separado en cada fuente. Encadenar tras base+lora:
#   B=$(sbatch --parsable slurm/run_base.slurm)
#   L=$(sbatch --parsable slurm/run_lora.slurm)
#   sbatch --dependency=afterok:$B:$L slurm/run_analysis.slurm

PROJECT_DIR=/data/jperez/projects/medgemma_CIARP
PYTHON=/data/jperez/app_data/miniconda/envs/medgemma/bin/python
cd "$PROJECT_DIR"
mkdir -p results/logs
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export XDG_CACHE_HOME=/data/jperez/app_data/cache
export HF_HUB_DISABLE_XET=1
mkdir -p "$MPLCONFIGDIR"

# base vs LoRA en dataset (media +/- std sobre las 5 semillas)
"$PYTHON" -m scripts.analyze_results \
  --base results/medgemma_base_dataset.json \
  --lora-glob "results/medgemma_lora_dataset_seed*.json" \
  --output results/analysis_dataset.json

# base vs LoRA en pipeline (media +/- std sobre las 5 semillas)
"$PYTHON" -m scripts.analyze_results \
  --base results/medgemma_base_pipeline.json \
  --lora-glob "results/medgemma_lora_pipeline_seed*.json" \
  --output results/analysis_pipeline.json
