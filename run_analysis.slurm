#!/bin/bash
#SBATCH --job-name=medgemma_analysis
#SBATCH --output=/data/jperez/projects/utils_medgemma/results/logs/analysis_%j.out
#SBATCH --error=/data/jperez/projects/utils_medgemma/results/logs/analysis_%j.err
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --propagate=NONE

set -euo pipefail

# Analisis post-hoc (extras NO contractuales): Recall@1 + diagnostico por clase
# sobre las generaciones ya evaluadas con M8. Es CPU puro (sin GPU). Pensado para
# encadenarse despues del base y del array LoRA, p.ej.:
#   B=$(sbatch --parsable run_medgemma_base_full.slurm)
#   L=$(sbatch --parsable run_experimental_medgemma_lora_full.slurm)
#   sbatch --dependency=afterok:$B:$L run_analysis.slurm

PROJECT_DIR=/data/jperez/projects/utils_medgemma
PYTHON=/data/jperez/app_data/miniconda/envs/medgemma/bin/python
cd "$PROJECT_DIR"
mkdir -p results/logs
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export XDG_CACHE_HOME=/data/jperez/app_data/cache
export HF_HUB_DISABLE_XET=1
mkdir -p "$MPLCONFIGDIR"

"$PYTHON" -m scripts.analyze_results
