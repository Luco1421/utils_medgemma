#!/bin/bash
#SBATCH --job-name=medgemma_lora
#SBATCH --output=/data/jperez/projects/utils_medgemma/results/slurm_medgemma_lora_%j.out
#SBATCH --error=/data/jperez/projects/utils_medgemma/results/slurm_medgemma_lora_%j.err
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=1-24:00:00
#SBATCH --nodelist=nukwa-03.cnca

set -euo pipefail

PROJECT_DIR=/data/jperez/projects/utils_medgemma
cd "$PROJECT_DIR"
mkdir -p results

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medgemma

python - <<'PY'
import sys
import torch
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY

python -m jupyter nbconvert \
  --to notebook \
  --execute notebooks/medgemma_lora.ipynb \
  --output-dir results \
  --output medgemma_lora_executed.ipynb
