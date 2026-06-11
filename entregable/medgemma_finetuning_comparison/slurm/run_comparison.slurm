#!/bin/bash
#SBATCH --job-name=medgemma_lora_cmp
#SBATCH --output=results/slurm_medgemma_lora_cmp_%j.out
#SBATCH --error=results/slurm_medgemma_lora_cmp_%j.err
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=1-24:00:00
#SBATCH --nodelist=nukwa-03.cnca

set -euo pipefail

PROJECT_DIR=/data/jperez/projects/utils_medgemma/entregable/medgemma_finetuning_comparison
cd "$PROJECT_DIR"
mkdir -p results checkpoints

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medgemma

python - <<'PY'
import logging
import sys
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.info("Python: %s", sys.version)
logging.info("Torch: %s", torch.__version__)
logging.info("CUDA available: %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logging.info("CUDA device: %s", torch.cuda.get_device_name(0))
PY

python scripts/run_comparison.py --config config.yaml
