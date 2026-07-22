# utils_medgemma

Research code for conditioning MedGemma to generate clinical descriptions of
glaucoma fundus (retina) photographs. `google/medgemma-1.5-4b-it` is used as a
frozen black box (base weights are never modified for inference) with
deterministic generation (`do_sample=False`, up to 512 new tokens), and the
generated text is evaluated under different ways of conditioning the input
image and prompt. A separate, clearly-labeled experimental LoRA fine-tune is
also included.

## Conditioning setup

Each image is described under six conditions, defined as static prompts in
`dataset/prompts.json` and combined along two axes:

- **Overlay**: the image is sent as-is, or with the optic-disc mask composited
  in red (50% image + 50% red inside the mask, `OVERLAY_RED_ALPHA` in
  `medgemma_utils/conditioning.py`).
- **Class-conditioned prompt**: glaucoma, normal, or unspecified.

| Condition | Overlay | Prompt |
|---|---|---|
| `with_overlay_glaucoma` | yes | glaucoma |
| `without_overlay_glaucoma` | no | glaucoma |
| `with_overlay_normal` | yes | normal |
| `without_overlay_normal` | no | normal |
| `with_overlay` | yes | unspecified |
| `without_overlay` | no | unspecified |

`without_overlay` (raw image, no class hint) is the fixed **baseline** used
for all paired comparisons.

## Evaluation sources

The six conditions are run over the same input structure, fed from two
sources:

- **dataset** (`--split-name test`): the label comes from ground truth and the
  overlay is pre-rendered from the GT disc mask. Normal cases have no disc
  mask, so this source has no overlay for normals.
- **pipeline** (`--pipeline-dir dataset/test`): the label comes from a real
  upstream classifier and the overlay from a SAM-based segmentor. All 26 test
  images ship with an overlay, so all six conditions are populated.

In the pipeline source, both the classifier prediction (`y_pred`) and the GT
label it is judged against (`y_true`) come from the same file,
`dataset/pipeline_classifications.json`, for traceability. Reference text
always comes from `dataset/annotations.json`.

## 2×2 design

The experiment crosses **model × evaluation source**:

|          | dataset | pipeline |
|----------|---------|----------|
| **base** | `medgemma_base_dataset.json` | `medgemma_base_pipeline.json` |
| **LoRA** | `medgemma_lora_dataset_seed*.json` | `medgemma_lora_pipeline_seed*.json` |

This separates two effects: how much the fine-tune helps (base → LoRA) and how
much quality drops going from ideal to real pipeline inputs (dataset →
pipeline). The base model is deterministic, so one run per source is enough;
LoRA is repeated over five seeds (42, 123, 456, 789, 1024) to report a
mean ± std.

## Dataset

`dataset/` holds the flat dataset (`annotations.json`, `split.json`, images
and `*_disc` / `*_cup` masks per glaucoma case), the pipeline-style test set
in `dataset/test/` (image + segmentor mask `*_obj_0.png` + pre-generated
overlays), the prompts (`prompts.json`), and the pipeline classifications
(`pipeline_classifications.json`).

| Split | Total | Glaucoma | Normal |
|---|---:|---:|---:|
| Train | 77 | 41 | 36 |
| Validation | 26 | 14 | 12 |
| Test | 26 | 14 | 12 |

The optic disc is the only mask used for conditioning; cup masks exist as
additional data but do not take part in evaluation.

## Evaluation

`medgemma_utils/evaluation.py` (`Evaluator`) measures, for every generated
text:

- **BERTScore** F1 with BiomedBERT and **sBERT similarity** with Bioclinical
  ModernBERT.
- **Calibrated** versions of both: the score against the true reference minus
  the score against shuffled references, to discount generic report
  boilerplate (every report mentions disc, cup-to-disc ratio, rim, ...).
- `finding_mentioned`: a heuristic that checks whether the text mentions terms
  consistent with the expected class.

Each condition is compared against the baseline with a paired Wilcoxon test
over the same images, with effect size and Holm correction for multiple
comparisons.

`scripts/analyze_results.py` adds a CPU-only post-hoc analysis (Recall@1 and
negation-aware, class-level diagnosis) that surfaces class collapse hidden by
the similarity metrics. It writes `results/analysis_dataset.json` and
`results/analysis_pipeline.json`.

## MedGemma-LoRA

A separate, standard LoRA (not QLoRA) experiment on top of MedGemma. **It is
trained only on the train split using ground truth**: raw image + generic
prompt → expert transcription. It never sees masks, classes, or conditioning,
so it can learn report style but not diagnosis. Evaluation reuses the same
six-condition protocol on both sources. With the current LoRA, similarity
metrics go up while Recall@1 stays at chance and recall on the normal class
drops — a sign of template memorization rather than learned diagnosis.

Config: `target_modules="all-linear"`, rank 8, alpha 16, lr 1e-4, 3 epochs,
tracked with Weights & Biases.

## Installation

```bash
pip install -r requirements.txt
```

Requires a CUDA 12.8 GPU (`torch==2.11.0+cu128`, installed from PyTorch's own
wheel index) — both generation and LoRA training raise at startup if no CUDA
device is available. `google/medgemma-1.5-4b-it` is a gated model on Hugging
Face, so a valid `HF_TOKEN` (or a cached `huggingface-cli login`) is required.
Run hyperparameters (model name, dtype, eval batch size, LoRA config) live in
`config.yaml` and are validated by `medgemma_utils/config.py`.

## Running

```bash
# Base MedGemma, oracle (dataset) source
python -m scripts.run_conditioned_medgemma --config config.yaml \
  --split-file dataset/split.json --split-name test \
  --output results/medgemma_base_dataset.json

# Pipeline source instead of oracle
python -m scripts.run_conditioned_medgemma --config config.yaml \
  --pipeline-dir dataset/test \
  --pipeline-classifications-json dataset/pipeline_classifications.json \
  --output results/medgemma_base_pipeline.json

# Evaluate a trained LoRA adapter instead of the base model
python -m scripts.run_conditioned_medgemma --config config.yaml \
  --adapter-path checkpoints/lora/seed42 --output results/medgemma_lora_dataset_seed42.json

# Train a LoRA adapter
python -m scripts.train_experimental_medgemma_lora --split-file dataset/split.json \
  --lora-rank 8 --lora-alpha 16 --learning-rate 1e-4 --num-epochs 3 --seed 42 \
  --output-dir checkpoints/lora/seed42

# CPU-only post-hoc analysis
python -m scripts.analyze_results --base results/medgemma_base_dataset.json \
  --lora-glob "results/medgemma_lora_dataset_seed*.json" \
  --output results/analysis_dataset.json

# Blinded CSV for a human Likert review
python -m scripts.export_likert_review --input results/medgemma_base_dataset.json \
  --review-csv review.csv --key-json review_key.json
```

## Running on Kabre (SLURM)

Jobs run on the `nukwa-long` partition, chained via job dependencies:

```bash
B=$(sbatch --parsable slurm/run_base.slurm)     # base × {dataset, pipeline}
L=$(sbatch --parsable slurm/run_lora.slurm)      # LoRA: 5 seeds, train + eval
sbatch --dependency=afterok:$B:$L slurm/run_analysis.slurm
```

These scripts hardcode cluster-specific paths and exclude older GPU nodes
(Volta / sm_70) that the pinned CUDA 12.8 torch build no longer supports —
they are not portable outside that cluster as-is.

Training curves (loss, learning rate, timing) are tracked on W&B:
<https://wandb.ai/byluco1421-tecnol-gico-de-costa-rica/utils-medgemma>. To run
without uploading: `sbatch --export=ALL,WANDB_MODE=disabled slurm/run_lora.slurm`.

## Results

The JSON files in `results/` include, besides the raw generations
(`results`), a per-condition summary with deltas against the baseline
(`summary_by_condition`), the paired comparisons
(`paired_comparisons_vs_baseline`), and `dataset_audit` / `runtime` sections
recording data provenance and run timings.
