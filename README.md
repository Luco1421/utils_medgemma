# Utils MedGemma

Flujo actual: MedGemma con el dataset oficial incluido en `dataset/`.

El trabajo actual corre sobre los splits `dataset/split_repetition_*.json`.

## Notebooks principales

- `notebooks/medgemma_base.ipynb`: evalua MedGemma base sin fine-tuning.
  - Clasificacion zero-shot `glaucoma` vs `non_glaucoma`.
  - Descripcion clinica y evaluacion con BERTScore contra `transcription`.
  - Ablation con mascara real de disco/copa para comparar comportamiento visual.

- `notebooks/medgemma_lora.ipynb`: entrena LoRA descriptivo sobre el dataset oficial.
  - Evalua MedGemma base antes de entrenar.
  - Entrena LoRA usando las transcripciones oficiales.
  - Compara BERTScore base vs LoRA.

## Mapeo del dataset

Cada entrada de split tiene rutas relativas al dataset:

- `image`: imagen de fondo de ojo.
- `annotation`: JSON con `label`, `transcription` y `locs_data.conditions`.
- `mask_cup_npy` / `mask_disc_npy`: mascaras reales para copa y disco.

La etiqueta medible se deriva asi:

- `glaucoma` si `locs_data.conditions` contiene `glaucoma`.
- `non_glaucoma` en caso contrario.

La descripcion medible usa `transcription` como referencia textual.

## Orden recomendado

1. Corre `notebooks/medgemma_base.ipynb`.
2. Revisa los JSON en `results/` para clasificacion, descripcion y ablation.
3. Si el flujo base esta bien, abre un runtime limpio y corre `notebooks/medgemma_lora.ipynb`.
4. Para una prueba rapida usa `MEDGEMMA_RUN_MODE=smoke`; para Kabre usa `MEDGEMMA_RUN_MODE=full`.

## Colab vs Kabre

Los notebooks detectan si estan corriendo en Colab:

- En Colab clonan el repo e instalan `requirements.txt`.
- Fuera de Colab asumen que el repo ya esta clonado y que el entorno de Python/GPU ya fue preparado por el cluster.

En Kabre, lo esperado es entrar al repo, activar el entorno que tenga CUDA/PyTorch y ejecutar Jupyter desde la raiz del proyecto.

## Prueba local

Para probar en una maquina local con GPU NVIDIA:

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Smoke de MedGemma base con un ejemplo:

```bash
$env:HF_TOKEN="..."
$env:MEDGEMMA_RUN_MODE="smoke"
$env:MEDGEMMA_EVAL_LIMIT="1"
$env:MEDGEMMA_ABLATION_LIMIT="0"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks\medgemma_base.ipynb --output-dir results --output local_smoke_base.ipynb --ExecutePreprocessor.timeout=2400
```

Smoke de LoRA con un ejemplo y un step:

```bash
$env:HF_TOKEN="..."
$env:MEDGEMMA_RUN_MODE="smoke"
$env:MEDGEMMA_TRAIN_LIMIT="1"
$env:MEDGEMMA_EVAL_LIMIT="1"
$env:MEDGEMMA_MAX_STEPS="1"
$env:MEDGEMMA_GRAD_ACCUM="1"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks\medgemma_lora.ipynb --output-dir results --output local_smoke_lora.ipynb --ExecutePreprocessor.timeout=2400
```

Los scripts Slurm incluidos asumen un entorno conda llamado `medgemma`:

```bash
conda activate medgemma
```

Tambien esperan que el token de Hugging Face este disponible como variable de entorno:

```bash
export HF_TOKEN=...
```

Por defecto los notebooks usan modo `smoke` en Colab y modo `full` fuera de Colab. En Kabre los scripts Slurm fuerzan:

```bash
export MEDGEMMA_RUN_MODE=full
```
