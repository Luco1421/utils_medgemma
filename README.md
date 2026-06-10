# Utils MedGemma

Flujo actual: MedGemma con el dataset oficial incluido en `dataset/`.

El trabajo actual corre sobre los splits `dataset/split_repetition_*.json`.

## Notebooks principales

- `notebooks/medgemma_base.ipynb`: evalua MedGemma base sin fine-tuning.
  - Clasificacion zero-shot `glaucoma` vs `non_glaucoma`.
  - Descripcion clinica y evaluacion con BERTScore contra `transcription`.
  - Ablation con mascara real de disco/copa para comparar comportamiento visual.

- `notebooks/medgemma_lora.ipynb`: entrena LoRA/QLoRA descriptivo sobre el dataset oficial.
  - Evalua baseline 4-bit antes de entrenar.
  - Entrena LoRA/QLoRA usando las transcripciones oficiales.
  - Compara BERTScore base vs QLoRA.

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
4. Para una prueba rapida de QLoRA, baja `TRAIN_LIMIT` y `max_steps`; para un experimento serio, usa todo el train split.

## Colab vs Kabre

Los notebooks detectan si estan corriendo en Colab:

- En Colab clonan el repo e instalan `requirements.txt`.
- Fuera de Colab asumen que el repo ya esta clonado y que el entorno de Python/GPU ya fue preparado por el cluster.

En Kabre, lo esperado es entrar al repo, activar el entorno que tenga CUDA/PyTorch y ejecutar Jupyter desde la raiz del proyecto.

Los scripts Slurm incluidos asumen un entorno conda llamado `medgemma`:

```bash
conda activate medgemma
```

Tambien esperan que el token de Hugging Face este disponible como variable de entorno:

```bash
export HF_TOKEN=...
```
