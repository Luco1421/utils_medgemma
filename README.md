# Utils MedGemma

Flujo actual: MedGemma con el dataset oficial incluido en `dataset/`.

El trabajo actual corre sobre los splits `dataset/split_repetition_*.json`.

## Contrato de referencia

Los PDFs de `ref/` no definen el contrato de M7:

- `Glaucoma_labels.pdf` documenta las etiquetas clínicas.
- `A_Benchmark_of_Eye_Anterior_Image_Semantic_Segmentation_in_Scarcely_Labeled_Datasets.pdf`
  es el paper de benchmark de segmentación.

El contrato de M7 está en `ref/.../experimental_design.md`,
`AGENTS.md` y `Pipeline Modules Explanation/M7_MedGemmaConditioner.md`. Es una
interfaz Python en memoria:

```python
generate(
    condition,
    image_raw,
    mask=None,
    prediction=None,
    distribution=None,
)
```

La implementación pública compatible está en
`modules/medgemma_conditioner.py`.

## Implementacion Python

La implementación para integración está en Python:

- `modules/medgemma_conditioner.py`: interfaz pública M7 indicada por `ref`.
- `medgemma_utils/medgemma_conditioner.py`: carga y generación con MedGemma.
- `medgemma_utils/conditioning.py`: prompts y overlays A-D2.
- `medgemma_utils/evaluation.py`: BERTScore médico y comparaciones.
- `medgemma_utils/dataset.py`: adaptación del dataset compartido.
- `medgemma_utils/lora_training.py`: entrenamiento LoRA estándar.
- `medgemma_utils/experiment.py`: ejecución completa A-D2.
- `scripts/run_conditioned_medgemma.py`: ejecución sin notebook.
- `scripts/train_medgemma_lora.py`: entrenamiento sin notebook.

La clase que consumirá el orquestador del proyecto real es:

```python
from modules.medgemma_conditioner import MedGemmaConditioner

conditioner = MedGemmaConditioner(config["medgemma"])
result = conditioner.generate(
    condition="D2",
    image_raw=image,
    mask=mask,
    distribution=classification["distribution"],
)
```

## Alcance actual

Este repositorio implementa y prueba solamente el componente MedGemma. La
clasificacion, las mascaras segmentadas y las distribuciones de probabilidad
seran producidas por los otros modulos del pipeline al integrarlos.

Mientras esos modulos no esten conectados, la ablation usa:

- Las mascaras ground truth del disco optico como entrada visual de prueba.
- Una distribucion mock 80/20 derivada de la etiqueta del dataset.

Estos valores simulados sirven para validar la interfaz de MedGemma; no deben
reportarse como resultados del clasificador o segmentador.

## Separacion entre entrada y experimento

La logica se divide en tres archivos:

- `medgemma_utils/conditioning.py`: M7, condiciones A-D2, prompts y overlay.
- `medgemma_utils/evaluation.py`: BERTScore medico y comparaciones
  estadisticas.
- `medgemma_utils/mock_inputs.py`: adapta el dataset actual a entradas mock
  80/20.
- `medgemma_utils/inputs.py`: define el contrato comun y permite cargar
  entradas reales desde JSON.

El experimento no busca probabilidades dentro de las anotaciones ni mezcla
fuentes. Por defecto usa el proveedor mock:

```python
from medgemma_utils.mock_inputs import build_dataset_mock_inputs
```

Cuando existan resultados reales no hay que cambiar el notebook. Se indica el
JSON de entrada:

```bash
export MEDGEMMA_INPUTS_JSON=outputs/pipeline_inputs.json
```

El JSON real esperado es una lista con este contrato:

```json
[
  {
    "image_id": "1217_right",
    "image": "dataset/1217/1217_right.jpg",
    "mask": "outputs/wsss/1217_right.npy",
    "prediction": "glaucoma",
    "distribution": {
      "glaucoma": 0.73,
      "non_glaucoma": 0.27
    },
    "reference": "Clinical photograph...",
    "mask_source": "wsss",
    "input_source": "real_pipeline"
  }
]
```

Hay un archivo de ejemplo en `examples/pipeline_inputs.example.json`.

El proveedor mock usa la etiqueta real y sirve para estudiar el efecto del
condicionamiento, no para medir rendimiento predictivo. Su confianza se puede
cambiar con:

```bash
export MEDGEMMA_MOCK_CONFIDENCE=0.80
```

Si `MEDGEMMA_INPUTS_JSON` no está definido se usan mocks. Si está definido se
usa exclusivamente ese JSON.

## Evaluacion textual

BERTScore usa:

- Modelo: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`.
- Capa: 12.
- Scores crudos, sin `rescale_with_baseline`, porque BERTScore no incluye una
  baseline precalculada para este checkpoint.

Cada condicion guarda Precision, Recall y F1 por imagen. El resumen reporta
media, desviacion estandar y diferencia de F1 respecto a la condicion A.
Tambien realiza comparaciones pareadas por imagen contra A: delta medio y
mediano, fraccion de casos que mejoran y test de Wilcoxon.

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

1. Corre `scripts/run_conditioned_medgemma.py`.
2. Revisa los JSON en `results/` para clasificacion, descripcion y ablation.
3. Si el flujo base esta bien, corre `scripts/train_medgemma_lora.py` y luego `scripts/run_conditioned_medgemma.py` con `--adapter-path`.
4. Para una prueba rapida usa los jobs `smoke`; para corrida completa usa los jobs `full`.

En modo `full`, el experimento condicionado usa todo el conjunto de test. Para
limitarlo manualmente:

```bash
export MEDGEMMA_ABLATION_LIMIT=3
```

## Ejecucion

Los scripts asumen que el repo ya está clonado y que el entorno de Python/GPU ya fue preparado por el cluster o la maquina local.

En Kabre, lo esperado es entrar al repo, activar el entorno que tenga CUDA/PyTorch y lanzar el job Slurm correspondiente.

Los jobs de Kabre guardan modelos y cachés fuera de `home`:

```bash
HF_HOME=/data/jperez/app_data/huggingface
XDG_CACHE_HOME=/data/jperez/app_data/cache
TORCH_HOME=/data/jperez/app_data/cache/torch
```

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
$env:MEDGEMMA_ABLATION_LIMIT="1"
.\.venv\Scripts\python.exe -m scripts.run_conditioned_medgemma --limit 1 --output results\local_smoke_base.json
```

Smoke de LoRA con un ejemplo y un step:

```bash
$env:HF_TOKEN="..."
$env:MEDGEMMA_RUN_MODE="smoke"
$env:MEDGEMMA_TRAIN_LIMIT="1"
$env:MEDGEMMA_EVAL_LIMIT="1"
$env:MEDGEMMA_MAX_STEPS="1"
$env:MEDGEMMA_GRAD_ACCUM="1"
.\.venv\Scripts\python.exe -m scripts.train_medgemma_lora --train-limit 1 --max-steps 1 --gradient-accumulation-steps 1 --output-dir checkpoints\local_smoke_lora
.\.venv\Scripts\python.exe -m scripts.run_conditioned_medgemma --limit 1 --adapter-path checkpoints\local_smoke_lora --output results\local_smoke_lora.json
```

Los scripts Slurm incluidos asumen un entorno conda llamado `medgemma`:

```bash
conda activate medgemma
```

Tambien esperan que el token de Hugging Face este disponible como variable de entorno:

```bash
export HF_TOKEN=...
```

Los jobs Slurm fuerzan:

```bash
export MEDGEMMA_RUN_MODE=smoke  # o full, segun el job
```
