# Utils MedGemma

Repositorio de apoyo para la parte de **Pablo** en el proyecto:

1. correr MedGemma v1.5 4B pretrained en Colab;
2. probar inferencia con texto, imagen, e imagen + texto;
3. ejecutar las 6 condiciones del pipeline mejorado;
4. preparar entrenamiento LoRA/QLoRA;
5. comparar baseline vs adapter LoRA.

La referencia del proyecto completo esta en:

```text
ref/Medgemma_Segmentation_CIARP_2026/pipeline_mejorado_informe.md
```

## Idea General

Hay dos carriles:

### 1. Baseline Pretrained

Se carga `google/medgemma-1.5-4b-it` sin entrenamiento adicional y se le hacen
preguntas. Esto responde: "Que puede hacer MedGemma tal como viene?".

### 2. LoRA / QLoRA

Se entrena un adapter pequeno sobre MedGemma usando pares:

```json
{"image": "fundus.jpg", "prompt": "Describe...", "answer": "Clinical photograph..."}
```

Esto responde: "Mejora MedGemma si lo adaptamos al lenguaje/descripciones de
nuestro dominio?".

## Estructura

```text
modules/
  medgemma_runtime.py        # carga MedGemma y genera texto
  medgemma_conditioner.py    # prepara condiciones A/B/C1/C2/D1/D2
experiments/
  smoke_medgemma_inference.py # pruebas simples: texto, imagen, imagen+texto
  run_ablation_baseline.py    # corre las 6 condiciones del pipeline
  train_lora_medgemma.py      # entrenamiento LoRA/QLoRA
examples/
  medgemma_lora_dataset.sample.jsonl
requirements-colab.txt
```

## Prueba Local

Esto no carga MedGemma. Solo valida la logica local.

```powershell
python -m unittest discover -s tests
```

## Colab: Preparacion

En Colab usa GPU T4/L4 y ejecuta:

```bash
!git clone https://github.com/Luco1421/utils_medgemma.git
%cd utils_medgemma
!pip install -r requirements-colab.txt
```

Luego inicia sesion en Hugging Face y acepta la licencia del modelo en la web.
MedGemma requiere aceptar los terminos del repositorio de Hugging Face.

```python
from huggingface_hub import login
login()
```

Modelo recomendado para tu tarea:

```text
google/medgemma-1.5-4b-it
```

## Smoke Test: Solo Texto

```bash
python experiments/smoke_medgemma_inference.py \
  --mode text \
  --prompt "Explain glaucoma in simple clinical terms." \
  --output results/text_only.json
```

## Smoke Test: Imagen + Texto

Sube una imagen `sample_fundus.jpg` al runtime de Colab:

```bash
python experiments/smoke_medgemma_inference.py \
  --mode image-text \
  --image sample_fundus.jpg \
  --prompt "Describe the ophthalmological findings in this fundus image." \
  --output results/image_text.json
```

## Ablation del Pipeline Mejorado

Si tienes imagen y mascara real:

```bash
python experiments/run_ablation_baseline.py \
  --image sample_fundus.jpg \
  --mask sample_mask.png \
  --prediction glaucoma \
  --distribution '{"glaucoma": 0.92, "normal": 0.08}' \
  --output results/ablation_baseline.json
```

Si aun no tienes mascara, puedes hacer una prueba visual falsa:

```bash
python experiments/run_ablation_baseline.py \
  --image sample_fundus.jpg \
  --make-dummy-mask \
  --output results/ablation_dummy.json
```

La mascara falsa solo sirve para probar el flujo. No sirve como resultado
experimental.

## Dataset para LoRA

Formato JSONL:

```json
{"image": "images/001.jpg", "prompt": "Describe the ophthalmological findings in this fundus image.", "answer": "Clinical photograph..."}
```

Cada linea es un ejemplo. Para empezar, usa pocas muestras y `--max-steps 10`
para comprobar que el entrenamiento arranca.

Si tienes ACRIMA en carpetas `Glaucoma/` y `Non Glaucoma/`, puedes crear un
JSONL demostrativo:

```bash
python experiments/build_acrima_jsonl.py \
  --dataset-dir dataset \
  --output data/train_medgemma_acrima.jsonl \
  --max-per-class 20
```

Ese JSONL sirve para probar el flujo tecnico de LoRA. No sustituye un dataset
con descripciones clinicas expertas.

## Entrenamiento LoRA / QLoRA

```bash
python experiments/train_lora_medgemma.py \
  --train-jsonl data/train_medgemma.jsonl \
  --image-root data \
  --output-dir checkpoints/medgemma_lora \
  --use-qlora \
  --max-steps 100
```

El resultado sera un adapter en:

```text
checkpoints/medgemma_lora
```

## Probar el Adapter LoRA

```bash
python experiments/smoke_medgemma_inference.py \
  --mode image-text \
  --image sample_fundus.jpg \
  --adapter-path checkpoints/medgemma_lora \
  --prompt "Describe the ophthalmological findings in this fundus image." \
  --output results/image_text_lora.json
```

## Como Interpretar Tu Aporte

Tu entrega no reemplaza el trabajo de SAM, CNN, WSSS o feature density. Tu parte
produce el componente textual:

```text
imagen / mascara / prediccion / distribucion
        -> prompt e imagen preparada
        -> MedGemma baseline o MedGemma+LoRA
        -> descripcion clinica
```

Luego el equipo puede comparar:

```text
baseline pretrained vs LoRA
A/B/C1/C2/D1/D2
LoRA mask vs WSSS mask vs FSL/FD mask
```
