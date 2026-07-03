# utils_medgemma

Condicionamiento de MedGemma para descripción clínica de fondo de ojo en
glaucoma. Se usa `google/medgemma-1.5-4b-it` como caja negra (sin tocar sus
pesos) con generación determinista (`do_sample=False`, hasta 512 tokens) y se
evalúa la calidad del texto generado bajo distintas formas de condicionar la
entrada.

## Condiciones de condicionamiento

Cada imagen se describe bajo seis condiciones, definidas como prompts estáticos
en `dataset/prompts.json` y combinadas en dos ejes:

- **Overlay**: la imagen se envía tal cual, o con la máscara del disco óptico
  compuesta en rojo (60% imagen + 40% rojo dentro de la máscara).
- **Prompt según la clase**: glaucoma, normal, o sin especificar.

| Condición | Overlay | Prompt |
|---|---|---|
| `with_overlay_glaucoma` | sí | glaucoma |
| `without_overlay_glaucoma` | no | glaucoma |
| `with_overlay_normal` | sí | normal |
| `without_overlay_normal` | no | normal |
| `with_overlay` | sí | sin especificar |
| `without_overlay` | no | sin especificar |

`without_overlay` (imagen cruda, sin pista de clase) es la condición **base**
contra la que se hacen todas las comparaciones pareadas.

## Fuentes de evaluación

Las condiciones se corren sobre una misma estructura de entrada, alimentada por
dos fuentes:

- **dataset** (`--split-name test`): la etiqueta viene del ground truth y el
  overlay se pre-renderiza desde la máscara GT del disco. Los casos normales no
  tienen máscara de disco, así que en esta fuente no hay overlay para normales.
- **pipeline** (`--pipeline-dir dataset/test`): la etiqueta viene del
  clasificador real y el overlay del segmentador (SAM). Las 26 imágenes de test
  traen overlay, así que se llenan las seis condiciones.

En la fuente pipeline, la predicción del clasificador (`y_pred`) y la etiqueta
GT que la juzga (`y_true`) salen del mismo archivo,
`dataset/pipeline_classifications.json`, para trazabilidad. El texto de
referencia siempre viene de `dataset/annotations.json`.

## Diseño 2×2

El experimento cruza **modelo × fuente de evaluación**:

|          | dataset | pipeline |
|----------|---------|----------|
| **base** | `medgemma_base_dataset.json` | `medgemma_base_pipeline.json` |
| **LoRA** | `medgemma_lora_dataset_seed*.json` | `medgemma_lora_pipeline_seed*.json` |

Esto separa dos efectos: cuánto aporta el fine-tuning (base → LoRA) y cuánto se
degrada al pasar de entradas ideales a las reales del pipeline (dataset →
pipeline). El base es determinista, así que basta una corrida por fuente; el
LoRA se repite con cinco semillas (42, 123, 456, 789, 1024) para reportar
media ± desviación.

## Dataset

`dataset/` contiene el dataset plano (`annotations.json`, `split.json`,
imágenes y máscaras `*_disc` / `*_cup` por caso de glaucoma), la versión
estilo-pipeline de test en `dataset/test/` (imagen + máscara del segmentador
`*_obj_0.png` + overlays pre-generados), los prompts (`prompts.json`) y las
clasificaciones del pipeline (`pipeline_classifications.json`).

| Split | Total | Glaucoma | Normal |
|---|---:|---:|---:|
| Train | 77 | 41 | 36 |
| Validation | 26 | 14 | 12 |
| Test | 26 | 14 | 12 |

El disco óptico es la única máscara usada; las de copa existen como información
adicional pero no intervienen en la evaluación.

## Evaluación

`medgemma_utils/evaluation.py` (`Evaluator`) mide, por cada texto generado:

- **BERTScore** F1 con BiomedBERT y **similitud sBERT** con Bioclinical
  ModernBERT.
- Versiones **calibradas** de ambas: el score contra la referencia propia menos
  el score contra referencias barajadas, para descontar el parecido genérico
  entre informes (todos hablan de disco, cup-to-disc, rim…).
- `finding_mentioned`: heurística que verifica si el texto menciona términos
  coherentes con la clase esperada.

La comparación entre cada condición y la base usa el test de Wilcoxon pareado
sobre las mismas imágenes, con tamaño de efecto y corrección de Holm por
comparaciones múltiples.

`scripts/analyze_results.py` añade un análisis post-hoc en CPU (Recall@1 y
diagnóstico por clase con detección de negación) que expone el colapso de clase
que la exactitud esconde. Escribe `results/analysis_dataset.json` y
`results/analysis_pipeline.json`.

## MedGemma-LoRA

Experimento aparte de LoRA estándar (no QLoRA) sobre MedGemma. **Se entrena solo
con el split de train usando el ground truth**: imagen original + prompt
genérico → transcripción del especialista. No ve máscaras, clases ni
condicionamiento, así que aprende el estilo del informe, no a diagnosticar. La
evaluación reusa el mismo protocolo de seis condiciones en las dos fuentes. Con
el LoRA actual las métricas de similitud suben pero Recall@1 queda en azar y el
recall de la clase normal cae, señal de que memoriza plantilla.

Config: `target_modules="all-linear"`, rank 8, alpha 16, lr 1e-4, 3 épocas.

## Ejecución (Kabre)

Los jobs corren en la partición `nukwa-long`. Encadenados:

```bash
B=$(sbatch --parsable slurm/run_base.slurm)     # base × {dataset, pipeline}
L=$(sbatch --parsable slurm/run_lora.slurm)      # LoRA: 5 semillas, entrena + evalúa
sbatch --dependency=afterok:$B:$L slurm/run_analysis.slurm
```

Tests:

```bash
python -m unittest discover -s tests
```

Las curvas de entrenamiento (loss, learning rate, tiempos) están en W&B:
<https://wandb.ai/byluco1421-tecnol-gico-de-costa-rica/utils-medgemma>. Para
correr sin subir a W&B: `sbatch --export=ALL,WANDB_MODE=disabled slurm/run_lora.slurm`.

## Resultados

Los JSON de `results/` incluyen, además de las generaciones (`results`), un
resumen por condición con deltas contra la base (`summary_by_condition`), las
comparaciones pareadas (`paired_comparisons_vs_baseline`), y `dataset_audit` /
`runtime` con la procedencia y los tiempos de la corrida.
