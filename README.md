# Utils MedGemma

Implementación modular de M7 (`MedGemmaConditioner`) y M8 (`Evaluator`) para
los experimentos de condicionamiento descritos en
`ref/Medgemma_Segmentation_CIARP_2026/`.

`ref/` se considera fuente contractual y no se modifica. La directiva adicional
de `ref/SPEC.md` establece que la segmentación de este proyecto usa únicamente
el disco óptico.

## Alcance

La ruta oficial usa `google/medgemma-1.5-4b-it` como caja negra, sin modificar
sus pesos. Implementa las seis condiciones:

| Condición | Imagen | Información del prompt |
|---|---|---|
| A | Original | Genérico |
| B | Overlay del disco óptico | Región resaltada |
| C1 | Original | Clase predicha |
| C2 | Original | Distribución del clasificador |
| D1 | Overlay del disco óptico | Clase y región |
| D2 | Overlay del disco óptico | Distribución y región |

El overlay conserva 60% de la imagen y añade 40% de rojo dentro de la máscara.
La generación es determinista (`do_sample=False`) y usa hasta 512 tokens.

## Dataset actual

El dataset es plano: `annotations.json`, `split.json`, imágenes y máscaras
están directamente en `dataset/`.

| Split | Total | Glaucoma | Normal | Máscara GT de disco |
|---|---:|---:|---:|---:|
| Train | 77 | 41 | 36 | 41 |
| Validation | 26 | 14 | 12 | 14 |
| Test | 26 | 14 | 12 | 14 |

Reglas aplicadas:

- `dataset/split.json` se usa exactamente como fue entregado.
- Los 69 casos de glaucoma tienen máscara GT de disco óptico.
- Los 60 casos normales no tienen máscara y no pasan por SAM.
- Los archivos de copa existen como información adicional del dataset, pero no
  forman parte de M7/M8 ni del objetivo de segmentación definido por `ref`.
- El adaptador usa `*_disc.png`, validado contra imagen y anotación.

La auditoría detallada está en [DATASET.md](DATASET.md).

## Flujo de segmentación

En el pipeline real, el clasificador se ejecuta antes de SAM:

1. Predicción `normal`: se omite SAM; se evalúan A, C1 y C2.
2. Predicción `glaucoma`: se ejecuta el segmentador; si produce máscara, también
   se evalúan B, D1 y D2.
3. IoU, Dice y SSIM solo se calculan cuando existe una máscara predicha y una
   GT de disco óptico.

No se fabrican máscaras vacías para normales y no se compara una GT contra sí
misma como si fuera una predicción.

Mientras llegan CNN y SAM, el experimento oracle usa:

- Clase derivada de la anotación.
- Distribución 80/20 configurable.
- Máscara GT de disco únicamente para condicionar B, D1 y D2.

Esto produce 120 textos por variante:

- A, C1 y C2: 26 imágenes de test.
- B, D1 y D2: 14 imágenes de glaucoma con máscara.
- Comparaciones pareadas A-D2: las mismas 14 imágenes.

## M8

`medgemma_utils/evaluation.py` expone la clase `Evaluator`:

```python
evaluator.evaluate_segmentation(pred_mask, gt_mask)
evaluator.evaluate_text(generated_text, reference_text, expected_finding)
evaluator.statistical_test(scores_a, scores_b)
```

M8 calcula IoU, Dice, SSIM, BERTScore con BiomedBERT, similitud sBERT con
Bioclinical ModernBERT, precisión de hallazgo, Wilcoxon pareado, effect size y
corrección Holm. También guarda baselines de referencias permutadas y scores
calibrados para interpretar el piso elevado de las métricas semánticas.

### Extras de análisis (no contractuales)

M8 es la evaluación principal, pero sus métricas de similitud (BERTScore, sBERT)
están **infladas por el boilerplate** de los informes: todos hablan de disco,
cup-to-disc, rim… así que el texto generado se parece a *cualquier* referencia.
Las calibradas ya lo exponen (≈0). Para una conclusión defendible se agregan, en
`scripts/analyze_results.py` (post-hoc, sin GPU, escribe
`results/analysis_extras.json`):

- **Recall@1**: ¿el texto generado identifica *su* imagen entre las referencias?
  (azar = 1/N). Mide especificidad de imagen.
- **Diagnóstico por clase (negación-aware)**: extrae la clase del texto y la
  compara con la etiqueta real; el **recall por clase** delata el colapso de
  clase que la accuracy esconde.

Hallazgo con el LoRA actual: las métricas de similitud suben mucho, pero Recall@1
queda en azar y el LoRA colapsa a "siempre glaucoma" (recall de la clase normal =
0.00). Es decir, aprende el **estilo/plantilla** del informe, no a diagnosticar.
La interpretación clínica fina requiere revisión cualitativa de oftalmólogos.

El campo Likert queda reservado para revisión clínica ciega mediante
`scripts/export_likert_review.py`.

## MedGemma-LoRA experimental

Además de M7 oficial se conserva un experimento separado de LoRA estándar sobre
MedGemma. No es QLoRA y no debe confundirse con SAM-LoRA de M3/M9.

Se entrena únicamente con las 77 muestras de `train`:

```text
imagen original + prompt genérico → transcripción del especialista
```

No usa máscaras, clases ni distribuciones durante el entrenamiento. Después se
evalúa con el mismo protocolo A-D2 para estudiar generalización y posible
memorización.

Es un extra **no contractual**: `ref` (M7) establece que a MedGemma no se le
aplica LoRA, y las semillas/ranks de `ref` pertenecen al CNN (M2) y a SAM-LoRA
(M3b), no a M7/M8. Por eso no se hace barrido de hiperparámetros: se usa **una
config justa** (rank 8, alpha 16, lr 1e-4, 3 épocas) repetida con **3 semillas**
(42, 43, 44) para reportar media ± std. El rigor de la comparación lo aporta el
Wilcoxon pareado de M8 (base vs LoRA sobre los mismos ítems), no la cantidad de
configuraciones. LoRA se aplica con `target_modules="all-linear"` (todo el
modelo, incluida la parte de visión).

## Configuración

`config.yaml` centraliza semilla, dataset, modelo, evaluación y el experimento
LoRA. Los scripts validan que:

- El objetivo sea `optic_disc`.
- Los normales omitan segmentación.
- El resumen del dataset coincida con 77/26/26.

## Ejecución en Kabre

Full base:

```bash
sbatch run_medgemma_base_full.slurm
```

LoRA experimental (config justa, 3 semillas en un array `0-2` → seeds 42/43/44):

```bash
sbatch run_experimental_medgemma_lora_full.slurm
```

El LoRA se entrena siempre con el prompt genérico tipo-A (imagen original →
transcripción del experto), por lo que aprende solo el estilo del experto y nunca
ve las pistas de condicionamiento usadas en la evaluación. Así la comparación
base-vs-LoRA aísla el efecto del fine-tuning sin sesgo de formato. Cada semilla
entrena y evalúa en test con el mismo protocolo A–D2 que el base
(`results/lora/test_lora_seed<42..44>.json`).

Los jobs oficiales no aplican límites de muestras: entrenan con todo `train` y
evalúan en el split indicado por el protocolo. Cada SLURM usa `nukwa-long` con
`--time=24:00:00`, pero los tiempos reportables se registran desde Python en
`training_metadata.json` y en `runtime` de los JSON de evaluación.

Los CLIs conservan flags manuales para ejecuciones controladas fuera del
protocolo oficial: `--train-limit` en entrenamiento y `--limit`/`--masked-only`
en evaluación. Cuando se usan, la metadata marca que la corrida no fue full.

Los jobs LoRA guardan:

- Checkpoints intermedios del adapter (`save_strategy=epoch` cuando se entrena
  por épocas; `save_strategy=steps` para corridas por `max_steps`).
- Adapter final (`adapter_model.safetensors` y `adapter_config.json`).
- Processor/tokenizer junto al adapter.
- `training_metadata.json` con configuración, resumen del dataset, métricas de
  entrenamiento, artefactos guardados y duración total.

Las curvas de entrenamiento (loss, learning rate, tiempos) se ven gráficamente
en el proyecto de W&B:

**https://wandb.ai/byluco1421-tecnol-gico-de-costa-rica/utils-medgemma**

El caché local de W&B (`results/wandb/`) no se versiona: son binarios pesados que
GitHub no renderiza y que ya están en el enlace de arriba. Los JSON de evaluación
y los logs de Slurm sí se versionan en `results/`.

W&B está integrado en `scripts.train_experimental_medgemma_lora` con los
argumentos `--wandb-project`, `--wandb-entity`, `--wandb-run-name`,
`--wandb-mode {online,offline,disabled}` y `--wandb-tags`. Por defecto corre en
modo online, igual que el script de referencia de
`imagine-laboratory/vlm-od-agriculture`:

```bash
sbatch run_experimental_medgemma_lora_full.slurm
```

Si se quiere correr sin subir a W&B:

```bash
sbatch --export=ALL,WANDB_MODE=disabled run_experimental_medgemma_lora_full.slurm
```

Los JSON de evaluación incluyen `dataset_audit` y `runtime`. Ahí queda explícito
si una corrida usó el split completo (`full_selected_split: true`), entradas
externas por `--inputs-json`, un límite manual con `--limit`, o el filtro
`--masked-only`.

## Integración con M9

Las salidas futuras de los otros módulos se reciben con
`--inputs-json`. El contrato está ejemplificado en
`examples/pipeline_inputs.example.json` e incluye imagen, predicción,
distribución, máscara predicha opcional, GT opcional, objetivo
`optic_disc` y estado de segmentación.

M7 no decide si se ejecuta SAM; consume la salida ya enrutada por M9. Si
`mask=None`, omite B, D1 y D2.

## Resultados

Los resultados nuevos usan `schema_version: m7-m8-v2` y se guardan en
`results/`. No se conservan resultados de contratos o datasets anteriores.
El formato se documenta en `results/README.md`.
