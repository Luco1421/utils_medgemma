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

`modules/evaluator.py` expone:

```python
evaluator.evaluate_segmentation(pred_mask, gt_mask)
evaluator.evaluate_text(generated_text, reference_text, expected_finding)
evaluator.statistical_test(scores_a, scores_b)
```

M8 calcula IoU, Dice, SSIM, BERTScore con BiomedBERT, similitud sBERT con
Bioclinical ModernBERT, precisión de hallazgo, Wilcoxon pareado, effect size y
corrección Holm. También guarda baselines de referencias permutadas y scores
calibrados para interpretar el piso elevado de las métricas semánticas.

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

## Configuración

`config.yaml` centraliza semilla, dataset, modelo, evaluación y el experimento
LoRA. Los scripts validan que:

- El objetivo sea `optic_disc`.
- Los normales omitan segmentación.
- El resumen del dataset coincida con 77/26/26.

Preflight local:

```bash
/data/jperez/app_data/miniconda/envs/medgemma/bin/python \
  -m scripts.preflight
```

## Ejecución en Kabre

Full base:

```bash
sbatch run_medgemma_base_full.slurm
```

Full LoRA experimental (entrenado con prompt genérico tipo-A):

```bash
sbatch run_experimental_medgemma_lora_full.slurm
```

Full LoRA experimental por-condición (entrenado con cada condición, con overlay
y pistas de clase, igual que en la evaluación):

```bash
sbatch run_experimental_medgemma_lora_conditioned_full.slurm
```

El modo de entrenamiento se elige con `--prompt-mode {generic,conditioned}` en
`scripts.train_experimental_medgemma_lora`. La variante `generic` aprende solo el
estilo del experto; la `conditioned` además aprende a usar la máscara y la clase.

Todos los jobs ejecutan preflight de dataset, CUDA y token de Hugging Face antes
de cargar los modelos. Los jobs full no aplican límites de muestras.

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
