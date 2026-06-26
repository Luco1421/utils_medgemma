# Resultados

Estructura:

```
results/
  baseline/                 # MedGemma base (sin entrenar), punto de comparacion
    conditioned_base.json
    logs/                   # .out/.err de Slurm
  lora/                     # LoRA generico (extra), config justa, multi-seed en test
    test_lora_seed42..44.json
    logs/
  analysis_extras.json      # extras post-hoc (recall@1 + diagnostico por clase)
                            # generado por: python -m scripts.analyze_results
```

Cada JSON usa `schema_version: m7-m8-v2`. Una corrida completa contiene 120
generaciones: 26 para A/C1/C2 y 14 para B/D1/D2.

Además incluye:

- `dataset_audit`: split usado, resumen 77/26/26, límite/filtro opcional y
  bandera `full_selected_split`.
- `runtime`: timestamps, duración total y tiempos por etapa.
- `generated_result_count`: número final de filas generadas/evaluadas.

Cada registro incluye:

- `generated_text` y `reference_text`.
- Condición, prompt y variante del modelo.
- Clasificación y distribución.
- `mask_target=optic_disc`.
- Fuente y estado de segmentación.
- Métricas textuales de M8.
- Métricas de segmentación solo cuando hay máscara predicha y GT.

`coverage_summary_by_condition` resume la cobertura completa.
`summary_by_condition` y `paired_comparisons_vs_A` usan las mismas 14 imágenes
con máscara para comparar las seis condiciones.

Los JSON de resultados y los logs de Slurm (`logs/`) se versionan, para poder
comparar corridas entre versiones. Solo se ignora el caché local de W&B
(`results/wandb/`), que es binario pesado y ya está en W&B online.
