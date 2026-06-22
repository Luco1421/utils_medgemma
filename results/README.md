# Resultados

Estructura:

```
results/
  baseline/                 # MedGemma base (sin entrenar), punto de comparacion
    conditioned_base.json
    logs/                   # .out/.err de Slurm
  sweep/                    # barrido de hiperparametros en el split de validacion
    val_idx0..5.json        # una corrida por config (epocas x modo de prompt)
    summary.json            # generado por scripts.aggregate_sweep (config ganadora)
    logs/
  final/                    # confirmacion multi-seed en el split de test
    test_<selector>_seed<N>.json
    logs/
```

Cada JSON usa `schema_version: m7-m8-v2`. Una corrida completa contiene 120
generaciones: 26 para A/C1/C2 y 14 para B/D1/D2.

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

Los JSON de resultados se pueden versionar. Los logs de Slurm (`logs/`) son
artefactos temporales ignorados por Git.
