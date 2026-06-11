# SPEC: MedGemma Base vs LoRA/QLoRA

## Objetivo

Comparar dos variantes de MedGemma para generar descripciones clinicas de imagenes
de fondo de ojo:

1. **MedGemma base**: modelo original sin modificar.
2. **MedGemma LoRA/QLoRA**: modelo base con adapter LoRA entrenado sobre
   transcripciones expertas.

Este experimento es una fase adicional al pipeline principal de condicionamiento.
No reemplaza el uso de MedGemma como caja negra en las condiciones A-D; agrega una
comparacion controlada para medir si el fine-tuning mejora la descripcion clinica.

## Pregunta Experimental

Usando el mismo split y la misma metrica, el adapter LoRA mejora el BERTScore F1
medio frente a MedGemma base?

## Datos

El experimento usa los splits existentes:

```text
dataset/split_repetition_1.json
```

Cada item del split apunta a:

- `image`: imagen de fondo de ojo.
- `annotation`: JSON con `transcription`, `label` y `locs_data`.

La transcripcion experta se usa como referencia y como target de SFT.

## Modulos

### M1: DataModule

Responsable de cargar splits y anotaciones. Devuelve filas normalizadas con:

- `split`
- `image`
- `annotation`
- `label`
- `conditions`
- `target_label`
- `prompt`
- `answer`
- `reference`
- `locs_data`

### M2: MedGemmaModel

Responsable de cargar MedGemma base o MedGemma + adapter LoRA para inferencia.
Tambien centraliza la generacion deterministica con `do_sample=False`.

### M3: LoraTrainer

Responsable del entrenamiento LoRA/QLoRA:

- carga 4-bit con bitsandbytes
- configura LoRA
- prepara conversaciones multimodales
- entrena con `SFTTrainer`
- guarda adapter y processor

### M4: Evaluator

Responsable de evaluar textos generados contra referencias expertas con BERTScore
y guardar resultados JSON reproducibles.

## Reproducibilidad

- Toda configuracion vive en `config.yaml`.
- Semilla global: `seed`.
- No se generan splits nuevos.
- No se usan `print`; los scripts usan `logging`.
- Las salidas incluyen timestamp, config y seed.

## Salidas

```text
results/base_results.json
results/lora_results.json
results/comparison.json
checkpoints/official_medgemma_qlora_description/
```

## Criterio Principal

```text
delta_f1 = qlora.bertscore_f1_mean - base.bertscore_f1_mean
```

Si `delta_f1 > 0`, LoRA mejora la similitud semantica promedio segun BERTScore.
