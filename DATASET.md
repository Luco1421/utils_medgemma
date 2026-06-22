# Contrato del dataset

## Estructura verificada

`dataset/annotations.json` contiene 129 anotaciones únicas y
`dataset/split.json` contiene esas mismas anotaciones embebidas en tres
particiones sin imágenes faltantes ni repetidas:

| Split | Total | Glaucoma | Normal |
|---|---:|---:|---:|
| Train | 77 | 41 | 36 |
| Validation | 26 | 14 | 12 |
| Test | 26 | 14 | 12 |

Cada anotación tiene imagen y transcripción. Los casos de glaucoma usan
`label=Pathological` y `locs_data.conditions=["glaucoma"]`; los normales usan
`label=Normal` y `locs_data={}`.

## Máscaras

Cada una de las 69 imágenes con glaucoma incluye:

- `<image_id>_disc.png`
- `<image_id>_disc.npy`
- `<image_id>_cup.png`
- `<image_id>_cup.npy`

Las 60 imágenes normales no incluyen máscaras. Esto es intencional: en el
pipeline, una salida normal del clasificador no se envía a SAM.

La especificación raíz de `ref` indica usar únicamente disco óptico. Por tanto:

- La GT contractual es `*_disc.png`.
- Copa no se combina con disco.
- Copa no se evalúa como segunda clase.
- No se crea una máscara vacía para normales.

Los 69 PNG de disco coinciden exactamente con sus NPY y tienen el mismo tamaño
que la imagen correspondiente. `1258_right_cup.npy` está vacío, pero ese archivo
queda fuera del contrato porque copa no se usa. El dataset recibido no se
modifica para ocultar o reparar esta anomalía.

## Uso por etapa

Entrenamiento de MedGemma-LoRA experimental:

- Usa las 77 imágenes y transcripciones de `train`.
- No usa máscaras ni señales del clasificador.

Evaluación oracle de M7:

- A/C1/C2 usan las 26 imágenes de test.
- B/D1/D2 usan las 14 imágenes de glaucoma y su GT de disco como entrada visual.
- No se reportan métricas de segmentación para esa GT, porque no es una
  predicción.

Evaluación futura del pipeline:

- `prediction=normal` implica `mask=None` y SAM omitido.
- `prediction=glaucoma` permite una máscara predicha de disco.
- Segmentación se puntúa solo si existen predicción y GT.
- Un falso negativo del clasificador puede tener GT pero no máscara predicha.
- Un falso positivo puede tener máscara predicha pero no GT; en ese caso no
  existe score de segmentación, aunque el texto sí puede evaluarse.
