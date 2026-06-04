# Proyecto: Detección de Glaucoma con Imágenes de Fondo de Ojo

> Documento de contexto / planificación. Aún en fase de estudio (oftalmología).
> No iniciar implementación todavía.

---

## 1. Objetivo

Detectar **glaucoma** a partir de **imágenes de fondo de ojo (fundus)**, comparando
distintos pipelines de procesamiento para encontrar el más sólido como detector.

La idea es experimentar con varias combinaciones (segmentación, clasificación,
descripción por LLM, comparación) y fine-tuning con **LoRA** sobre los modelos
para garantizar los mejores resultados.

---

## 2. Aclaración importante sobre el órgano objetivo

- Inicialmente se habló de "iris", pero para **glaucoma** lo relevante es el
  **fondo de ojo / disco óptico** (segmento posterior), NO el iris (segmento anterior).
- La *iridología* (diagnóstico de enfermedades por el iris) **no tiene respaldo científico**;
  evitar ese marco.
- Señales clínicas clave del glaucoma en fundus:
  - **Relación copa/disco (CDR)** — la métrica central.
  - Adelgazamiento del **anillo neurorretiniano** (regla **ISNT**).
  - Defectos de la **capa de fibras nerviosas (RNFL)**.
  - **Hemorragias en astilla** (disco óptico).
  - **Atrofia peripapilar**.

→ El pipeline gira en torno a **segmentar copa y disco óptico** y derivar el **CDR**
  como feature clínica interpretable (no solo máscara para el LLM).

---

## 3. Modelos candidatos

| Modelo | Rol propuesto | Notas |
|--------|---------------|-------|
| **MedSAM / SAM (MedSAM3)** | Segmentación de copa y disco óptico | El CDR derivado puede usarse como feature interpretable, no solo máscara. Fine-tune con LoRA. |
| **MedSigLIP** | Clasificación / embeddings | Útil como clasificador o para zero-shot. |
| **MedGemma** | Generación de descripciones / razonamiento multimodal | Recibe máscara o imagen y produce descripción/diagnóstico. |
| **RETFound / RETZero** | Backbone o baseline de comparación | RETFound = foundation model entrenado en retina; baseline fuerte. RETZero para comparación. |

**Fine-tuning:** LoRA en todos (eficiente en parámetros).

---

## 4. Pipelines / experimentos a comparar

Formalizar como matriz de experimentos con variables controladas (comparación justa):

1. **Segmentador → LLM** (máscara → descripción).
2. **Clasificador → LLM**.
3. **Segmentador → Clasificador → LLM** (máscara → clasif → descripción).
4. (Posibles variantes adicionales: imagen cruda → LLM, CDR como feature → clasif, etc.)

---

## 5. Datasets públicos de glaucoma

| Dataset | Contenido | Útil para |
|---------|-----------|-----------|
| **REFUGE / REFUGE2** | Segmentación copa-disco + etiqueta glaucoma | ⭐ Ideal para este caso (seg + clasif). |
| **ORIGA** | Fundus con etiquetas de glaucoma | Clasificación / CDR. |
| **RIM-ONE** | Disco óptico, glaucoma | Segmentación / clasificación. |
| **DRISHTI-GS** | Segmentación copa-disco | Validación del segmentador. |
| **G1020** | Fundus glaucoma | Clasificación. |
| **ACRIMA** | Fundus glaucoma | Clasificación. |
| **PAPILA** | Fundus + datos clínicos | Multimodal. |
| **LAG** | Large-scale glaucoma | Clasificación. |

→ Conseguir datos con **máscaras de copa/disco** es clave para validar el segmentador
  (REFUGE2 y DRISHTI-GS destacan aquí).

---

## 6. Decisiones a tomar antes de tocar código

1. **Datasets**: elegir base principal (REFUGE2 recomendado) + datasets de validación externa.
2. **Métricas**:
   - ML: AUC, sensibilidad, especificidad.
   - Clínico: **sensibilidad a alta especificidad** (clave en cribado).
3. **Diseño experimental**: tabla de experimentos con variables controladas.
4. Definir splits, validación cruzada y validación externa (generalización entre datasets).

---

## 7. Primeros pasos sugeridos (cuando se inicie)

- Armar la **matriz de experimentos**.
- Montar esqueleto del repo.
- Explorar/preparar un dataset inicial (p. ej. REFUGE).

---

_Última actualización: 2026-06-02_
