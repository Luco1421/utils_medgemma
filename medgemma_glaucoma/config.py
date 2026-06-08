# -*- coding: utf-8 -*-
"""Configuracion central del proyecto (modelo, dataset, prompts)."""
import os

MODEL_ID = "google/medgemma-4b-it"

# --- Modo clasificador: dataset local con subcarpetas por clase ---
CLASS_LABELS = {"Glaucoma": "glaucoma", "Non Glaucoma": "no glaucoma"}

# --- Modo descriptor: de donde salen las descripciones ---
# Cuando tengas tu propio dataset, pon DESC_SOURCE = "csv" y define DESC_CSV,
# o cambia el repo/columnas de Hugging Face. El resto del codigo no cambia.
DESC_SOURCE  = "hf"                       # "hf" | "csv"
HF_REPO      = "lxirich/MM-Retinal-Reason"
HF_SPLIT     = "complex"
HF_IMAGE_COL = "image"
HF_TEXT_COL  = "caption"
DESC_CSV     = None                       # csv con columnas: filename,description

CLASSIFIER_Q = "Does this fundus image suggest glaucoma? Answer 'Glaucoma' or 'No glaucoma'."
DESCRIPTOR_Q = "Describe this ophthalmic fundus image."

# Rutas donde buscar el dataset local (la primera que exista gana).
_DATASET_CANDIDATES = (
    "/content/dataset",            # Colab (subido a /content)
    "dataset",                     # ejecutando desde la raiz del repo
    "../dataset",
    "utils_medgemma/dataset",      # repo clonado en Colab
)


def default_dataset_dir():
    """Devuelve la primera ruta de dataset que exista (o 'dataset' por defecto)."""
    for c in _DATASET_CANDIDATES:
        if os.path.isdir(c):
            return c
    return "dataset"
