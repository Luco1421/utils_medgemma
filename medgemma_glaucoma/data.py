# -*- coding: utf-8 -*-
"""Carga de datos - 2 modos: 'classifier' (dataset local) y 'descriptor' (HF o CSV)."""
import os, glob, random, io, base64
from PIL import Image

from . import config


def _glob_images(folder):
    out = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        out += glob.glob(os.path.join(folder, ext))
    return out


def load_records(mode="classifier", *, limit=None, seed=42,
                 dataset_dir=None, class_labels=None,
                 desc_source=None, desc_csv=None,
                 hf_repo=None, hf_split=None,
                 hf_image_col=None, hf_text_col=None):
    """Lista de dicts {image, question, answer[, label]}.
       'image' puede ser ruta (str) o PIL.Image.

       mode="classifier" -> dataset LOCAL (etiqueta = nombre de carpeta).
       mode="descriptor" -> descripciones desde Hugging Face (o CSV propio).
    """
    dataset_dir  = dataset_dir  or config.default_dataset_dir()
    class_labels = class_labels or config.CLASS_LABELS
    desc_source  = desc_source  or config.DESC_SOURCE
    desc_csv     = desc_csv if desc_csv is not None else config.DESC_CSV
    hf_repo      = hf_repo      or config.HF_REPO
    hf_split     = hf_split     or config.HF_SPLIT
    hf_image_col = hf_image_col or config.HF_IMAGE_COL
    hf_text_col  = hf_text_col  or config.HF_TEXT_COL

    rng = random.Random(seed)
    records = []

    if mode == "classifier":
        for folder, label in class_labels.items():
            ans = "Glaucoma." if label == "glaucoma" else "No glaucoma."
            for p in _glob_images(os.path.join(dataset_dir, folder)):
                records.append({"image": p, "question": config.CLASSIFIER_Q,
                                "answer": ans, "label": label})
        if not records:
            raise FileNotFoundError(
                f"0 imagenes encontradas en '{os.path.abspath(dataset_dir)}' "
                f"(esperaba subcarpetas {list(class_labels)}). "
                f"Sube el dataset o pasa dataset_dir='...' a load_records().")

    elif mode == "descriptor":
        if desc_source == "csv":
            # Tu propio dataset: imagenes locales + captions en un CSV.
            import csv
            assert desc_csv, "Define desc_csv='ruta.csv' con columnas: filename,description"
            desc = {}
            with open(desc_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    desc[row["filename"]] = row["description"]
            for folder in class_labels:
                for p in _glob_images(os.path.join(dataset_dir, folder)):
                    cap = desc.get(os.path.basename(p))
                    if cap:
                        records.append({"image": p, "question": config.DESCRIPTOR_Q,
                                        "answer": cap})
        else:
            # Dataset de descripciones desde Hugging Face (cambiable).
            from datasets import load_dataset
            raw = load_dataset(hf_repo, split=hf_split).shuffle(seed=seed)
            for ex in raw:
                cap = (ex.get(hf_text_col) or "").strip()
                if not cap:
                    continue
                img = ex[hf_image_col]
                if isinstance(img, list):
                    img = img[0] if img else None
                if isinstance(img, str):              # base64 -> PIL
                    try:
                        img = Image.open(io.BytesIO(base64.b64decode(img)))
                    except Exception:
                        continue
                if img is None:
                    continue
                records.append({"image": img.convert("RGB"),
                                "question": config.DESCRIPTOR_Q, "answer": cap})
                if limit and len(records) >= limit:
                    break
        if not records:
            raise ValueError(
                "0 descripciones cargadas. Revisa desc_source/hf_repo/hf_split o desc_csv.")
    else:
        raise ValueError(f"modo desconocido: {mode!r} (usa 'classifier' o 'descriptor')")

    rng.shuffle(records)
    return records[:limit] if limit else records


def train_test_split(records, n_test=20, n_train=None, seed=42):
    """Split simple; si hay 'label', balancea el test por clase."""
    rng = random.Random(seed)
    recs = records[:]; rng.shuffle(recs)
    if recs and "label" in recs[0]:
        by = {}
        for r in recs:
            by.setdefault(r["label"], []).append(r)
        per = max(1, n_test // max(1, len(by)))
        test = [r for grp in by.values() for r in grp[:per]]
        test_ids = {id(r) for r in test}
        train = [r for r in recs if id(r) not in test_ids]
    else:
        test, train = recs[:n_test], recs[n_test:]
    return (train[:n_train] if n_train else train), test


def build_hf_dataset(records):
    """Convierte la lista de records en un datasets.Dataset listo para entrenar."""
    from datasets import Dataset, Image as HFImage
    ds = Dataset.from_dict({
        "image":    [r["image"]    for r in records],
        "question": [r["question"] for r in records],
        "answer":   [r["answer"]   for r in records],
    })
    return ds.cast_column("image", HFImage())  # rutas o PIL -> imagen HF
