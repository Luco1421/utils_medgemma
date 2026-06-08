# -*- coding: utf-8 -*-
"""Evaluacion en lote para ambos modos."""
from PIL import Image

from .model import ask_image


def evaluate(model, processor, records, mode="classifier",
             max_new_tokens=None, verbose=True):
    """Corre el modelo sobre 'records'. En 'classifier' calcula accuracy.

    Devuelve la accuracy (float) en modo classifier, o None en descriptor.
    """
    if not records:
        print("No hay ejemplos para evaluar (records vacio).")
        return None

    if max_new_tokens is None:
        max_new_tokens = 64 if mode == "classifier" else 220

    correct = 0
    for i, rec in enumerate(records):
        img = rec["image"]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        pred = ask_image(model, processor, img, rec["question"],
                         max_new_tokens=max_new_tokens)
        if verbose:
            print(f"\n[{i+1}] PRED: {pred}\n     REF: {rec['answer']}")
        if mode == "classifier":
            pred_glauc = ("glaucoma" in pred.lower()) and ("no glaucoma" not in pred.lower())
            correct += int(pred_glauc == (rec["label"] == "glaucoma"))

    if mode == "classifier":
        acc = correct / len(records)
        print(f"\nAccuracy ({mode}): {correct}/{len(records)} = {acc:.1%}")
        return acc
    return None
