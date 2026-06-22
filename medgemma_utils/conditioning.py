"""MedGemma ablation conditions and medical-text evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

CONDITIONS = ("A", "B", "C1", "C2", "D1", "D2")
MASK_CONDITIONS = frozenset({"B", "D1", "D2"})

PROMPT_TEMPLATES = {
    "A": "Describe the ophthalmological findings in this fundus image.",
    "B": (
        "The region highlighted in red was identified by an automatic "
        "segmentation system. Describe the ophthalmological findings, "
        "focusing on the highlighted region."
    ),
    "C1": (
        "An ophthalmological classifier identifies the primary finding in "
        "this fundus image as: {prediction}. Describe the ophthalmological "
        "findings."
    ),
    "C2": (
        "An ophthalmological classifier analyzed this fundus image and "
        "estimates: {distribution}. Describe the ophthalmological findings."
    ),
    "D1": (
        "An ophthalmological classifier identifies the primary finding as: "
        "{prediction}. The region highlighted in red indicates the area where "
        "this finding is located. Describe the findings focusing on the "
        "highlighted region."
    ),
    "D2": (
        "An ophthalmological classifier estimates: {distribution}. The region "
        "highlighted in red indicates the area where the main finding is "
        "located. Describe the findings in detail, focusing on the highlighted "
        "region and its relationship with the suggested diagnosis."
    ),
}


def load_mask(mask: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        path = Path(mask)
        array = (
            np.load(path)
            if path.suffix.lower() == ".npy"
            else np.asarray(Image.open(path).convert("L"))
        )
    else:
        array = np.asarray(mask)
    return np.squeeze(array).astype(bool)


def make_overlay(image: Any, mask: Any, alpha: float = 0.4) -> Image.Image:
    if isinstance(image, (str, Path)):
        image_array = np.asarray(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        image_array = np.asarray(image.convert("RGB"))
    else:
        image_array = np.asarray(image)

    binary_mask = load_mask(mask)
    if binary_mask.shape != image_array.shape[:2]:
        raise ValueError(
            f"mask shape {binary_mask.shape} does not match image shape "
            f"{image_array.shape[:2]}"
        )

    overlay = image_array.astype(np.float32).copy()
    overlay[binary_mask] = (
        overlay[binary_mask] * (1.0 - alpha)
        + np.array([255.0, 0.0, 0.0]) * alpha
    )
    return Image.fromarray(np.rint(overlay).clip(0, 255).astype(np.uint8))


def format_distribution(distribution: Mapping[str, float]) -> str:
    return ", ".join(
        f"{label} ({probability:.0%})"
        for label, probability in sorted(
            distribution.items(), key=lambda item: item[1], reverse=True
        )
    )


def validate_condition_inputs(
    condition: str,
    mask: Any = None,
    prediction: str | None = None,
    distribution: Mapping[str, float] | None = None,
) -> str:
    condition = condition.upper()
    required = {
        "A": (False, False, False),
        "B": (True, False, False),
        "C1": (False, True, False),
        "C2": (False, False, True),
        "D1": (True, True, False),
        "D2": (True, False, True),
    }
    if condition not in required:
        raise ValueError(f"Unknown condition: {condition}")
    needs_mask, needs_prediction, needs_distribution = required[condition]
    if needs_mask != (mask is not None):
        action = "requires" if needs_mask else "does not accept"
        raise ValueError(f"Condition {condition} {action} mask")
    if needs_prediction != (prediction is not None):
        action = "requires" if needs_prediction else "does not accept"
        raise ValueError(f"Condition {condition} {action} prediction")
    if needs_distribution != (distribution is not None):
        action = "requires" if needs_distribution else "does not accept"
        raise ValueError(f"Condition {condition} {action} distribution")
    if prediction is not None and not str(prediction).strip():
        raise ValueError("prediction cannot be empty")
    if distribution is not None:
        probabilities = [float(value) for value in distribution.values()]
        if not distribution or any(value < 0 for value in probabilities):
            raise ValueError("distribution must contain non-negative values")
        if not np.isclose(sum(probabilities), 1.0, atol=1e-4):
            raise ValueError("distribution probabilities must sum to 1")
    return condition


def build_prompt(
    condition: str,
    *,
    prediction: str | None = None,
    distribution: Mapping[str, float] | None = None,
) -> str:
    return PROMPT_TEMPLATES[condition].format(
        prediction=prediction,
        distribution=(
            format_distribution(distribution)
            if distribution is not None
            else None
        ),
    )
