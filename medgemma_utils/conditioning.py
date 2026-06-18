"""MedGemma ablation conditions and medical-text evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from .inputs import ConditioningInput

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


def build_condition_request(
    condition: str,
    sample: ConditioningInput,
) -> dict[str, Any]:
    condition = condition.upper()
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")

    image = (
        make_overlay(sample.image, sample.mask)
        if condition in MASK_CONDITIONS
        else sample.image
    )
    prompt = PROMPT_TEMPLATES[condition].format(
        prediction=sample.prediction,
        distribution=format_distribution(sample.distribution),
    )
    return {
        "condition": condition,
        "image": image,
        "prompt": prompt,
        "image_was_overlaid": condition in MASK_CONDITIONS,
    }


def run_condition_experiment(
    sample: ConditioningInput,
    *,
    generate: Callable[[str, Any, int], str],
    model_variant: str,
    max_new_tokens: int = 384,
    conditions: Sequence[str] = CONDITIONS,
) -> list[dict[str, Any]]:
    results = []
    for condition in conditions:
        request = build_condition_request(condition, sample)
        generated = generate(
            request["prompt"],
            request["image"],
            max_new_tokens,
        ).strip()
        results.append(
            {
                "image_id": sample.image_id,
                "condition": condition,
                "model_variant": model_variant,
                "input_source": sample.input_source,
                "mask_source": sample.mask_source,
                "prediction": sample.prediction,
                "distribution": sample.distribution,
                "prompt_used": request["prompt"],
                "image_was_overlaid": request["image_was_overlaid"],
                "reference": sample.reference,
                "generated": generated,
            }
        )
    return results
