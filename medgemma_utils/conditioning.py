"""MedGemma conditioning: prompt specs loaded from prompts.json and overlays.

The six experimental conditions are static prompts defined in
``dataset/prompts.json``, combined along two axes:

* overlay vs. no overlay (``use_overlay``)
* prompt family by classifier label: glaucoma, normal, or unspecified
  (derived from the condition name suffix)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "dataset" / "prompts.json"

# The unspecified, no-overlay prompt is the reference condition for every
# paired comparison (no class hint, raw image): the analogue of the old "A".
BASELINE_CONDITION = "without_overlay"


@dataclass(frozen=True)
class ConditionSpec:
    """One experimental condition declared in prompts.json."""

    name: str
    use_overlay: bool
    prompt: str
    label_scope: str  # "glaucoma" | "normal" | "any"


def _label_scope(name: str) -> str:
    if name.endswith("_glaucoma"):
        return "glaucoma"
    if name.endswith("_normal"):
        return "normal"
    return "any"


def load_condition_specs(
    path: str | Path = PROMPTS_PATH,
) -> dict[str, ConditionSpec]:
    """Load the condition specifications from prompts.json (order preserved)."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("prompts.json must contain an object keyed by condition")
    specs: dict[str, ConditionSpec] = {}
    for name, entry in data.items():
        if "use_overlay" not in entry or "prompt" not in entry:
            raise ValueError(
                f"Condition {name!r} must define 'use_overlay' and 'prompt'"
            )
        prompt = str(entry["prompt"]).strip()
        if not prompt:
            raise ValueError(f"Condition {name!r} has an empty prompt")
        specs[name] = ConditionSpec(
            name=name,
            use_overlay=bool(entry["use_overlay"]),
            prompt=prompt,
            label_scope=_label_scope(name),
        )
    if BASELINE_CONDITION not in specs:
        raise ValueError(
            f"prompts.json must define the baseline condition "
            f"{BASELINE_CONDITION!r}"
        )
    return specs


CONDITION_SPECS = load_condition_specs()
CONDITIONS = tuple(CONDITION_SPECS)
MASK_CONDITIONS = frozenset(
    name for name, spec in CONDITION_SPECS.items() if spec.use_overlay
)


def condition_applies(
    spec: ConditionSpec,
    prediction: str,
    *,
    has_overlay: bool,
) -> bool:
    """Whether a condition is run for a sample with the given label/overlay.

    A condition runs only when its prompt family matches the classified label
    (or is unspecified), and overlay conditions require an overlay image.
    """

    if spec.label_scope not in ("any", prediction):
        return False
    if spec.use_overlay and not has_overlay:
        return False
    return True


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
