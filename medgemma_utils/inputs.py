"""Normalized input contract consumed by the MedGemma experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConditioningInput:
    image_id: str
    image: Any
    mask: Any | None
    prediction: str
    distribution: dict[str, float]
    reference: str
    mask_source: str
    input_source: str
    expected_finding: str | None = None
    ground_truth_mask: Any | None = None
    mask_target: str = "optic_disc"
    segmentation_status: str | None = None


def load_json_inputs(path: str | Path) -> list[ConditioningInput]:
    """Load future real pipeline outputs using one explicit JSON schema."""

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Pipeline input JSON must contain a list")
    inputs = [ConditioningInput(**record) for record in records]
    for sample in inputs:
        if sample.mask_target != "optic_disc":
            raise ValueError(
                f"Unsupported mask target for {sample.image_id}: "
                f"{sample.mask_target!r}; ref requires 'optic_disc'"
            )
        if sample.prediction not in {"glaucoma", "normal"}:
            raise ValueError(
                f"Unsupported prediction for {sample.image_id}: "
                f"{sample.prediction!r}"
            )
        probabilities = {
            str(label): float(value)
            for label, value in sample.distribution.items()
        }
        if set(probabilities) != {"glaucoma", "normal"}:
            raise ValueError(
                f"Distribution for {sample.image_id} must contain exactly "
                "'glaucoma' and 'normal'"
            )
        if any(value < 0.0 for value in probabilities.values()):
            raise ValueError(
                f"Distribution for {sample.image_id} contains negative values"
            )
        if abs(sum(probabilities.values()) - 1.0) > 1e-4:
            raise ValueError(
                f"Distribution for {sample.image_id} must sum to 1"
            )
        if sample.prediction == "normal" and sample.mask is not None:
            raise ValueError(
                f"Normal prediction must bypass segmentation: {sample.image_id}"
            )
    return inputs
