"""Normalized input contract consumed by the MedGemma experiment.

A ``ConditioningInput`` carries everything M7 needs for one image, already in
final form. The conditioner never builds overlays: it receives the raw image
and, when available, the pre-rendered overlay image (red segmentation mask
composited onto the fundus). Two providers build these inputs:

* ``oracle_inputs.build_dataset_oracle_inputs`` — the curated dataset with
  ground-truth optic-disc masks (stand-in for the classifier/segmentor).
* ``pipeline_inputs.load_pipeline_dir_inputs`` / ``load_json_inputs`` — the
  real upstream pipeline (classifier prediction + segmentor mask/overlay).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConditioningInput:
    image_id: str
    image: Any
    prediction: str
    reference: str
    mask_source: str
    source_data: str
    overlay_image: Any | None = None
    mask: Any | None = None
    expected_finding: str | None = None
    ground_truth_mask: Any | None = None


def load_json_inputs(path: str | Path) -> list[ConditioningInput]:
    """Load real pipeline outputs using one explicit JSON schema.

    Each record provides the classifier ``prediction`` and, optionally, the
    segmentor's ``overlay_image`` (pre-rendered) and/or ``mask`` (for
    segmentation scoring). Overlays are no longer derived from the mask.
    """

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Pipeline input JSON must contain a list")
    inputs = [ConditioningInput(**record) for record in records]
    for sample in inputs:
        if sample.prediction not in {"glaucoma", "normal"}:
            raise ValueError(
                f"Unsupported prediction for {sample.image_id}: "
                f"{sample.prediction!r}"
            )
    return inputs
