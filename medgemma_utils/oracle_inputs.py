"""Oracle inputs used to isolate MedGemma conditioning from upstream models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .conditioning import make_overlay
from .inputs import ConditioningInput


def build_dataset_oracle_inputs(
    rows: Sequence[Mapping],
    *,
    require_mask: bool = False,
) -> list[ConditioningInput]:
    """Create M7 inputs from labels and optic-disc GT masks.

    Labels and masks from this provider are oracle signals, not outputs from a
    classifier or SAM. The overlay image is pre-rendered here from the GT disc
    mask (red composite) so the conditioner only ever receives final images.
    Normal images have no GT disc mask, so they get no overlay and the
    overlay conditions simply do not run for them in this mode.
    """

    inputs = []
    for row in rows:
        mask = row.get("ground_truth_mask")
        if require_mask and mask is None:
            raise ValueError(
                f"Dataset row {row.get('image_id', '<unknown>')} has no "
                "optic-disc ground-truth mask"
            )
        prediction = str(row["target_label"])
        has_mask = mask is not None
        overlay_image = make_overlay(row["image"], mask) if has_mask else None
        inputs.append(
            ConditioningInput(
                image_id=str(row["image_id"]),
                image=row["image"],
                overlay_image=overlay_image,
                prediction=prediction,
                reference=str(row["transcription"]),
                mask_source="dataset_GT" if has_mask else "none",
                source_data="dataset",
                expected_finding=prediction,
            )
        )
    return inputs
