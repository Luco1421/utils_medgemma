"""Oracle inputs used to isolate MedGemma conditioning from upstream models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .inputs import ConditioningInput


def build_dataset_oracle_inputs(
    rows: Sequence[Mapping],
    *,
    confidence: float = 0.80,
    require_mask: bool = False,
) -> list[ConditioningInput]:
    """Create M7 inputs from labels and optic-disc GT masks.

    Classification values and masks from this provider are oracle signals, not
    outputs from CNN or SAM. Consequently, no segmentation score is calculated
    by comparing the conditioning mask against itself.
    """

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    inputs = []
    for row in rows:
        mask = row.get("ground_truth_mask")
        if require_mask and not mask:
            raise ValueError(
                f"Dataset row {row.get('image_id', '<unknown>')} has no "
                "optic-disc ground-truth mask"
            )
        prediction = str(row["target_label"])
        other = "normal" if prediction == "glaucoma" else "glaucoma"
        inputs.append(
            ConditioningInput(
                image_id=str(row["image_id"]),
                image=row["image"],
                mask=mask,
                prediction=prediction,
                distribution={
                    prediction: confidence,
                    other: 1.0 - confidence,
                },
                reference=str(row["transcription"]),
                mask_source=(
                    "ground_truth_optic_disc"
                    if mask
                    else "not_applicable_normal"
                ),
                input_source="dataset_oracle",
                expected_finding=prediction,
                ground_truth_mask=None,
                mask_target=str(row["mask_target"]),
                segmentation_status=(
                    "ground_truth_conditioning"
                    if mask
                    else "bypassed_normal"
                ),
            )
        )
    return inputs
