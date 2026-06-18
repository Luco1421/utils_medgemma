"""Mock input provider backed by the current shared dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .inputs import ConditioningInput


def build_dataset_mock_inputs(
    rows: Sequence[Mapping],
    *,
    mask_key: str = "mask_disc_npy",
    confidence: float = 0.80,
) -> list[ConditioningInput]:
    """Create oracle-style 80/20 inputs from the dataset labels."""

    inputs = []
    for row in rows:
        prediction = row["target_label"]
        other = "non_glaucoma" if prediction == "glaucoma" else "glaucoma"
        inputs.append(
            ConditioningInput(
                image_id=row["image_id"],
                image=row["image"],
                mask=row[mask_key],
                prediction=prediction,
                distribution={
                    prediction: confidence,
                    other: 1.0 - confidence,
                },
                reference=(
                    row["transcription"]
                    if "transcription" in row
                    else row["reference"]
                ),
                mask_source=f"ground_truth_{mask_key.removeprefix('mask_').removesuffix('_npy')}",
                input_source="dataset_oracle_mock",
            )
        )
    return inputs
