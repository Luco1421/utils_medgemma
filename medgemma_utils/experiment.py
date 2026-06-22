"""Execute M7 conditions and evaluate them through the public M8 contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .conditioning import CONDITIONS, MASK_CONDITIONS
from .evaluation import (
    Evaluator,
    evaluate_generated_texts,
    paired_comparisons_to_baseline,
    summarize_by_condition,
)
from .inputs import ConditioningInput


def run_conditioned_experiment(
    conditioner: Any,
    evaluator: Evaluator,
    samples: Sequence[ConditioningInput],
    *,
    pipeline_name: str,
    model_variant: str,
) -> dict[str, Any]:
    """Run every M7 condition supported by each sample and evaluate with M8."""

    results = []
    segmentation_by_image: dict[str, dict[str, float] | None] = {}
    for sample in samples:
        if sample.mask is not None and sample.ground_truth_mask is not None:
            segmentation_by_image[sample.image_id] = (
                evaluator.evaluate_segmentation(
                    sample.mask,
                    sample.ground_truth_mask,
                )
            )
        else:
            segmentation_by_image[sample.image_id] = None

        for condition in CONDITIONS:
            if condition in MASK_CONDITIONS and sample.mask is None:
                continue
            m7_result = conditioner.generate(
                condition=condition,
                image_raw=sample.image,
                mask=sample.mask if condition in MASK_CONDITIONS else None,
                prediction=(
                    sample.prediction if condition in {"C1", "D1"} else None
                ),
                distribution=(
                    sample.distribution if condition in {"C2", "D2"} else None
                ),
            )
            results.append({
                "image_id": sample.image_id,
                "pipeline": pipeline_name,
                "model_variant": model_variant,
                "input_source": sample.input_source,
                "mask_source": sample.mask_source,
                "mask_target": sample.mask_target,
                "segmentation_status": sample.segmentation_status,
                "condition": m7_result["condition"],
                "generated_text": m7_result["text"],
                "reference_text": sample.reference,
                "prompt_used": m7_result["prompt_used"],
                "image_was_overlaid": m7_result["image_was_overlaid"],
                "classification": {
                    "prediction": sample.prediction,
                    "distribution": sample.distribution,
                },
                "expected_finding": sample.expected_finding,
                "segmentation_metrics": segmentation_by_image[sample.image_id],
            })

    evaluate_generated_texts(results, evaluator=evaluator)

    conditions_by_image: dict[str, set[str]] = {}
    for item in results:
        conditions_by_image.setdefault(item["image_id"], set()).add(
            item["condition"]
        )
    paired_image_ids = sorted(
        image_id
        for image_id, available in conditions_by_image.items()
        if set(CONDITIONS).issubset(available)
    )
    paired_results = [
        item for item in results if item["image_id"] in paired_image_ids
    ]
    return {
        "coverage_summary_by_condition": summarize_by_condition(
            results,
            include_delta_vs_a=False,
        ),
        "paired_image_count": len(paired_image_ids),
        "paired_image_ids": paired_image_ids,
        "summary_by_condition": summarize_by_condition(paired_results),
        "paired_comparisons_vs_A": paired_comparisons_to_baseline(
            paired_results,
            evaluator,
        ),
        "results": results,
    }
