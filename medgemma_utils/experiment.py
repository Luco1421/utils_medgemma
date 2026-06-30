"""Execute M7 conditions and evaluate them through the public M8 contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .conditioning import (
    BASELINE_CONDITION,
    CONDITION_SPECS,
    condition_applies,
)
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

        for condition, spec in CONDITION_SPECS.items():
            if not condition_applies(
                spec,
                sample.prediction,
                has_overlay=sample.overlay_image is not None,
            ):
                continue
            m7_result = conditioner.generate(
                condition=condition,
                image_raw=(
                    sample.overlay_image if spec.use_overlay else sample.image
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
                },
                "expected_finding": sample.expected_finding,
                "segmentation_metrics": segmentation_by_image[sample.image_id],
            })

    evaluate_generated_texts(results, evaluator=evaluator)

    # Conditions are applied per classified label, so no image carries every
    # condition. Comparisons are made against the label-agnostic baseline
    # (``without_overlay``), which each image always receives.
    baseline_image_ids = sorted(
        {
            item["image_id"]
            for item in results
            if item["condition"] == BASELINE_CONDITION
        }
    )
    return {
        "baseline_condition": BASELINE_CONDITION,
        "coverage_summary_by_condition": summarize_by_condition(
            results,
            include_delta_vs_baseline=False,
        ),
        "baseline_image_count": len(baseline_image_ids),
        "baseline_image_ids": baseline_image_ids,
        "summary_by_condition": summarize_by_condition(results),
        "paired_comparisons_vs_baseline": paired_comparisons_to_baseline(
            results,
            evaluator,
        ),
        "results": results,
    }
