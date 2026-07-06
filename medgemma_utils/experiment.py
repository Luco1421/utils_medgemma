"""Run the conditioning conditions per sample and evaluate the generations."""

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
    model_variant: str,
) -> dict[str, Any]:
    """Run every M7 condition supported by each sample and evaluate with M8."""

    results = []
    for sample in samples:
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
                "source_data": sample.source_data,
                "model_variant": model_variant,
                "mask_source": sample.mask_source,
                "condition": m7_result["condition"],
                "generated_text": m7_result["text"],
                "reference_text": sample.reference,
                "prompt_used": m7_result["prompt_used"],
                "classification": {
                    "predicted": sample.prediction,
                    "ground_truth": sample.expected_finding,
                },
            })

    evaluate_generated_texts(results, evaluator=evaluator)

    # Conditions are applied per classified label, so no image carries every
    # condition. Comparisons are made against the label-agnostic baseline
    # (``without_overlay``), which each image always receives.
    baseline_image_count = len(
        {
            item["image_id"]
            for item in results
            if item["condition"] == BASELINE_CONDITION
        }
    )
    return {
        "baseline_image_count": baseline_image_count,
        "summary_by_condition": summarize_by_condition(results),
        "paired_comparisons_vs_baseline": paired_comparisons_to_baseline(
            results,
            evaluator,
        ),
        "reference_similarity_baseline": _reference_similarity_baseline(
            results,
            evaluator,
        ),
        "boilerplate_dilution": _boilerplate_dilution(results, evaluator),
        "results": results,
    }


def _unique_references(
    results: Sequence[dict[str, Any]],
) -> tuple[list[str], list[str]] | None:
    """Unique expert references and their labels, or ``None`` if fewer than two
    references span both classes."""

    unique: dict[str, tuple[str, str]] = {}
    for item in results:
        label = item["classification"].get("ground_truth")
        if label in ("glaucoma", "normal") and item["image_id"] not in unique:
            unique[item["image_id"]] = (item["reference_text"], label)
    if len(unique) < 2 or len({label for _, label in unique.values()}) < 2:
        return None
    references = [text for text, _ in unique.values()]
    labels = [label for _, label in unique.values()]
    return references, labels


def _reference_similarity_baseline(
    results: Sequence[dict[str, Any]],
    evaluator: Evaluator,
) -> dict[str, Any]:
    """In-category vs cross-category baseline over the unique expert references.

    Model-independent (references only), so it is identical across conditions
    and seeds; computed here because it needs the GPU-backed encoders. Guarded
    so a failure never aborts a generation run.
    """

    refs = _unique_references(results)
    if refs is None:
        return {"skipped": "need >=2 references spanning both classes"}
    try:
        return evaluator.reference_baseline(*refs)
    except Exception as error:  # noqa: BLE001 - never abort a run for this extra
        return {"error": str(error)}


def _boilerplate_dilution(
    results: Sequence[dict[str, Any]],
    evaluator: Evaluator,
) -> dict[str, Any]:
    """Boilerplate-dilution control (full vs diagnostic-span similarity plus the
    cup-to-disc class separation) over the unique expert references. Same
    guarded, model-independent contract as the reference baseline."""

    refs = _unique_references(results)
    if refs is None:
        return {"skipped": "need >=2 references spanning both classes"}
    try:
        return evaluator.boilerplate_dilution(*refs)
    except Exception as error:  # noqa: BLE001 - never abort a run for this extra
        return {"error": str(error)}
