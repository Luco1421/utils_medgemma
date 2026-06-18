"""Run and evaluate all six M7 conditions for base or LoRA MedGemma."""

from __future__ import annotations

from typing import Any, Sequence

from .conditioning import CONDITIONS
from .evaluation import (
    evaluate_generated_texts,
    paired_comparisons_to_baseline,
    summarize_by_condition,
)
from .inputs import ConditioningInput


def run_conditioned_experiment(
    conditioner: Any,
    samples: Sequence[ConditioningInput],
    *,
    model_variant: str,
    bertscore_model: str,
    bertscore_num_layers: int,
) -> dict[str, Any]:
    results = []
    for sample in samples:
        for condition in CONDITIONS:
            generated = conditioner.generate(
                condition=condition,
                image_raw=sample.image,
                mask=sample.mask if condition in {"B", "D1", "D2"} else None,
                prediction=sample.prediction if condition in {"C1", "D1"} else None,
                distribution=sample.distribution if condition in {"C2", "D2"} else None,
            )
            results.append({
                "image_id": sample.image_id,
                "model_variant": model_variant,
                "input_source": sample.input_source,
                "mask_source": sample.mask_source,
                "prediction": sample.prediction,
                "distribution": sample.distribution,
                "reference": sample.reference,
                "generated": generated["text"],
                **generated,
            })

    evaluate_generated_texts(
        results,
        model_type=bertscore_model,
        num_layers=bertscore_num_layers,
    )
    return {
        "summary_by_condition": summarize_by_condition(results),
        "paired_comparisons_vs_A": paired_comparisons_to_baseline(results),
        "results": results,
    }
