"""Medical BERTScore evaluation and comparisons between ablation conditions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .conditioning import CONDITIONS


def evaluate_generated_texts(
    results: list[dict[str, Any]],
    *,
    model_type: str,
    num_layers: int,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    if not results:
        return results
    from bert_score import score as bertscore

    precision, recall, f1 = bertscore(
        [item["generated"] for item in results],
        [item["reference"] for item in results],
        model_type=model_type,
        num_layers=num_layers,
        rescale_with_baseline=False,
        batch_size=batch_size,
        verbose=True,
    )
    for item, p_value, r_value, f1_value in zip(results, precision, recall, f1):
        item["bertscore_precision"] = float(p_value)
        item["bertscore_recall"] = float(r_value)
        item["bertscore_f1"] = float(f1_value)
    return results


def summarize_by_condition(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {}
    for condition in CONDITIONS:
        values = [
            float(item["bertscore_f1"])
            for item in results
            if item["condition"] == condition
        ]
        if values:
            summary[condition] = {
                "count": len(values),
                "bertscore_f1_mean": float(np.mean(values)),
                "bertscore_f1_std": float(np.std(values)),
            }

    if "A" in summary:
        baseline = summary["A"]["bertscore_f1_mean"]
        for metrics in summary.values():
            metrics["delta_f1_vs_A"] = metrics["bertscore_f1_mean"] - baseline
    return summary


def paired_comparisons_to_baseline(
    results: Sequence[Mapping[str, Any]],
    baseline_condition: str = "A",
) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    scores = {}
    for item in results:
        scores.setdefault(item["condition"], {})[item["image_id"]] = float(
            item["bertscore_f1"]
        )

    baseline = scores.get(baseline_condition)
    if not baseline:
        return {}

    comparisons = {}
    for condition in CONDITIONS:
        if condition == baseline_condition or condition not in scores:
            continue
        image_ids = sorted(set(baseline) & set(scores[condition]))
        baseline_values = np.array([baseline[image_id] for image_id in image_ids])
        condition_values = np.array([scores[condition][image_id] for image_id in image_ids])
        deltas = condition_values - baseline_values
        if np.allclose(deltas, 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(condition_values, baseline_values)
            statistic, p_value = float(test.statistic), float(test.pvalue)

        comparisons[condition] = {
            "paired_count": len(image_ids),
            "mean_delta_f1": float(np.mean(deltas)),
            "median_delta_f1": float(np.median(deltas)),
            "improved_fraction": float(np.mean(deltas > 0)),
            "wilcoxon_statistic": statistic,
            "wilcoxon_p_value": p_value,
            "significant_at_0_05": p_value < 0.05,
        }
    return comparisons
