"""Evaluacion de descripciones generadas por MedGemma."""

from __future__ import annotations

import logging
from typing import Any

from bert_score import score as bertscore

logger = logging.getLogger(__name__)


class DescriptionEvaluator:
    """Calcula BERTScore y comparaciones agregadas."""

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Config global del experimento.
        """
        self.config = config
        self.eval_config = config["evaluation"]

    def evaluate_outputs(
        self,
        outputs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Evalua una lista de outputs con BERTScore."""
        candidates = [item["generated"] for item in outputs]
        references = [item["reference"] for item in outputs]

        logger.info("Calculando BERTScore para %d descripciones", len(outputs))
        precision, recall, f1 = bertscore(
            candidates,
            references,
            lang=self.eval_config["bertscore_lang"],
            rescale_with_baseline=bool(self.eval_config["rescale_with_baseline"]),
            verbose=True,
        )

        for item, p_value, r_value, f_value in zip(outputs, precision, recall, f1):
            item["bertscore_precision"] = float(p_value)
            item["bertscore_recall"] = float(r_value)
            item["bertscore_f1"] = float(f_value)

        summary = {
            "count": len(outputs),
            "bertscore_precision_mean": float(precision.mean()),
            "bertscore_recall_mean": float(recall.mean()),
            "bertscore_f1_mean": float(f1.mean()),
        }
        return summary, outputs

    def compare(self, base_summary: dict[str, Any], lora_summary: dict[str, Any]) -> dict[str, Any]:
        """Compara resumen base contra resumen LoRA."""
        return {
            "base": base_summary,
            "lora": lora_summary,
            "delta_f1": lora_summary["bertscore_f1_mean"]
            - base_summary["bertscore_f1_mean"],
            "delta_precision": lora_summary["bertscore_precision_mean"]
            - base_summary["bertscore_precision_mean"],
            "delta_recall": lora_summary["bertscore_recall_mean"]
            - base_summary["bertscore_recall_mean"],
        }
