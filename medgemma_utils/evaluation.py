"""M8 evaluation: segmentation, clinical text metrics and statistics."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .conditioning import BASELINE_CONDITION, CONDITIONS, load_mask

DEFAULT_BERTSCORE_MODEL = (
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)
DEFAULT_SBERT_MODEL = "NeuML/bioclinical-modernbert-base-embeddings"

GLAUCOMA_TERMS = (
    r"\bglaucom\w*",
    r"\bcupping\b",
    r"\bcup[- ]to[- ]disc\b",
    r"\bc\/d ratio\b",
    r"\bneuroretinal rim\b",
    r"\brnfl\b",
    r"\bretinal nerve fiber layer\b",
    r"\bbayonet\w*",
    r"\bnasalization\b",
    r"\bdisc hemorrhage\b",
)
NORMAL_TERMS = (
    r"\bnormal\b",
    r"\bno (?:vascular )?abnormalit\w*",
    r"\bno signs? of glaucoma\b",
    r"\bwithout glaucomatous\b",
    r"\bpreserved neuroretinal rim\b",
)


def _binary_mask(mask: Any) -> np.ndarray:
    array = load_mask(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, found shape {array.shape}")
    return array.astype(bool)


def _ssim(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute SSIM using the standard local Gaussian-window formulation."""

    from scipy.ndimage import gaussian_filter

    a = mask_a.astype(np.float64)
    b = mask_b.astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = gaussian_filter(a, sigma=1.5)
    mu_b = gaussian_filter(b, sigma=1.5)
    sigma_a = gaussian_filter(a * a, sigma=1.5) - mu_a * mu_a
    sigma_b = gaussian_filter(b * b, sigma=1.5) - mu_b * mu_b
    sigma_ab = gaussian_filter(a * b, sigma=1.5) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (
        (mu_a * mu_a + mu_b * mu_b + c1)
        * (sigma_a + sigma_b + c2)
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def finding_mentioned(text: str, expected_finding: str | None) -> bool:
    """Heuristic sanity metric required by M8."""

    if not expected_finding:
        return False
    expected = expected_finding.casefold().replace("-", "_")
    if expected in {"glaucoma", "pathological"}:
        return _contains_any(text, GLAUCOMA_TERMS)
    if expected == "normal":
        return _contains_any(text, NORMAL_TERMS)
    return expected.replace("_", " ") in text.casefold()


class Evaluator:
    """Reference-compatible M8 evaluator with lazy model loading."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        config = dict(config or {})
        self.config = config
        self.bertscore_model = config.get(
            "bertscore_model",
            DEFAULT_BERTSCORE_MODEL,
        )
        self.bertscore_num_layers = int(config.get("bertscore_num_layers", 12))
        self.sbert_model = config.get("sbert_model", DEFAULT_SBERT_MODEL)
        self.significance_level = float(config.get("significance_level", 0.05))
        self.seed = int(config.get("seed", 42))
        self.token = config.get("token")
        self.batch_size = int(config.get("batch_size", 16))
        self.baseline_permutations = int(
            config.get("baseline_permutations", 5)
        )
        self._sbert_tokenizer = None
        self._sbert_encoder = None
        self._bert_scorer = None

    def evaluate_segmentation(
        self,
        pred_mask: Any,
        gt_mask: Any,
    ) -> dict[str, float]:
        prediction = _binary_mask(pred_mask)
        ground_truth = _binary_mask(gt_mask)
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"Mask shapes differ: {prediction.shape} vs {ground_truth.shape}"
            )

        intersection = np.logical_and(prediction, ground_truth).sum()
        union = np.logical_or(prediction, ground_truth).sum()
        prediction_area = prediction.sum()
        ground_truth_area = ground_truth.sum()
        iou = 1.0 if union == 0 else intersection / union
        denominator = prediction_area + ground_truth_area
        dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
        return {
            "iou": float(iou),
            "dice": float(dice),
            "ssim": _ssim(prediction, ground_truth),
        }

    def _bertscore(
        self,
        generated: Sequence[str],
        references: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._bert_scorer is None:
            import torch
            from bert_score import BERTScorer

            self._bert_scorer = BERTScorer(
                model_type=self.bertscore_model,
                num_layers=self.bertscore_num_layers,
                rescale_with_baseline=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        precision, recall, f1 = self._bert_scorer.score(
            self._truncate_for_bert(generated),
            self._truncate_for_bert(references),
            batch_size=self.batch_size,
        )
        return (
            precision.cpu().numpy(),
            recall.cpu().numpy(),
            f1.cpu().numpy(),
        )

    def _truncate_for_bert(self, texts: Sequence[str]) -> list[str]:
        """Clip texts so BERTScore stays within the model's 512-token limit.

        Untuned base outputs can exceed BiomedBERT's max position embeddings
        (512), which crashes the encoder. We re-tokenize with the scorer's own
        tokenizer and keep a margin for the [CLS]/[SEP] special tokens.
        """
        tokenizer = self._bert_scorer._tokenizer
        max_content = 510
        clipped: list[str] = []
        for text in texts:
            ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_content,
            )
            clipped.append(tokenizer.decode(ids, skip_special_tokens=True))
        return clipped

    def _load_sbert(self):
        if self._sbert_encoder is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._sbert_tokenizer = AutoTokenizer.from_pretrained(
            self.sbert_model,
            token=self.token,
        )
        self._sbert_encoder = AutoModel.from_pretrained(
            self.sbert_model,
            token=self.token,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sbert_encoder = self._sbert_encoder.to(device).eval()

    def _encode_sbert(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        self._load_sbert()
        assert self._sbert_tokenizer is not None
        assert self._sbert_encoder is not None
        device = next(self._sbert_encoder.parameters()).device
        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            inputs = self._sbert_tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self._sbert_encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand_as(output)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def _mismatched_indices(
        self,
        group_ids: Sequence[str],
    ) -> list[list[int]]:
        count = len(group_ids)
        if count < 2 or self.baseline_permutations <= 0:
            return []
        if len(set(group_ids)) < 2:
            return []
        rng = random.Random(self.seed)
        baselines = []
        original = list(range(count))
        attempts = 0
        while len(baselines) < self.baseline_permutations and attempts < 1000:
            candidate = original.copy()
            rng.shuffle(candidate)
            attempts += 1
            if all(
                group_ids[index] != group_ids[value]
                for index, value in enumerate(candidate)
            ):
                baselines.append(candidate)
        return baselines

    @staticmethod
    def _calibrated(raw: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        denominator = np.maximum(1.0 - baseline, 1e-8)
        return (raw - baseline) / denominator

    def evaluate_text_batch(
        self,
        generated_texts: Sequence[str],
        reference_texts: Sequence[str],
        expected_findings: Sequence[str | None] | None = None,
        group_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if len(generated_texts) != len(reference_texts):
            raise ValueError("Generated and reference text counts differ")
        if not generated_texts:
            return []
        if expected_findings is None:
            expected_findings = [None] * len(generated_texts)
        if len(expected_findings) != len(generated_texts):
            raise ValueError("Expected-finding count differs from text count")
        if group_ids is None:
            group_ids = [str(index) for index in range(len(generated_texts))]
        if len(group_ids) != len(generated_texts):
            raise ValueError("Group-ID count differs from text count")

        precision, recall, f1 = self._bertscore(
            generated_texts,
            reference_texts,
        )
        generated_embeddings = self._encode_sbert(generated_texts)
        reference_embeddings = self._encode_sbert(reference_texts)
        sbert = np.sum(generated_embeddings * reference_embeddings, axis=1)

        permutations = self._mismatched_indices(group_ids)
        if permutations:
            bert_baselines = []
            sbert_baselines = []
            for permutation in permutations:
                mismatch_refs = [reference_texts[index] for index in permutation]
                _, _, mismatch_f1 = self._bertscore(
                    generated_texts,
                    mismatch_refs,
                )
                bert_baselines.append(mismatch_f1)
                mismatch_embeddings = reference_embeddings[permutation]
                sbert_baselines.append(
                    np.sum(generated_embeddings * mismatch_embeddings, axis=1)
                )
            bert_baseline = np.mean(bert_baselines, axis=0)
            sbert_baseline = np.mean(sbert_baselines, axis=0)
        else:
            bert_baseline = np.zeros_like(f1)
            sbert_baseline = np.zeros_like(sbert)

        bert_calibrated = self._calibrated(f1, bert_baseline)
        sbert_calibrated = self._calibrated(sbert, sbert_baseline)
        return [
            {
                "bertscore_precision": float(precision[index]),
                "bertscore_recall": float(recall[index]),
                "bertscore_f1": float(f1[index]),
                "bertscore_random_baseline": float(bert_baseline[index]),
                "bertscore_calibrated": float(bert_calibrated[index]),
                "sbert_similarity": float(sbert[index]),
                "sbert_random_baseline": float(sbert_baseline[index]),
                "sbert_calibrated": float(sbert_calibrated[index]),
                "finding_mentioned": finding_mentioned(
                    generated_texts[index],
                    expected_findings[index],
                ),
                "likert_score": None,
            }
            for index in range(len(generated_texts))
        ]

    def evaluate_text(
        self,
        generated_text: str,
        reference_text: str,
        expected_finding: str | None = None,
    ) -> dict[str, Any]:
        return self.evaluate_text_batch(
            [generated_text],
            [reference_text],
            [expected_finding],
            ["single"],
        )[0]

    def statistical_test(
        self,
        scores_a: Sequence[float],
        scores_b: Sequence[float],
    ) -> dict[str, Any]:
        from scipy.stats import norm, wilcoxon

        if len(scores_a) != len(scores_b):
            raise ValueError("Wilcoxon inputs must be paired")
        if not scores_a:
            raise ValueError("Wilcoxon inputs cannot be empty")
        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)
        deltas = b - a
        if np.allclose(deltas, 0):
            statistic = 0.0
            p_value = 1.0
            effect_size = 0.0
        else:
            result = wilcoxon(b, a)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            z_value = float(norm.isf(p_value / 2.0))
            effect_size = z_value / math.sqrt(len(deltas))
        return {
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.significance_level,
            "effect_size": effect_size,
        }


TEXT_METRICS = (
    "bertscore_f1",
    "bertscore_calibrated",
    "sbert_similarity",
    "sbert_calibrated",
    "finding_mentioned",
)


def summarize_by_condition(
    results: Sequence[Mapping[str, Any]],
    *,
    include_delta_vs_baseline: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [item for item in results if item["condition"] == condition]
        if not rows:
            continue
        metrics: dict[str, Any] = {"count": len(rows)}
        for metric in TEXT_METRICS:
            values = [
                float(item["text_metrics"][metric])
                for item in rows
                if item.get("text_metrics", {}).get(metric) is not None
            ]
            if values:
                metrics[f"{metric}_mean"] = float(np.mean(values))
                metrics[f"{metric}_std"] = float(np.std(values))
        summary[condition] = metrics

    if include_delta_vs_baseline and BASELINE_CONDITION in summary:
        baseline = summary[BASELINE_CONDITION]
        for condition, metrics in summary.items():
            if condition == BASELINE_CONDITION:
                continue
            for metric in TEXT_METRICS:
                key = f"{metric}_mean"
                if key in metrics and key in baseline:
                    metrics[f"delta_{metric}_vs_baseline"] = (
                        metrics[key] - baseline[key]
                    )
    return summary


def paired_comparisons_to_baseline(
    results: Sequence[Mapping[str, Any]],
    evaluator: Evaluator,
    baseline_condition: str = BASELINE_CONDITION,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for condition in CONDITIONS:
        if condition == baseline_condition:
            continue
        comparisons[condition] = {}
        for metric in (
            "bertscore_f1",
            "bertscore_calibrated",
            "sbert_similarity",
            "sbert_calibrated",
        ):
            baseline = {
                item["image_id"]: float(item["text_metrics"][metric])
                for item in results
                if item["condition"] == baseline_condition
            }
            candidate = {
                item["image_id"]: float(item["text_metrics"][metric])
                for item in results
                if item["condition"] == condition
            }
            image_ids = sorted(set(baseline) & set(candidate))
            if not image_ids:
                continue
            baseline_values = [baseline[image_id] for image_id in image_ids]
            candidate_values = [candidate[image_id] for image_id in image_ids]
            deltas = np.asarray(candidate_values) - np.asarray(baseline_values)
            comparisons[condition][metric] = {
                "paired_count": len(image_ids),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "improved_fraction": float(np.mean(deltas > 0)),
                **evaluator.statistical_test(
                    baseline_values,
                    candidate_values,
                ),
            }
    for metric in (
        "bertscore_f1",
        "bertscore_calibrated",
        "sbert_similarity",
        "sbert_calibrated",
    ):
        entries = [
            (condition, metrics[metric]["p_value"])
            for condition, metrics in comparisons.items()
            if metric in metrics
        ]
        ordered = sorted(entries, key=lambda item: item[1])
        adjusted: dict[str, float] = {}
        running = 0.0
        total = len(ordered)
        for rank, (condition, p_value) in enumerate(ordered):
            corrected = min(1.0, p_value * (total - rank))
            running = max(running, corrected)
            adjusted[condition] = running
        for condition, p_value in adjusted.items():
            comparisons[condition][metric]["p_value_holm"] = p_value
            comparisons[condition][metric]["significant_holm"] = (
                p_value < evaluator.significance_level
            )
    return comparisons


def evaluate_generated_texts(
    results: list[dict[str, Any]],
    *,
    evaluator: Evaluator,
) -> list[dict[str, Any]]:
    metrics = evaluator.evaluate_text_batch(
        [item["generated_text"] for item in results],
        [item["reference_text"] for item in results],
        [item.get("expected_finding") for item in results],
        [item["image_id"] for item in results],
    )
    for item, text_metrics in zip(results, metrics):
        item["text_metrics"] = text_metrics
    return results
