"""M8 evaluation: segmentation, clinical text metrics and statistics."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
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


_WORD_RE = re.compile(r"[a-z0-9]+")


def _word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence (row-rolled DP, O(len(b)))."""

    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        diagonal = 0
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            current[j] = diagonal + 1 if token_a == token_b else max(
                current[j - 1], previous[j]
            )
            diagonal = previous[j]
        previous = current
    return previous[-1]


def rouge_l(generated: str, reference: str) -> float:
    """ROUGE-L F1 on word tokens (LCS-based structural overlap)."""

    g, r = _word_tokens(generated), _word_tokens(reference)
    length = _lcs_length(g, r)
    if length == 0:
        return 0.0
    precision, recall = length / len(g), length / len(r)
    return float(2 * precision * recall / (precision + recall))


def sentence_bleu(generated: str, reference: str, max_n: int = 4) -> float:
    """Sentence-level BLEU-``max_n`` with a brevity penalty and epsilon
    smoothing on zero-count n-grams, so short/paraphrased clinical prose yields
    the characteristically small but non-degenerate scores reported for
    generative models."""

    g, r = _word_tokens(generated), _word_tokens(reference)
    if not g or not r:
        return 0.0
    weight = 1.0 / max_n
    log_precision = 0.0
    for n in range(1, max_n + 1):
        g_ngrams = Counter(zip(*(g[i:] for i in range(n))))
        r_ngrams = Counter(zip(*(r[i:] for i in range(n))))
        overlap = sum(
            min(count, r_ngrams[ngram]) for ngram, count in g_ngrams.items()
        )
        total = max(len(g) - n + 1, 1)
        precision = overlap / total if overlap > 0 else 1e-9
        log_precision += weight * math.log(precision)
    brevity = 1.0 if len(g) > len(r) else math.exp(1.0 - len(r) / len(g))
    return float(brevity * math.exp(log_precision))


# Cup-to-disc value, the single scalar that carries most of the glaucoma
# diagnosis in these reports (a shared phrase whose *number* differs by class).
_CDR_VALUE_RE = re.compile(
    r"cup[- ]to[- ]disc(?:\s+ratio)?(?:\s*\(cdr\))?\s*"
    r"(?:of|is|:|=|approximately|~)?\s*(\d(?:\.\d+)?)",
    re.IGNORECASE,
)
# Clauses that carry graded diagnostic content rather than shared anatomy.
_DIAGNOSTIC_MARKERS = (
    _CDR_VALUE_RE,
    re.compile(r"\bcupping\b", re.IGNORECASE),
    re.compile(r"\bbayonet\w*", re.IGNORECASE),
    re.compile(r"\bnasalization\b", re.IGNORECASE),
    re.compile(r"\bdisc hemorrhage\b", re.IGNORECASE),
    re.compile(r"\b(?:rnfl|retinal nerve fiber layer)\b", re.IGNORECASE),
    re.compile(r"\brim\b[^,.;]*\b(?:thinning|notch\w*|loss)\b", re.IGNORECASE),
    re.compile(r"\b(?:thinning|notch\w*|loss)\b[^,.;]*\brim\b", re.IGNORECASE),
    re.compile(r"\bpallor\b", re.IGNORECASE),
    re.compile(r"\bperipapillary atroph\w*", re.IGNORECASE),
    re.compile(r"\bISNT\b", re.IGNORECASE),
    re.compile(r"\bpreserved (?:neuroretinal )?rim\b", re.IGNORECASE),
    re.compile(r"\bno (?:vascular )?abnormalit\w*", re.IGNORECASE),
    re.compile(r"\bno signs? of glaucoma\b", re.IGNORECASE),
    re.compile(r"\bwithout glaucomatous\b", re.IGNORECASE),
    re.compile(r"\bglaucom\w*", re.IGNORECASE),
)
# Split on comma/semicolon/newline, and on a period only when it is not a
# decimal point, so a value such as "0.6" is never broken across clauses.
_CLAUSE_SPLIT = re.compile(r"(?<!\d)\.(?!\d)|[,;\n]")

_CDR_GRADE_TO_RATIO = {0: 0.2, 1: 0.45, 2: 0.65, 3: 0.85, 4: 1.0}


def diagnostic_span(text: str) -> str:
    """Return the diagnostic clauses of a report, dropping the shared template.

    Each report is split into clauses; a clause is kept only if it states a
    cup-to-disc value or an explicit finding (rim thinning, RNFL defect,
    bayoneting, an ``ISNT``/normal statement, etc.). The anatomical scaffold
    present in every report -- bare mentions of the disc, macula or vessels --
    is discarded. Restricting a similarity metric to this span isolates how much
    of its score reflects clinical content rather than report structure.
    """

    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(text) if c.strip()]
    kept = [c for c in clauses if any(p.search(c) for p in _DIAGNOSTIC_MARKERS)]
    return ", ".join(kept)


def reference_cdr(text: str) -> float | None:
    """Cup-to-disc value stated in a report, normalized to a 0--1 ratio; integer
    grades 0--4 are mapped to their schema ratios. ``None`` if none is stated."""

    match = _CDR_VALUE_RE.search(text)
    if not match:
        return None
    raw = match.group(1)
    value = float(raw)
    if "." in raw:
        return value
    return _CDR_GRADE_TO_RATIO.get(int(value)) if value <= 4 else None


def cdr_class_separation(references: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    """How well the stated cup-to-disc value alone separates the two classes.

    Reports the per-class value distribution and the rank-based separability
    (the probability that a random glaucoma report states a higher value than a
    random normal one -- equivalently the ROC AUC of the scalar), evidence that
    the diagnostic signal these similarity metrics ignore is a single number.
    """

    values = [(reference_cdr(t), l) for t, l in zip(references, labels)]
    per_class: dict[str, list[float]] = {"glaucoma": [], "normal": []}
    for value, label in values:
        if value is not None and label in per_class:
            per_class[label].append(value)
    g, n = per_class["glaucoma"], per_class["normal"]
    if g and n:
        wins = sum((gv > nv) + 0.5 * (gv == nv) for gv in g for nv in n)
        auc = float(wins / (len(g) * len(n)))
    else:
        auc = float("nan")
    return {
        "stated_fraction": float(np.mean([v is not None for v, _ in values])) if values else float("nan"),
        "glaucoma": {"n": len(g), "mean": float(np.mean(g)) if g else float("nan"),
                     "min": float(min(g)) if g else float("nan"),
                     "max": float(max(g)) if g else float("nan")},
        "normal": {"n": len(n), "mean": float(np.mean(n)) if n else float("nan"),
                   "min": float(min(n)) if n else float("nan"),
                   "max": float(max(n)) if n else float("nan")},
        "roc_auc": auc,
    }


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
                "rouge_l": rouge_l(generated_texts[index], reference_texts[index]),
                "bleu": sentence_bleu(
                    generated_texts[index], reference_texts[index]
                ),
                "finding_mentioned": finding_mentioned(
                    generated_texts[index],
                    expected_findings[index],
                ),
                "likert_score": None,
            }
            for index in range(len(generated_texts))
        ]

    def reference_baseline(
        self,
        references: Sequence[str],
        labels: Sequence[str],
        *,
        bootstrap_resamples: int = 5000,
    ) -> dict[str, Any]:
        """Cross-validation control for the similarity metrics.

        Computes every ordered pair of expert references (``i != j``) and splits
        the pairwise scores into *in-category* (same diagnostic label) and
        *cross-category* (different label). A cross-category score that stays
        high reveals the shared report boilerplate that inflates the raw
        metrics -- the ``illusion of efficacy``. Returns, per metric, the two
        means with bootstrap 95% CIs and a Mann-Whitney U test between them.
        """

        if len(references) != len(labels):
            raise ValueError("references and labels must be aligned")
        if len(references) < 2:
            raise ValueError("need at least two references")
        return self._pairwise_baseline(
            list(references), list(labels), bootstrap_resamples=bootstrap_resamples
        )

    def _pairwise_baseline(
        self,
        references: Sequence[str],
        labels: Sequence[str],
        *,
        bootstrap_resamples: int,
    ) -> dict[str, Any]:
        count = len(references)
        cand_idx: list[int] = []
        ref_idx: list[int] = []
        for i in range(count):
            for j in range(count):
                if i != j:
                    cand_idx.append(i)
                    ref_idx.append(j)

        candidate_texts = [references[i] for i in cand_idx]
        reference_texts = [references[j] for j in ref_idx]
        _, _, bert_f1 = self._bertscore(candidate_texts, reference_texts)
        embeddings = self._encode_sbert(list(references))
        sbert_pairs = np.sum(
            embeddings[cand_idx] * embeddings[ref_idx], axis=1
        )
        rouge_pairs = np.asarray(
            [rouge_l(candidate_texts[k], reference_texts[k])
             for k in range(len(cand_idx))]
        )
        bleu_pairs = np.asarray(
            [sentence_bleu(candidate_texts[k], reference_texts[k])
             for k in range(len(cand_idx))]
        )
        same_category = np.asarray(
            [labels[i] == labels[j] for i, j in zip(cand_idx, ref_idx)]
        )

        metric_arrays = {
            "bertscore_f1": np.asarray(bert_f1, dtype=float),
            "sbert_similarity": sbert_pairs,
            "rouge_l": rouge_pairs,
            "bleu": bleu_pairs,
        }
        rng = np.random.default_rng(self.seed)
        report: dict[str, Any] = {
            "pair_count": len(cand_idx),
            "in_category_pairs": int(same_category.sum()),
            "cross_category_pairs": int((~same_category).sum()),
        }
        for metric, values in metric_arrays.items():
            in_values = values[same_category]
            cross_values = values[~same_category]
            report[metric] = {
                "in_category": _summarize_baseline(in_values, rng, bootstrap_resamples),
                "cross_category": _summarize_baseline(cross_values, rng, bootstrap_resamples),
                "mann_whitney": _mann_whitney(in_values, cross_values),
            }
        return report

    def boilerplate_dilution(
        self,
        references: Sequence[str],
        labels: Sequence[str],
        *,
        bootstrap_resamples: int = 5000,
    ) -> dict[str, Any]:
        """Quantify how much of the cross-validation similarity is boilerplate.

        Recomputes the pairwise reference baseline on the full text and on the
        diagnostic span alone (:func:`diagnostic_span`). If the cross-category
        similarity survives the stripping, the metric was rewarding shared report
        structure rather than diagnostic content. Also reports how well the
        stated cup-to-disc value by itself separates the classes, the scalar the
        similarity metrics are blind to.
        """

        if len(references) != len(labels):
            raise ValueError("references and labels must be aligned")
        if len(references) < 2:
            raise ValueError("need at least two references")
        references = list(references)
        labels = list(labels)
        spans = [diagnostic_span(text) or "no findings stated" for text in references]
        retained = float(np.mean([
            len(diagnostic_span(text)) / max(len(text), 1) for text in references
        ]))
        return {
            "full": self._pairwise_baseline(
                references, labels, bootstrap_resamples=bootstrap_resamples
            ),
            "diagnostic_span": self._pairwise_baseline(
                spans, labels, bootstrap_resamples=bootstrap_resamples
            ),
            "span_retained_char_fraction": retained,
            "cdr_class_separation": cdr_class_separation(references, labels),
        }

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


def _summarize_baseline(
    values: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, float]:
    """Mean of ``values`` with a percentile bootstrap 95% CI."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan"), "n": 0}
    if resamples > 0 and values.size > 1:
        draws = rng.integers(0, values.size, size=(resamples, values.size))
        means = values[draws].mean(axis=1)
        ci_lower, ci_upper = np.percentile(means, [2.5, 97.5])
    else:
        ci_lower = ci_upper = float(values.mean())
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n": int(values.size),
    }


def _mann_whitney(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Two-sided Mann-Whitney U with Cliff's delta effect size."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return {"u_statistic": float("nan"), "p_value": float("nan"),
                "cliffs_delta": float("nan")}
    from scipy.stats import mannwhitneyu

    result = mannwhitneyu(a, b, alternative="two-sided")
    # Cliff's delta from U: delta = 2U/(n1 n2) - 1.
    cliffs = 2.0 * float(result.statistic) / (a.size * b.size) - 1.0
    return {
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "cliffs_delta": float(cliffs),
    }


TEXT_METRICS = (
    "bertscore_f1",
    "bertscore_calibrated",
    "sbert_similarity",
    "sbert_calibrated",
    "rouge_l",
    "bleu",
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
        [item["classification"].get("ground_truth") for item in results],
        [item["image_id"] for item in results],
    )
    for item, text_metrics in zip(results, metrics):
        item["text_metrics"] = text_metrics
    return results
