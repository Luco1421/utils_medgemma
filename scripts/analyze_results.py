"""Post-hoc analysis of M7/M8 result JSONs with metrics that actually discriminate.

Reference-similarity metrics (BERTScore, sBERT, ROUGE) are dominated by the shared
report boilerplate in this domain, so they do not rank description quality. This
script adds two signal-bearing views, computed from the stored generations
(no GPU, no re-run):

- Recall@1 (image identification): for each generated text, rank the unique image
  references by ROUGE-L and check whether the text's own image ranks first. This
  measures image-specificity. Random baseline = 1 / num_images.
- Diagnostic accuracy (negation-aware): extract the stated class (glaucoma/normal)
  from the generated text and compare it to the true label. Conditions A and B do
  NOT give the class to the model, so they are the honest diagnostic test; C/D give
  the oracle class, so they only measure whether the model echoes it.

The M8-contractual metrics (bertscore_f1, sbert_similarity, finding_mentioned) are
reported as descriptive context, explicitly flagged as boilerplate-inflated.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from medgemma_utils.conditioning import BASELINE_CONDITION, CONDITIONS
from medgemma_utils.evaluation import GLAUCOMA_TERMS, NORMAL_TERMS

_WORD = re.compile(r"[a-z0-9]+")
_NEG_CUES = re.compile(
    r"\b(no|not|without|absence|absent|free|negative|denies|deny|unremarkable|"
    r"none|neither|nor|rules? out|ruled out)\b",
    flags=re.IGNORECASE,
)
_GLAUCOMA = [re.compile(p, re.IGNORECASE) for p in GLAUCOMA_TERMS]
_NORMAL = [re.compile(p, re.IGNORECASE) for p in NORMAL_TERMS]


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous_row = [0] * (len(b) + 1)
    for token_a in a:
        diagonal = 0
        current_row = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current_row[j] = diagonal + 1
            else:
                current_row[j] = max(current_row[j - 1], previous_row[j])
            diagonal = previous_row[j]
        previous_row = current_row
    return previous_row[-1]


def rouge_l(generated: str, reference: str) -> float:
    g, r = _tokens(generated), _tokens(reference)
    length = _lcs(g, r)
    if length == 0:
        return 0.0
    precision, recall = length / len(g), length / len(r)
    return 2 * precision * recall / (precision + recall)


def _is_negated(text: str, match_start: int, window: int = 45) -> bool:
    return bool(_NEG_CUES.search(text[max(0, match_start - window):match_start]))


def predicted_class(text: str) -> str:
    """Negation-aware extraction of the stated class from a generated report."""

    glaucoma_positive = any(
        any(not _is_negated(text, m.start()) for m in pattern.finditer(text))
        for pattern in _GLAUCOMA
    )
    if glaucoma_positive:
        return "glaucoma"
    normal_indicated = any(pattern.search(text) for pattern in _NORMAL)
    # A glaucoma term that is fully negated also implies a normal read.
    negated_glaucoma = any(pattern.search(text) for pattern in _GLAUCOMA)
    if normal_indicated or negated_glaucoma:
        return "normal"
    return "unknown"


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _recall_at_1(payload: dict[str, Any]) -> dict[str, float]:
    refs = {row["image_id"]: row["reference_text"] for row in payload["results"]}
    ref_ids = list(refs)
    by_condition: dict[str, list[float]] = defaultdict(list)
    overall: list[float] = []
    for row in payload["results"]:
        scores = {rid: rouge_l(row["generated_text"], refs[rid]) for rid in ref_ids}
        best = max(scores.values())
        winners = [rid for rid, sc in scores.items() if sc == best]
        hit = (1.0 / len(winners)) if row["image_id"] in winners else 0.0
        overall.append(hit)
        by_condition[row["condition"]].append(hit)
    result = {"ALL": float(np.mean(overall)), "_n_images": len(ref_ids)}
    for condition, values in by_condition.items():
        result[condition] = float(np.mean(values))
    return result


def _recall(pairs: list[tuple[str, str]], cls: str) -> float:
    items = [p for p, t in pairs if t == cls]
    if not items:
        return float("nan")
    return float(np.mean([p == cls for p in items]))


def _clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    """Exact binomial (Clopper-Pearson) confidence interval for a proportion."""

    from scipy.stats import beta

    if total == 0:
        return [float("nan"), float("nan")]
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2, successes, total - successes + 1)
    )
    upper = 1.0 if successes == total else float(
        beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    )
    return [lower, upper]


def _sensitivity_ci(pairs: list[tuple[str, str]], cls: str) -> dict[str, Any]:
    """Per-class sensitivity (recall) with an exact 95% CI."""

    truths = [(p, t) for p, t in pairs if t == cls]
    total = len(truths)
    successes = sum(1 for p, _ in truths if p == cls)
    point = float("nan") if total == 0 else successes / total
    return {
        "value": point,
        "ci95": _clopper_pearson(successes, total),
        "support": total,
    }


def _cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa between the text-asserted class and the ground truth,
    over the cases the model committed to a class (unknown excluded)."""

    decided = [(p, t) for p, t in pairs if p in {"glaucoma", "normal"}]
    n = len(decided)
    if n == 0:
        return float("nan")
    labels = ("glaucoma", "normal")
    observed = np.mean([p == t for p, t in decided])
    p_pred = {c: np.mean([p == c for p, _ in decided]) for c in labels}
    p_true = {c: np.mean([t == c for _, t in decided]) for c in labels}
    expected = sum(p_pred[c] * p_true[c] for c in labels)
    if np.isclose(expected, 1.0):
        return float("nan")
    return float((observed - expected) / (1 - expected))


def _confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Row = ground truth, column = text-asserted class (incl. ``unknown``)."""

    matrix = {
        truth: {"glaucoma": 0, "normal": 0, "unknown": 0}
        for truth in ("glaucoma", "normal")
    }
    for pred, truth in pairs:
        if truth in matrix and pred in matrix[truth]:
            matrix[truth][pred] += 1
    return matrix


def _diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
    """Per-class diagnosis quality. Accuracy alone hides class collapse, so we
    report balanced accuracy and per-class recall. The baseline condition
    (without_overlay) is the balanced, no-class-given test; the glaucoma/normal
    prompts bake the class into the text, so they only measure echoing."""

    by_condition: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in payload["results"]:
        truth = row["classification"].get("ground_truth")
        if truth not in {"glaucoma", "normal"}:
            continue
        by_condition[row["condition"]].append((predicted_class(row["generated_text"]), truth))

    a = by_condition.get(BASELINE_CONDITION, [])
    rec_g, rec_n = _recall(a, "glaucoma"), _recall(a, "normal")
    all_normal = [pair for pairs in by_condition.values() for pair in pairs if pair[1] == "normal"]
    return {
        "baseline_acc": float(np.mean([p == t for p, t in a])) if a else float("nan"),
        "baseline_balanced_acc": float(np.nanmean([rec_g, rec_n])),
        "baseline_recall_glaucoma": rec_g,
        "baseline_recall_normal": rec_n,
        "baseline_unknown": float(np.mean([p == "unknown" for p, _ in a])) if a else float("nan"),
        "normal_recall_all": _recall(all_normal, "normal"),
        # Publication-grade packaging for the honest (baseline) diagnostic probe.
        "baseline_cohens_kappa": _cohens_kappa(a),
        "baseline_sensitivity_glaucoma": _sensitivity_ci(a, "glaucoma"),
        "baseline_sensitivity_normal": _sensitivity_ci(a, "normal"),
        "baseline_confusion": _confusion(a),
    }


def _context_means(payload: dict[str, Any]) -> dict[str, float]:
    metrics = ("bertscore_f1", "sbert_similarity", "rouge_l", "bleu",
               "finding_mentioned", "sbert_calibrated")
    out = {}
    for metric in metrics:
        vals = [
            (1.0 if row["text_metrics"][metric] is True else
             0.0 if row["text_metrics"][metric] is False else
             float(row["text_metrics"][metric]))
            for row in payload["results"]
            if row["text_metrics"].get(metric) is not None
        ]
        out[metric] = float(np.mean(vals)) if vals else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="results/baseline/conditioned_base.json")
    parser.add_argument("--lora-glob", default="results/lora/test_lora_seed*.json")
    parser.add_argument("--output", default="results/analysis_extras.json")
    args = parser.parse_args()

    models: dict[str, dict[str, Any]] = {"base": _load(args.base)}
    for path in sorted(glob.glob(args.lora_glob)):
        models[Path(path).stem.replace("test_", "")] = _load(path)

    summary: dict[str, Any] = {
        "note": (
            "Extras de analisis (NO contractuales). M8 es lo principal; estas "
            "metricas existen porque las de similitud de M8 estan infladas por el "
            "boilerplate de los informes. recall_at_1 mide especificidad de imagen "
            "(azar=1/N); el diagnostico por clase delata el colapso de clase."
        ),
        "models": {},
    }
    for name, payload in models.items():
        summary["models"][name] = {
            "recall_at_1": _recall_at_1(payload),
            "diagnostic": _diagnostic(payload),
            "m8_context_means": _context_means(payload),
        }

    # Model-independent similarity control (in-category vs cross-category over
    # the expert references), computed at generation time on the base run.
    reference_baseline = models["base"].get("reference_similarity_baseline")
    if reference_baseline:
        summary["reference_similarity_baseline"] = reference_baseline

    print("\n===================== RECALL@1 (identificacion de imagen) =====================")
    print("¿el texto generado identifica SU imagen entre las referencias? (azar ~ 1/N)")
    for name, payload in models.items():
        r = _recall_at_1(payload)
        chance = 1.0 / r["_n_images"]
        conds = "  ".join(f"{c}:{r[c]:.2f}" for c in CONDITIONS if c in r)
        print(f"  [{name:12s}] ALL={r['ALL']:.3f} (azar={chance:.3f})   {conds}")

    print(f"\n============ DIAGNOSTICO por clase (condicion {BASELINE_CONDITION}, balanceada) ============")
    print(f"{BASELINE_CONDITION} no le da la clase al modelo. balanced_acc y recall por clase delatan colapso.")
    for name, payload in models.items():
        d = _diagnostic(payload)
        print(f"  [{name:12s}] balanced_acc={d['baseline_balanced_acc']:.3f}  "
              f"recall[glaucoma]={d['baseline_recall_glaucoma']:.2f}  recall[normal]={d['baseline_recall_normal']:.2f}  "
              f"unknown={d['baseline_unknown']:.2f}  | recall_normal(global)={d['normal_recall_all']:.2f}")

    print("\n===================== CONTEXTO M8 (inflado por boilerplate) ====================")
    for name, payload in models.items():
        c = _context_means(payload)
        print(f"  [{name:12s}] bertF1={c['bertscore_f1']:.3f}  sbert={c['sbert_similarity']:.3f}  "
              f"rougeL={c['rouge_l']:.3f}  bleu={c['bleu']:.3f}  "
              f"find={c['finding_mentioned']:.3f}  sbert_calib={c['sbert_calibrated']:+.3f}")

    if reference_baseline and "bertscore_f1" in reference_baseline:
        print("\n========= CROSS-VALIDATION BASELINE (referencia vs referencia) =========")
        print("in-category vs cross-category: si cross sigue alto, la metrica premia boilerplate.")
        for metric in ("bertscore_f1", "sbert_similarity", "rouge_l", "bleu"):
            block = reference_baseline[metric]
            inc, cro = block["in_category"], block["cross_category"]
            print(f"  {metric:16s} in={inc['mean']:.3f} [{inc['ci_lower']:.3f},{inc['ci_upper']:.3f}]  "
                  f"cross={cro['mean']:.3f} [{cro['ci_lower']:.3f},{cro['ci_upper']:.3f}]  "
                  f"p={block['mann_whitney']['p_value']:.1e}  delta={block['mann_whitney']['cliffs_delta']:+.2f}")

    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
