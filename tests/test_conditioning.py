import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from medgemma_utils.conditioning import (
    BASELINE_CONDITION,
    CONDITION_SPECS,
    condition_applies,
    load_condition_specs,
    make_overlay,
)
from medgemma_utils.evaluation import (
    Evaluator,
    cdr_class_separation,
    diagnostic_span,
    finding_mentioned,
    paired_comparisons_to_baseline,
    reference_cdr,
    rouge_l,
    sentence_bleu,
    summarize_by_condition,
)
from medgemma_utils.experiment import run_conditioned_experiment
from medgemma_utils.inputs import ConditioningInput, load_json_inputs
from medgemma_utils.oracle_inputs import build_dataset_oracle_inputs
from medgemma_utils.pipeline_inputs import load_pipeline_dir_inputs


class ConditionSpecTests(unittest.TestCase):
    def test_six_conditions_with_expected_axes(self):
        specs = load_condition_specs()
        self.assertEqual(
            set(specs),
            {
                "with_overlay_glaucoma",
                "without_overlay_glaucoma",
                "with_overlay_normal",
                "without_overlay_normal",
                "with_overlay",
                "without_overlay",
            },
        )
        self.assertTrue(specs["with_overlay_glaucoma"].use_overlay)
        self.assertFalse(specs["without_overlay"].use_overlay)
        self.assertEqual(specs["with_overlay_glaucoma"].label_scope, "glaucoma")
        self.assertEqual(specs["without_overlay_normal"].label_scope, "normal")
        self.assertEqual(specs["with_overlay"].label_scope, "any")
        self.assertIn(BASELINE_CONDITION, specs)

    def test_prompts_are_static_without_placeholders(self):
        for spec in CONDITION_SPECS.values():
            self.assertNotIn("{prediction}", spec.prompt)
            self.assertNotIn("{distribution}", spec.prompt)

    def test_condition_applies_respects_label_and_overlay(self):
        glaucoma = CONDITION_SPECS["with_overlay_glaucoma"]
        normal = CONDITION_SPECS["with_overlay_normal"]
        generic_overlay = CONDITION_SPECS["with_overlay"]
        baseline = CONDITION_SPECS["without_overlay"]

        # Glaucoma-only prompt never runs on a normal image.
        self.assertFalse(condition_applies(glaucoma, "normal", has_overlay=True))
        # Overlay requires an overlay image.
        self.assertFalse(condition_applies(glaucoma, "glaucoma", has_overlay=False))
        self.assertTrue(condition_applies(glaucoma, "glaucoma", has_overlay=True))
        # Normal images CAN now take an overlay (segmentor runs on them).
        self.assertTrue(condition_applies(normal, "normal", has_overlay=True))
        self.assertFalse(condition_applies(normal, "normal", has_overlay=False))
        # Generic prompts run on any label.
        self.assertTrue(condition_applies(generic_overlay, "normal", has_overlay=True))
        self.assertTrue(condition_applies(baseline, "glaucoma", has_overlay=False))


class InputProviderTests(unittest.TestCase):
    def test_dataset_oracle_uses_label_without_distribution(self):
        rows = [
            {
                "image_id": "case",
                "image": np.zeros((4, 4, 3), dtype=np.uint8),
                "ground_truth_mask": np.ones((4, 4), dtype=bool),
                "target_label": "glaucoma",
                "transcription": "reference",
            }
        ]
        sample = build_dataset_oracle_inputs(rows)[0]
        self.assertEqual(sample.prediction, "glaucoma")
        self.assertFalse(hasattr(sample, "distribution"))
        self.assertEqual(sample.source_data, "dataset")
        self.assertEqual(sample.mask_source, "dataset_GT")
        self.assertIsNotNone(sample.overlay_image)  # overlay pre-rendered

    def test_dataset_oracle_normal_has_no_overlay(self):
        rows = [
            {
                "image_id": "normal",
                "image": np.zeros((4, 4, 3), dtype=np.uint8),
                "ground_truth_mask": None,
                "target_label": "normal",
                "transcription": "reference",
            }
        ]
        sample = build_dataset_oracle_inputs(rows)[0]
        self.assertIsNone(sample.overlay_image)
        self.assertEqual(sample.mask_source, "none")

    def test_real_json_provider_uses_explicit_contract(self):
        record = {
            "image_id": "case",
            "image": "image.jpg",
            "overlay_image": "overlay.jpg",
            "prediction": "glaucoma",
            "reference": "reference",
            "mask_source": "pipeline",
            "source_data": "pipeline",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.json"
            path.write_text(json.dumps([record]), encoding="utf-8")
            sample = load_json_inputs(path)[0]
        self.assertEqual(sample, ConditioningInput(**record))

    def test_real_json_allows_overlay_for_normal_prediction(self):
        record = {
            "image_id": "normal",
            "image": "image.jpg",
            "overlay_image": "overlay.jpg",
            "prediction": "normal",
            "reference": "reference",
            "mask_source": "pipeline",
            "source_data": "pipeline",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.json"
            path.write_text(json.dumps([record]), encoding="utf-8")
            sample = load_json_inputs(path)[0]
        self.assertEqual(sample.overlay_image, "overlay.jpg")


class PipelineLoaderTests(unittest.TestCase):
    def _make_dir(self, directory: Path) -> Path:
        image_dir = directory / "test"
        overlay_dir = image_dir / "overlays_rojos" / "Imagenes_overlay"
        overlay_dir.mkdir(parents=True)
        (image_dir / "0001.jpg").write_bytes(b"")
        (image_dir / "0001_obj_0.png").write_bytes(b"")
        (overlay_dir / "0001_overlay.jpg").write_bytes(b"")
        annotations = [
            {
                "image_filename": "0001.jpg",
                "label": "Pathological",
                "transcription": "expert reference",
                "locs_data": {"conditions": ["glaucoma"]},
            }
        ]
        (directory / "annotations.json").write_text(
            json.dumps(annotations), encoding="utf-8"
        )
        return image_dir

    def test_loads_overlay_mask_and_gt_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = self._make_dir(root)
            inputs = load_pipeline_dir_inputs(
                image_dir,
                annotations_file=root / "annotations.json",
            )
        self.assertEqual(len(inputs), 1)
        sample = inputs[0]
        self.assertEqual(sample.image_id, "0001")
        self.assertEqual(sample.prediction, "glaucoma")
        self.assertEqual(sample.expected_finding, "glaucoma")
        self.assertEqual(sample.reference, "expert reference")
        # The pipeline overlay is the one delivered by the segmentation module.
        self.assertTrue(sample.overlay_image.endswith("0001_overlay.jpg"))
        self.assertTrue(sample.mask.endswith("0001_obj_0.png"))
        self.assertEqual(sample.source_data, "pipeline")
        self.assertEqual(sample.mask_source, "pipeline")

    def test_predictions_override_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = self._make_dir(root)
            inputs = load_pipeline_dir_inputs(
                image_dir,
                annotations_file=root / "annotations.json",
                predictions={"0001": "normal"},
            )
        sample = inputs[0]
        self.assertEqual(sample.prediction, "normal")  # classifier stand-in
        self.assertEqual(sample.expected_finding, "glaucoma")  # ground truth

    def test_ground_truth_labels_override_annotation(self):
        # For traceability the GT label used to judge the classifier can come
        # from the pipeline artifact (y_true) instead of annotations.json.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = self._make_dir(root)
            inputs = load_pipeline_dir_inputs(
                image_dir,
                annotations_file=root / "annotations.json",
                predictions={"0001": "glaucoma"},
                ground_truth_labels={"0001": "normal"},
            )
        sample = inputs[0]
        self.assertEqual(sample.prediction, "glaucoma")  # classifier y_pred
        self.assertEqual(sample.expected_finding, "normal")  # pipeline y_true


class LexicalMetricTests(unittest.TestCase):
    def test_rouge_l_bounds(self):
        self.assertEqual(rouge_l("optic disc cupping", "optic disc cupping"), 1.0)
        self.assertEqual(rouge_l("abc def", "xyz uvw"), 0.0)
        partial = rouge_l("the optic disc is cupped", "optic disc cupping present")
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 1.0)

    def test_sentence_bleu_bounds(self):
        self.assertAlmostEqual(
            sentence_bleu("advanced disc cupping here now", "advanced disc cupping here now"),
            1.0,
            places=5,
        )
        self.assertEqual(sentence_bleu("", "anything"), 0.0)
        self.assertGreaterEqual(sentence_bleu("a b c", "x y z"), 0.0)


class DiagnosticStatsTests(unittest.TestCase):
    def test_clopper_pearson_and_kappa(self):
        from scripts.analyze_results import (
            _clopper_pearson,
            _cohens_kappa,
            _confusion,
        )

        lower, upper = _clopper_pearson(13, 14)
        self.assertLess(lower, 13 / 14)
        self.assertLessEqual(upper, 1.0)
        empty = _clopper_pearson(0, 0)
        self.assertEqual(len(empty), 2)
        self.assertTrue(all(np.isnan(x) for x in empty))

        pairs = [("glaucoma", "glaucoma")] * 3 + [("normal", "normal")] * 3
        self.assertAlmostEqual(_cohens_kappa(pairs), 1.0)
        confusion = _confusion([("glaucoma", "normal"), ("unknown", "glaucoma")])
        self.assertEqual(confusion["normal"]["glaucoma"], 1)
        self.assertEqual(confusion["glaucoma"]["unknown"], 1)


class OverlayTests(unittest.TestCase):
    def test_make_overlay_keeps_image_size(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        overlay = make_overlay(image, mask)
        self.assertEqual(overlay.size, (4, 4))


class ComparisonTests(unittest.TestCase):
    def test_summary_and_paired_comparison(self):
        results = []
        for image_id, score_base, score_other in (
            ("one", 0.5, 0.6),
            ("two", 0.4, 0.5),
            ("three", 0.3, 0.4),
        ):
            results.extend(
                [
                    {
                        "image_id": image_id,
                        "condition": BASELINE_CONDITION,
                        "text_metrics": {
                            "bertscore_f1": score_base,
                            "bertscore_calibrated": score_base,
                            "sbert_similarity": score_base,
                            "sbert_calibrated": score_base,
                            "finding_mentioned": True,
                        },
                    },
                    {
                        "image_id": image_id,
                        "condition": "with_overlay",
                        "text_metrics": {
                            "bertscore_f1": score_other,
                            "bertscore_calibrated": score_other,
                            "sbert_similarity": score_other,
                            "sbert_calibrated": score_other,
                            "finding_mentioned": True,
                        },
                    },
                ]
            )
        summary = summarize_by_condition(results)
        comparison = paired_comparisons_to_baseline(results, Evaluator())
        self.assertAlmostEqual(
            summary["with_overlay"]["delta_bertscore_f1_vs_baseline"],
            0.1,
        )
        self.assertAlmostEqual(
            comparison["with_overlay"]["bertscore_f1"]["mean_delta"],
            0.1,
        )


class EvaluatorTests(unittest.TestCase):
    def test_segmentation_metrics_identical_and_disjoint(self):
        evaluator = Evaluator()
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:4, 1:4] = True
        identical = evaluator.evaluate_segmentation(mask, mask)
        self.assertEqual(identical["iou"], 1.0)
        self.assertEqual(identical["dice"], 1.0)
        self.assertAlmostEqual(identical["ssim"], 1.0)

        other = np.zeros((8, 8), dtype=bool)
        other[5:8, 5:8] = True
        disjoint = evaluator.evaluate_segmentation(mask, other)
        self.assertEqual(disjoint["iou"], 0.0)
        self.assertEqual(disjoint["dice"], 0.0)

    def test_finding_mentioned(self):
        self.assertTrue(
            finding_mentioned(
                "Advanced optic disc cupping with neuroretinal rim loss.",
                "glaucoma",
            )
        )
        self.assertTrue(
            finding_mentioned(
                "Normal macula with no vascular abnormalities.",
                "normal",
            )
        )
        self.assertFalse(
            finding_mentioned(
                "Normal macula with no vascular abnormalities.",
                "glaucoma",
            )
        )

    def test_statistical_test_reports_effect_size(self):
        result = Evaluator().statistical_test(
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        )
        self.assertGreater(result["effect_size"], 0)
        self.assertIn("significant", result)


class FakeEvaluator(Evaluator):
    def evaluate_text_batch(
        self,
        generated_texts,
        reference_texts,
        expected_findings=None,
        group_ids=None,
    ):
        return [{
            "bertscore_precision": 0.5,
            "bertscore_recall": 0.5,
            "bertscore_f1": 0.5,
            "bertscore_random_baseline": 0.4,
            "bertscore_calibrated": 1 / 6,
            "sbert_similarity": 0.5,
            "sbert_random_baseline": 0.4,
            "sbert_calibrated": 1 / 6,
            "finding_mentioned": True,
            "likert_score": None,
        } for _ in generated_texts]


class FakeConditioner:
    def generate(self, condition, image_raw):
        spec = CONDITION_SPECS[condition]
        return {
            "text": f"text {condition}",
            "condition": condition,
            "prompt_used": spec.prompt,
        }


class LabelDrivenExperimentTests(unittest.TestCase):
    def test_conditions_selected_by_label_and_overlay_availability(self):
        glaucoma = ConditioningInput(
            image_id="glaucoma",
            image="image.jpg",
            overlay_image="overlay.jpg",
            prediction="glaucoma",
            reference="reference",
            mask_source="pipeline",
            source_data="pipeline",
            expected_finding="glaucoma",
        )
        normal_overlay = ConditioningInput(
            image_id="normal_ov",
            image="image.jpg",
            overlay_image="overlay.jpg",
            prediction="normal",
            reference="reference",
            mask_source="pipeline",
            source_data="pipeline",
            expected_finding="normal",
        )
        normal_plain = ConditioningInput(
            image_id="normal_no",
            image="image.jpg",
            overlay_image=None,
            prediction="normal",
            reference="reference",
            mask_source="none",
            source_data="dataset",
            expected_finding="normal",
        )

        report = run_conditioned_experiment(
            FakeConditioner(),
            FakeEvaluator(),
            [glaucoma, normal_overlay, normal_plain],
            model_variant="base",
        )

        counts = {
            condition: metrics["count"]
            for condition, metrics in report["summary_by_condition"].items()
        }
        # Glaucoma (with overlay): 4 conditions.
        self.assertEqual(counts["with_overlay_glaucoma"], 1)
        self.assertEqual(counts["without_overlay_glaucoma"], 1)
        # Normal WITH overlay now runs the overlay+normal condition too.
        self.assertEqual(counts["with_overlay_normal"], 1)
        self.assertEqual(counts["without_overlay_normal"], 2)
        # Generic overlay runs for the two samples that have an overlay.
        self.assertEqual(counts["with_overlay"], 2)
        # Baseline runs for every sample.
        self.assertEqual(counts["without_overlay"], 3)

        self.assertEqual(report["baseline_image_count"], 3)
        # expected_finding is now nested under classification.
        self.assertEqual(
            report["results"][0]["classification"],
            {"predicted": "glaucoma", "ground_truth": "glaucoma"},
        )


class BoilerplateDilutionTests(unittest.TestCase):
    def test_diagnostic_span_keeps_findings_drops_scaffold(self):
        text = ("color fundus photography of a right eye, cup-to-disc ratio of "
                "0.9, normal macula, diffuse thinning of the neuroretinal rim")
        span = diagnostic_span(text)
        # The graded findings survive; the bare anatomical scaffold is dropped.
        self.assertIn("cup-to-disc ratio of 0.9", span)
        self.assertIn("thinning of the neuroretinal rim", span)
        self.assertNotIn("fundus photography", span)
        self.assertNotIn("normal macula", span)

    def test_diagnostic_span_does_not_split_decimals(self):
        # The period in "0.6" must not break the clause.
        self.assertIn("0.6", diagnostic_span("cup-to-disc ratio of 0.6"))

    def test_reference_cdr_ratio_and_grade(self):
        self.assertAlmostEqual(reference_cdr("cup-to-disc ratio of 0.8"), 0.8)
        # A bare integer is read as an ordinal grade (2 -> 0.65).
        self.assertAlmostEqual(reference_cdr("cup to disc ratio 2"), 0.65)
        self.assertIsNone(reference_cdr("optic disc with preserved rim"))

    def test_cdr_class_separation(self):
        refs = ["cup-to-disc ratio of 0.9", "cup-to-disc ratio of 0.8",
                "cup-to-disc ratio of 0.3", "cup-to-disc ratio of 0.4"]
        labels = ["glaucoma", "glaucoma", "normal", "normal"]
        sep = cdr_class_separation(refs, labels)
        self.assertEqual(sep["glaucoma"]["n"], 2)
        self.assertEqual(sep["normal"]["n"], 2)
        self.assertGreater(sep["glaucoma"]["mean"], sep["normal"]["mean"])
        self.assertEqual(sep["roc_auc"], 1.0)  # perfectly separable


if __name__ == "__main__":
    unittest.main()
