import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from medgemma_utils.conditioning import build_prompt, make_overlay, validate_condition_inputs
from medgemma_utils.evaluation import (
    Evaluator,
    finding_mentioned,
    paired_comparisons_to_baseline,
    summarize_by_condition,
)
from medgemma_utils.experiment import run_conditioned_experiment
from medgemma_utils.inputs import ConditioningInput, load_json_inputs
from medgemma_utils.oracle_inputs import build_dataset_oracle_inputs


class InputProviderTests(unittest.TestCase):
    def test_dataset_oracle_uses_label_and_80_20_distribution(self):
        rows = [
            {
                "image_id": "case",
                "image": "image.jpg",
                "ground_truth_mask": "mask.png",
                "target_label": "glaucoma",
                "transcription": "reference",
                "mask_target": "optic_disc",
            }
        ]
        sample = build_dataset_oracle_inputs(rows)[0]
        self.assertEqual(sample.prediction, "glaucoma")
        self.assertAlmostEqual(sample.distribution["glaucoma"], 0.8)
        self.assertAlmostEqual(sample.distribution["normal"], 0.2)
        self.assertEqual(sample.input_source, "dataset_oracle")
        self.assertIsNone(sample.ground_truth_mask)

    def test_real_json_provider_uses_explicit_contract(self):
        record = {
            "image_id": "case",
            "image": "image.jpg",
            "mask": "mask.npy",
            "prediction": "glaucoma",
            "distribution": {"glaucoma": 0.7, "normal": 0.3},
            "reference": "reference",
            "mask_source": "wsss",
            "input_source": "real_pipeline",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.json"
            path.write_text(json.dumps([record]), encoding="utf-8")
            sample = load_json_inputs(path)[0]
        self.assertEqual(sample, ConditioningInput(**record))

    def test_real_json_rejects_mask_for_normal_prediction(self):
        record = {
            "image_id": "normal",
            "image": "image.jpg",
            "mask": "mask.npy",
            "prediction": "normal",
            "distribution": {"glaucoma": 0.1, "normal": 0.9},
            "reference": "reference",
            "mask_source": "unexpected",
            "input_source": "real_pipeline",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.json"
            path.write_text(json.dumps([record]), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "must bypass segmentation",
            ):
                load_json_inputs(path)


class ConditionRequestTests(unittest.TestCase):
    def setUp(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        self.image = image
        self.mask = mask

    def test_condition_a_uses_raw_image(self):
        condition = validate_condition_inputs("A")
        self.assertEqual(condition, "A")
        self.assertIn("ophthalmological findings", build_prompt("A"))

    def test_condition_d2_uses_overlay_and_distribution(self):
        condition = validate_condition_inputs(
            "D2",
            mask=self.mask,
            distribution={"glaucoma": 0.8, "normal": 0.2},
        )
        prompt = build_prompt(
            condition,
            distribution={"glaucoma": 0.8, "normal": 0.2},
        )
        overlay = make_overlay(self.image, self.mask)
        self.assertEqual(overlay.size, (4, 4))
        self.assertIn("glaucoma (80%)", prompt)

    def test_rejects_parameters_not_used_by_condition(self):
        with self.assertRaisesRegex(ValueError, "does not accept mask"):
            validate_condition_inputs("A", mask=self.mask)


class ComparisonTests(unittest.TestCase):
    def test_summary_and_paired_comparison(self):
        results = []
        for image_id, score_a, score_b in (
            ("one", 0.5, 0.6),
            ("two", 0.4, 0.5),
            ("three", 0.3, 0.4),
        ):
            results.extend(
                [
                    {
                        "image_id": image_id,
                        "condition": "A",
                        "text_metrics": {
                            "bertscore_f1": score_a,
                            "bertscore_calibrated": score_a,
                            "sbert_similarity": score_a,
                            "sbert_calibrated": score_a,
                            "finding_mentioned": True,
                        },
                    },
                    {
                        "image_id": image_id,
                        "condition": "B",
                        "text_metrics": {
                            "bertscore_f1": score_b,
                            "bertscore_calibrated": score_b,
                            "sbert_similarity": score_b,
                            "sbert_calibrated": score_b,
                            "finding_mentioned": True,
                        },
                    },
                ]
            )
        summary = summarize_by_condition(results)
        comparison = paired_comparisons_to_baseline(results, Evaluator())
        self.assertAlmostEqual(
            summary["B"]["delta_bertscore_f1_vs_A"],
            0.1,
        )
        self.assertAlmostEqual(
            comparison["B"]["bertscore_f1"]["mean_delta"],
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


class CoverageAwareExperimentTests(unittest.TestCase):
    def test_mask_conditions_are_skipped_and_pairing_uses_complete_cases(self):
        class FakeConditioner:
            def generate(self, condition, **kwargs):
                return {
                    "text": f"text {condition}",
                    "condition": condition,
                    "prompt_used": condition,
                    "image_was_overlaid": condition in {"B", "D1", "D2"},
                }

        masked = ConditioningInput(
            image_id="masked",
            image="image.jpg",
            mask="mask.npy",
            prediction="glaucoma",
            distribution={"glaucoma": 0.8, "normal": 0.2},
            reference="reference",
            mask_source="ground_truth_disc",
            input_source="dataset_oracle",
        )
        unmasked = ConditioningInput(
            image_id="unmasked",
            image="image.jpg",
            mask=None,
            prediction="normal",
            distribution={"normal": 0.8, "glaucoma": 0.2},
            reference="reference",
            mask_source="ground_truth_disc",
            input_source="dataset_oracle",
        )

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

        report = run_conditioned_experiment(
            FakeConditioner(),
            FakeEvaluator(),
            [masked, unmasked],
            pipeline_name="test",
            model_variant="base",
        )

        counts = {
            condition: metrics["count"]
            for condition, metrics in report[
                "coverage_summary_by_condition"
            ].items()
        }
        self.assertEqual(counts["A"], 2)
        self.assertEqual(counts["B"], 1)
        self.assertEqual(report["paired_image_ids"], ["masked"])


if __name__ == "__main__":
    unittest.main()
