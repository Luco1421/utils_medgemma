import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from medgemma_utils.conditioning import (
    build_condition_request,
)
from medgemma_utils.evaluation import (
    paired_comparisons_to_baseline,
    summarize_by_condition,
)
from medgemma_utils.inputs import ConditioningInput, load_json_inputs
from medgemma_utils.mock_inputs import build_dataset_mock_inputs


class InputProviderTests(unittest.TestCase):
    def test_dataset_mock_uses_label_and_80_20_distribution(self):
        rows = [
            {
                "image_id": "case",
                "image": "image.jpg",
                "mask_disc_npy": "mask.npy",
                "target_label": "glaucoma",
                "transcription": "reference",
            }
        ]
        sample = build_dataset_mock_inputs(rows)[0]
        self.assertEqual(sample.prediction, "glaucoma")
        self.assertAlmostEqual(sample.distribution["glaucoma"], 0.8)
        self.assertAlmostEqual(sample.distribution["non_glaucoma"], 0.2)
        self.assertEqual(sample.input_source, "dataset_oracle_mock")

    def test_real_json_provider_uses_explicit_contract(self):
        record = {
            "image_id": "case",
            "image": "image.jpg",
            "mask": "mask.npy",
            "prediction": "glaucoma",
            "distribution": {"glaucoma": 0.7, "non_glaucoma": 0.3},
            "reference": "reference",
            "mask_source": "wsss",
            "input_source": "real_pipeline",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.json"
            path.write_text(json.dumps([record]), encoding="utf-8")
            sample = load_json_inputs(path)[0]
        self.assertEqual(sample, ConditioningInput(**record))


class ConditionRequestTests(unittest.TestCase):
    def setUp(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        self.sample = ConditioningInput(
            image_id="case",
            image=image,
            mask=mask,
            prediction="glaucoma",
            distribution={"glaucoma": 0.8, "non_glaucoma": 0.2},
            reference="reference",
            mask_source="ground_truth_disc",
            input_source="dataset_oracle_mock",
        )

    def test_condition_a_uses_raw_image(self):
        request = build_condition_request("A", self.sample)
        self.assertFalse(request["image_was_overlaid"])

    def test_condition_d2_uses_overlay_and_distribution(self):
        request = build_condition_request("D2", self.sample)
        self.assertTrue(request["image_was_overlaid"])
        self.assertIn("glaucoma (80%)", request["prompt"])


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
                    {"image_id": image_id, "condition": "A", "bertscore_f1": score_a},
                    {"image_id": image_id, "condition": "B", "bertscore_f1": score_b},
                ]
            )
        summary = summarize_by_condition(results)
        comparison = paired_comparisons_to_baseline(results)
        self.assertAlmostEqual(summary["B"]["delta_f1_vs_A"], 0.1)
        self.assertAlmostEqual(comparison["B"]["mean_delta_f1"], 0.1)


if __name__ == "__main__":
    unittest.main()
