import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from medgemma_utils.config import load_project_config
from medgemma_utils.dataset import (
    build_sft_examples,
    load_split_rows,
    validate_optic_disc_mask_pixels,
)
from medgemma_utils.oracle_inputs import build_dataset_oracle_inputs


class DatasetAdapterTests(unittest.TestCase):
    def test_project_config_declares_optic_disc_and_normal_bypass(self):
        config = load_project_config()
        self.assertEqual(config["data"]["mask_target"], "optic_disc")
        self.assertTrue(config["data"]["normal_bypasses_segmentation"])

    def _write_dataset(self, root: Path) -> Path:
        records = {
            "train": [
                {
                    "id": 1,
                    "image_filename": "pathological.jpg",
                    "label": "Pathological",
                    "transcription": "Glaucomatous optic nerve.",
                    "locs_data": {"conditions": ["glaucoma"]},
                },
                {
                    "id": 2,
                    "image_filename": "normal.jpg",
                    "label": "Normal",
                    "transcription": "Normal optic nerve.",
                    "locs_data": {},
                },
            ],
            "validation": [],
            "test": [],
        }
        for filename in ("pathological.jpg", "normal.jpg"):
            Image.new("RGB", (4, 4)).save(root / filename)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        Image.fromarray(mask).save(root / "pathological_disc.png")
        annotations = [*records["train"]]
        (root / "annotations.json").write_text(
            json.dumps(annotations),
            encoding="utf-8",
        )
        split = root / "split.json"
        split.write_text(json.dumps(records), encoding="utf-8")
        return split

    def test_loads_embedded_annotations_and_optional_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = load_split_rows(self._write_dataset(Path(directory)))["train"]

        pathological, normal = rows
        self.assertEqual(pathological["target_label"], "glaucoma")
        self.assertTrue(pathological["has_ground_truth_mask"])
        self.assertEqual(pathological["mask_target"], "optic_disc")
        self.assertEqual(normal["target_label"], "normal")
        self.assertFalse(normal["has_ground_truth_mask"])
        self.assertIsNone(normal["ground_truth_mask"])

    def test_sft_uses_all_rows_without_requiring_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = load_split_rows(self._write_dataset(Path(directory)))["train"]
            examples = build_sft_examples(rows)

        self.assertEqual(len(examples), 2)

    def test_validates_only_the_contractual_disc_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = load_split_rows(self._write_dataset(Path(directory)))
            count = validate_optic_disc_mask_pixels(rows)

        self.assertEqual(count, 1)

    def test_oracle_inputs_mark_normal_as_bypassed(self):
        row = {
            "image_id": "normal",
            "image": "normal.jpg",
            "ground_truth_mask": None,
            "target_label": "normal",
            "transcription": "Normal optic nerve.",
            "mask_target": "optic_disc",
        }
        sample = build_dataset_oracle_inputs([row])[0]
        self.assertIsNone(sample.mask)
        self.assertEqual(sample.mask_source, "not_applicable_normal")
        self.assertEqual(sample.segmentation_status, "bypassed_normal")
        self.assertIsNone(sample.ground_truth_mask)

    def test_oracle_inputs_can_require_a_mask_when_requested(self):
        row = {
            "image_id": "normal",
            "image": "normal.jpg",
            "ground_truth_mask": None,
            "target_label": "normal",
            "transcription": "Normal optic nerve.",
            "mask_target": "optic_disc",
        }
        with self.assertRaisesRegex(
            ValueError,
            "has no optic-disc ground-truth mask",
        ):
            build_dataset_oracle_inputs([row], require_mask=True)


if __name__ == "__main__":
    unittest.main()
