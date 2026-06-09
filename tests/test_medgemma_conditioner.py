from __future__ import annotations

import unittest

import numpy as np

from modules.medgemma_conditioner import MedGemmaConditioner


class FakeMedGemmaConditioner(MedGemmaConditioner):
    def _generate_text(self, image_rgb: np.ndarray, prompt: str) -> str:
        self.last_image = image_rgb
        self.last_prompt = prompt
        return "Generated clinical description."


class MedGemmaConditionerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((2, 2, 3), dtype=np.uint8)
        self.mask = np.array([[True, False], [False, True]])
        self.conditioner = FakeMedGemmaConditioner({"seed": 42, "overlay_alpha": 0.4})

    def test_condition_a_uses_raw_image_and_generic_prompt(self) -> None:
        result = self.conditioner.generate("A", self.image)

        self.assertEqual(result["condition"], "A")
        self.assertFalse(result["image_was_overlaid"])
        self.assertEqual(
            result["prompt_used"],
            "Describe the ophthalmological findings in this fundus image.",
        )
        np.testing.assert_array_equal(self.conditioner.last_image, self.image)

    def test_condition_b_applies_red_overlay(self) -> None:
        result = self.conditioner.generate("B", self.image, mask=self.mask)

        self.assertTrue(result["image_was_overlaid"])
        np.testing.assert_array_equal(self.conditioner.last_image[0, 0], np.array([102, 0, 0]))
        np.testing.assert_array_equal(self.conditioner.last_image[0, 1], np.array([0, 0, 0]))

    def test_condition_c1_requires_prediction(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires prediction"):
            self.conditioner.generate("C1", self.image)

    def test_condition_d2_formats_distribution_descending(self) -> None:
        result = self.conditioner.generate(
            "D2",
            self.image,
            mask=self.mask,
            distribution={"normal": 0.08, "glaucoma": 0.92},
        )

        self.assertIn("glaucoma (92%), normal (8%)", result["prompt_used"])

    def test_rejects_unexpected_inputs_for_condition(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not receive mask"):
            self.conditioner.generate("A", self.image, mask=self.mask)

    def test_rejects_mask_shape_mismatch(self) -> None:
        bad_mask = np.zeros((3, 3), dtype=bool)

        with self.assertRaisesRegex(ValueError, "mask must have shape"):
            self.conditioner.generate("B", self.image, mask=bad_mask)

    def test_rejects_invalid_distribution_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "probabilities must be"):
            self.conditioner.generate(
                "C2",
                self.image,
                distribution={"glaucoma": 1.2, "normal": -0.2},
            )


if __name__ == "__main__":
    unittest.main()
