from __future__ import annotations

import unittest

import numpy as np

from modules.medgemma_runtime import MedGemmaRuntime


class MedGemmaRuntimeTest(unittest.TestCase):
    def test_build_text_only_message(self) -> None:
        messages = MedGemmaRuntime.build_messages("Explain glaucoma.")

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], [{"type": "text", "text": "Explain glaucoma."}])

    def test_build_image_text_message(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        messages = MedGemmaRuntime.build_messages("Describe.", image=image)

        self.assertEqual(messages[0]["content"][0]["type"], "image")
        self.assertEqual(messages[0]["content"][1], {"type": "text", "text": "Describe."})

    def test_rejects_non_rgb_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "RGB"):
            MedGemmaRuntime.to_pil_image(np.zeros((2, 2), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
