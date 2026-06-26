from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from medgemma_utils.experimental_lora_training import (
    _collect_saved_artifacts,
    write_training_metadata,
)


class TrainingArtifactTests(unittest.TestCase):
    def test_collect_saved_artifacts_requires_adapter_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Missing saved LoRA adapter"):
                _collect_saved_artifacts(output_dir)

    def test_collect_saved_artifacts_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output_dir / "adapter_model.safetensors").write_bytes(b"weights")

            artifacts = _collect_saved_artifacts(output_dir)
            self.assertEqual(artifacts["adapter_model.safetensors"], 7)

            write_training_metadata(output_dir, {"full_train_split": True})
            metadata = json.loads(
                (output_dir / "training_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(metadata["full_train_split"])


if __name__ == "__main__":
    unittest.main()
