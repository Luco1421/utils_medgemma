"""Convierte ACRIMA por carpetas de clase a JSONL para smoke tests de LoRA.

El dataset esperado tiene esta forma:
    dataset/
      Glaucoma/
      Non Glaucoma/

La salida es util para probar el flujo tecnico de LoRA. No reemplaza un dataset
con descripciones expertas reales.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


CLASS_CONFIG = {
    "Glaucoma": {
        "label": "glaucoma",
        "answer": (
            "This fundus image is labeled as glaucoma. The description should focus on "
            "possible glaucomatous optic neuropathy findings such as increased cup-to-disc "
            "ratio, neuroretinal rim thinning, and optic disc changes, while noting that "
            "this label comes from the dataset annotation."
        ),
    },
    "Non Glaucoma": {
        "label": "normal",
        "answer": (
            "This fundus image is labeled as non-glaucoma. The description should indicate "
            "that no glaucomatous finding is provided by the dataset label, while noting "
            "that this is based on the available class annotation."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSONL examples from ACRIMA folders.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output", default="data/train_medgemma_acrima.jsonl")
    parser.add_argument("--max-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_images(class_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix in suffixes)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    random.seed(args.seed)

    rows = []
    for class_name, class_config in CLASS_CONFIG.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        images = iter_images(class_dir)
        if args.max_per_class > 0:
            images = random.sample(images, min(args.max_per_class, len(images)))

        for image_path in sorted(images):
            rows.append(
                {
                    "image": str(image_path.relative_to(dataset_dir)).replace("\\", "/"),
                    "prompt": (
                        "Describe the ophthalmological findings in this fundus image and "
                        "state whether the dataset label suggests glaucoma."
                    ),
                    "answer": class_config["answer"],
                    "label": class_config["label"],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {output_path}")
    print("Use this for technical LoRA smoke tests, not as expert clinical supervision.")


if __name__ == "__main__":
    main()
