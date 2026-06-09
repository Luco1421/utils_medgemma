"""Ejecuta las 6 condiciones de MedGemmaConditioner sobre una imagen.

Este script es para probar el baseline/pretrained. Si se pasa --adapter-path,
usa el mismo flujo pero con un adapter LoRA cargado.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.medgemma_conditioner import CONDITION_SPECS, MedGemmaConditioner
from modules.medgemma_runtime import MedGemmaRuntime


class RuntimeConditioner(MedGemmaConditioner):
    """Conditioner que delega la generacion en MedGemmaRuntime."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.runtime = MedGemmaRuntime(config)

    def _generate_text(self, image_rgb: np.ndarray, prompt: str) -> str:
        return self.runtime.generate(prompt=prompt, image=image_rgb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B/C1/C2/D1/D2 MedGemma ablations.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", default=None, help="Path to .npy mask or grayscale image mask.")
    parser.add_argument("--make-dummy-mask", action="store_true")
    parser.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--prediction", default="glaucoma")
    parser.add_argument("--distribution", default='{"glaucoma": 0.92, "normal": 0.08}')
    parser.add_argument("--output", default="results/ablation_medgemma.json")
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: str | None, image_shape: tuple[int, int], make_dummy: bool) -> np.ndarray:
    if path:
        mask_path = Path(path)
        if mask_path.suffix.lower() == ".npy":
            mask = np.load(mask_path)
        else:
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != image_shape:
            raise ValueError(f"mask shape {mask.shape} does not match image shape {image_shape}.")
        return mask.astype(bool)

    if not make_dummy:
        raise ValueError("Provide --mask or use --make-dummy-mask for a visual smoke test.")

    height, width = image_shape
    yy, xx = np.ogrid[:height, :width]
    radius = min(height, width) // 6
    center_y, center_x = height // 2, width // 2
    return ((yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2).astype(bool)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    image = load_image(args.image)
    mask = load_mask(args.mask, image.shape[:2], args.make_dummy_mask)
    distribution = json.loads(args.distribution)

    conditioner = RuntimeConditioner(
        {
            "model_id": args.model_id,
            "adapter_path": args.adapter_path,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "max_new_tokens": 512,
            "do_sample": False,
            "seed": 42,
        }
    )

    results = []
    for condition, spec in CONDITION_SPECS.items():
        result = conditioner.generate(
            condition=condition,
            image_raw=image,
            mask=mask if spec.uses_mask else None,
            prediction=args.prediction if spec.uses_prediction else None,
            distribution=distribution if spec.uses_distribution else None,
        )
        results.append(result)
        print(f"\n[{condition}]\n{result['text']}\n")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "image": args.image,
        "mask": args.mask,
        "prediction": args.prediction,
        "distribution": distribution,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
