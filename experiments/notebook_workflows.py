"""Funciones pensadas para usarse directamente desde notebooks Colab."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from modules.medgemma_conditioner import CONDITION_SPECS, MedGemmaConditioner
from modules.medgemma_runtime import MedGemmaRuntime
from experiments.build_acrima_jsonl import build_acrima_jsonl
from experiments.train_lora_medgemma import train_lora_medgemma


class RuntimeConditioner(MedGemmaConditioner):
    """Conditioner que usa MedGemmaRuntime para generar texto."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.runtime = MedGemmaRuntime(config)

    def _generate_text(self, image_rgb: np.ndarray, prompt: str) -> str:
        return self.runtime.generate(prompt=prompt, image=image_rgb)


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Carga una imagen RGB como ndarray uint8."""
    return np.asarray(Image.open(path).convert("RGB"))


def make_dummy_mask(image_shape: tuple[int, int]) -> np.ndarray:
    """Crea una mascara circular central para smoke tests tecnicos."""
    height, width = image_shape
    yy, xx = np.ogrid[:height, :width]
    radius = min(height, width) // 6
    center_y, center_x = height // 2, width // 2
    return ((yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2).astype(bool)


def run_smoke_inference(
    model_id: str,
    prompt: str,
    image: str | Path | np.ndarray | None = None,
    hf_token: str | None = None,
    adapter_path: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Ejecuta una inferencia simple de MedGemma en el kernel actual."""
    runtime = MedGemmaRuntime(
        {
            "model_id": model_id,
            "hf_token": hf_token,
            "adapter_path": adapter_path,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "max_new_tokens": 512,
            "do_sample": False,
        }
    )
    text = runtime.generate(prompt=prompt, image=image)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "adapter_path": adapter_path,
        "image": str(image) if image is not None and not isinstance(image, np.ndarray) else None,
        "prompt": prompt,
        "text": text,
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


__all__ = [
    "build_acrima_jsonl",
    "load_rgb_image",
    "make_dummy_mask",
    "run_ablation",
    "run_smoke_inference",
    "train_lora_medgemma",
]


def run_ablation(
    model_id: str,
    image_path: str | Path,
    prediction: str = "glaucoma",
    distribution: dict[str, float] | None = None,
    mask: np.ndarray | None = None,
    hf_token: str | None = None,
    adapter_path: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Ejecuta condiciones A/B/C1/C2/D1/D2 desde el kernel actual."""
    image = load_rgb_image(image_path)
    mask_array = mask if mask is not None else make_dummy_mask(image.shape[:2])
    distribution = distribution or {"glaucoma": 0.92, "normal": 0.08}

    conditioner = RuntimeConditioner(
        {
            "model_id": model_id,
            "hf_token": hf_token,
            "adapter_path": adapter_path,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "max_new_tokens": 512,
            "do_sample": False,
            "seed": 42,
        }
    )

    results = []
    for condition, spec in CONDITION_SPECS.items():
        results.append(
            conditioner.generate(
                condition=condition,
                image_raw=image,
                mask=mask_array if spec.uses_mask else None,
                prediction=prediction if spec.uses_prediction else None,
                distribution=distribution if spec.uses_distribution else None,
            )
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "adapter_path": adapter_path,
        "image": str(image_path),
        "mask": "dummy" if mask is None else "provided",
        "prediction": prediction,
        "distribution": distribution,
        "results": results,
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
