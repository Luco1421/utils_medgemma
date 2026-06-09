"""Condicionamiento de MedGemma para descripciones clinicas oftalmologicas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .medgemma_runtime import MedGemmaRuntime
from .utils.seed import set_global_seed

@dataclass(frozen=True)
class ConditionSpec:
    """Define los insumos requeridos por una condicion de ablation."""

    uses_mask: bool
    uses_prediction: bool
    uses_distribution: bool
    prompt_template: str


CONDITION_SPECS: dict[str, ConditionSpec] = {
    "A": ConditionSpec(
        uses_mask=False,
        uses_prediction=False,
        uses_distribution=False,
        prompt_template="Describe the ophthalmological findings in this fundus image.",
    ),
    "B": ConditionSpec(
        uses_mask=True,
        uses_prediction=False,
        uses_distribution=False,
        prompt_template=(
            "The region highlighted in red was identified by an automatic segmentation "
            "system. Describe the ophthalmological findings, focusing on the highlighted "
            "region."
        ),
    ),
    "C1": ConditionSpec(
        uses_mask=False,
        uses_prediction=True,
        uses_distribution=False,
        prompt_template=(
            "An ophthalmological classifier identifies the primary finding in this fundus "
            "image as: {prediction}. Describe the ophthalmological findings."
        ),
    ),
    "C2": ConditionSpec(
        uses_mask=False,
        uses_prediction=False,
        uses_distribution=True,
        prompt_template=(
            "An ophthalmological classifier analyzed this fundus image and estimates: "
            "{distribution}. Describe the ophthalmological findings."
        ),
    ),
    "D1": ConditionSpec(
        uses_mask=True,
        uses_prediction=True,
        uses_distribution=False,
        prompt_template=(
            "An ophthalmological classifier identifies the primary finding as: "
            "{prediction}. The region highlighted in red indicates the area where this "
            "finding is located. Describe the findings focusing on the highlighted region."
        ),
    ),
    "D2": ConditionSpec(
        uses_mask=True,
        uses_prediction=False,
        uses_distribution=True,
        prompt_template=(
            "An ophthalmological classifier estimates: {distribution}. The region "
            "highlighted in red indicates the area where the main finding is located. "
            "Describe the findings in detail, focusing on the highlighted region and its "
            "relationship with the suggested diagnosis."
        ),
    ),
}


class MedGemmaConditioner:
    """Genera descripciones clinicas con MedGemma bajo 6 condiciones de ablation.

    MedGemma se usa como caja negra. Este modulo solo modifica la imagen de
    entrada y el prompt para medir el efecto del condicionamiento externo.
    """

    def __init__(self, config: dict[str, Any]):
        """Inicializa el condicionador con configuracion centralizada."""
        self.config = dict(config)
        self.seed = int(self.config.get("seed", 42))
        set_global_seed(self.seed)

        self.model_name = str(
            self.config.get(
                "model_id",
                self.config.get("model_name", "google/medgemma-1.5-4b-it"),
            )
        )
        self.torch_dtype = str(self.config.get("torch_dtype", "bfloat16"))
        self.device_map = self.config.get("device_map", "auto")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 512))
        self.do_sample = bool(self.config.get("do_sample", False))
        self.overlay_alpha = float(self.config.get("overlay_alpha", 0.4))

        self._runtime: MedGemmaRuntime | None = None

    def generate(
        self,
        condition: str,
        image_raw: np.ndarray,
        mask: np.ndarray | None = None,
        prediction: str | None = None,
        distribution: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Genera un texto clinico para una condicion de ablation.

        Args:
            condition: Una de `A`, `B`, `C1`, `C2`, `D1`, `D2`.
            image_raw: Imagen RGB sin normalizar con forma `(H, W, 3)`.
            mask: Mascara binaria `(H, W)` para `B`, `D1` y `D2`.
            prediction: Clase predicha para `C1` y `D1`.
            distribution: Distribucion de probabilidades para `C2` y `D2`.

        Returns:
            Diccionario con texto generado, condicion, prompt y bandera de overlay.
        """
        self._validate_inputs(condition, image_raw, mask, prediction, distribution)
        spec = CONDITION_SPECS[condition]
        prompt = self.build_prompt(condition, prediction=prediction, distribution=distribution)
        prepared_image = (
            self.apply_mask_overlay(image_raw, mask, alpha=self.overlay_alpha)
            if spec.uses_mask
            else self.ensure_rgb_uint8(image_raw)
        )

        text = self._generate_text(prepared_image, prompt)
        return {
            "text": text,
            "condition": condition,
            "prompt_used": prompt,
            "image_was_overlaid": spec.uses_mask,
        }

    @classmethod
    def build_prompt(
        cls,
        condition: str,
        prediction: str | None = None,
        distribution: dict[str, float] | None = None,
    ) -> str:
        """Construye el prompt exacto para una condicion."""
        if condition not in CONDITION_SPECS:
            allowed = ", ".join(CONDITION_SPECS)
            raise ValueError(f"Unknown condition '{condition}'. Expected one of: {allowed}.")

        spec = CONDITION_SPECS[condition]
        return spec.prompt_template.format(
            prediction=prediction,
            distribution=cls.format_distribution(distribution) if distribution else None,
        )

    @staticmethod
    def format_distribution(distribution: dict[str, float]) -> str:
        """Formatea una distribucion ordenada de mayor a menor probabilidad."""
        if not distribution:
            raise ValueError("distribution must not be empty.")

        parts = []
        for label, probability in sorted(distribution.items(), key=lambda item: item[1], reverse=True):
            probability_float = float(probability)
            if probability_float < 0 or probability_float > 1:
                raise ValueError("distribution probabilities must be in [0, 1].")
            parts.append(f"{label} ({probability_float:.0%})")
        return ", ".join(parts)

    @staticmethod
    def ensure_rgb_uint8(image_raw: np.ndarray) -> np.ndarray:
        """Valida y devuelve una imagen RGB uint8."""
        image = np.asarray(image_raw)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_raw must have shape (H, W, 3).")
        if image.dtype == np.uint8:
            return image.copy()
        return np.clip(image, 0, 255).astype(np.uint8)

    @classmethod
    def apply_mask_overlay(
        cls,
        image_raw: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Aplica overlay rojo semitransparente sobre la mascara."""
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0, 1].")

        image = cls.ensure_rgb_uint8(image_raw).astype(np.float32)
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.shape != image.shape[:2]:
            raise ValueError("mask must have shape (H, W) matching image_raw.")

        red = np.array([255, 0, 0], dtype=np.float32)
        image[mask_bool] = image[mask_bool] * (1.0 - alpha) + red * alpha
        return np.rint(image).clip(0, 255).astype(np.uint8)

    def _validate_inputs(
        self,
        condition: str,
        image_raw: np.ndarray,
        mask: np.ndarray | None,
        prediction: str | None,
        distribution: dict[str, float] | None,
    ) -> None:
        """Valida que los insumos correspondan a la condicion solicitada."""
        if condition not in CONDITION_SPECS:
            allowed = ", ".join(CONDITION_SPECS)
            raise ValueError(f"Unknown condition '{condition}'. Expected one of: {allowed}.")

        self.ensure_rgb_uint8(image_raw)
        spec = CONDITION_SPECS[condition]

        if spec.uses_mask and mask is None:
            raise ValueError(f"Condition {condition} requires mask.")
        if not spec.uses_mask and mask is not None:
            raise ValueError(f"Condition {condition} must not receive mask.")
        if spec.uses_prediction and not prediction:
            raise ValueError(f"Condition {condition} requires prediction.")
        if not spec.uses_prediction and prediction is not None:
            raise ValueError(f"Condition {condition} must not receive prediction.")
        if spec.uses_distribution and not distribution:
            raise ValueError(f"Condition {condition} requires distribution.")
        if not spec.uses_distribution and distribution is not None:
            raise ValueError(f"Condition {condition} must not receive distribution.")
        if mask is not None and np.asarray(mask).shape != np.asarray(image_raw).shape[:2]:
            raise ValueError("mask must have shape (H, W) matching image_raw.")

    def _generate_text(self, image_rgb: np.ndarray, prompt: str) -> str:
        """Ejecuta inferencia con MedGemma."""
        if self._runtime is None:
            self._runtime = MedGemmaRuntime(
                {
                    **self.config,
                    "model_id": self.model_name,
                    "max_new_tokens": self.max_new_tokens,
                    "torch_dtype": self.torch_dtype,
                    "device_map": self.device_map,
                    "do_sample": self.do_sample,
                }
            )
        return self._runtime.generate(prompt=prompt, image=image_rgb)
