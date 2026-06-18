"""M7 implementation compatible with the interface defined in ref."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .conditioning import MASK_CONDITIONS, PROMPT_TEMPLATES, format_distribution, make_overlay


class MedGemmaConditioner:
    """Generate descriptions for the six MedGemma ablation conditions."""

    def __init__(self, config: dict[str, Any]):
        self.config = config.copy()
        self.model_name = config.get("model_name", "google/medgemma-1.5-4b-it")
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.seed = int(config.get("seed", 42))
        self.token = config.get("token")
        self.device_map = config.get("device_map")
        self._set_seed()

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            token=self.token,
        )
        dtype = self._resolve_dtype(config.get("torch_dtype", "auto"))
        model_kwargs = {
            "dtype": dtype,
            "token": self.token,
        }
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        adapter_path = config.get("adapter_path")
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        if self.device_map is None:
            if not torch.cuda.is_available():
                raise RuntimeError("MedGemma requires a CUDA GPU")
            self.model = self.model.to("cuda")
        self.model.eval()

    @classmethod
    def from_components(
        cls,
        model: Any,
        processor: Any,
        *,
        max_new_tokens: int = 512,
        seed: int = 42,
    ) -> "MedGemmaConditioner":
        instance = cls.__new__(cls)
        instance.config = {}
        instance.model_name = getattr(model.config, "_name_or_path", "medgemma")
        instance.max_new_tokens = max_new_tokens
        instance.seed = seed
        instance.token = None
        instance.device_map = getattr(model, "hf_device_map", None)
        instance.processor = processor
        instance.model = model
        instance._set_seed()
        instance.model.eval()
        return instance

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _resolve_dtype(value: str | torch.dtype) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        if value == "float16":
            return torch.float16
        if value == "bfloat16":
            return torch.bfloat16
        if value != "auto":
            raise ValueError(f"Unsupported torch_dtype: {value}")
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16

    @staticmethod
    def _as_image(image_raw: Any) -> Image.Image:
        if isinstance(image_raw, Image.Image):
            return image_raw.convert("RGB")
        if isinstance(image_raw, (str, bytes, Path)):
            return Image.open(image_raw).convert("RGB")
        return Image.fromarray(np.asarray(image_raw).astype(np.uint8)).convert("RGB")

    @staticmethod
    def _validate_inputs(
        condition: str,
        mask: np.ndarray | None,
        prediction: str | None,
        distribution: dict[str, float] | None,
    ) -> None:
        required = {
            "A": (False, False, False),
            "B": (True, False, False),
            "C1": (False, True, False),
            "C2": (False, False, True),
            "D1": (True, True, False),
            "D2": (True, False, True),
        }
        if condition not in required:
            raise ValueError(f"Unknown condition: {condition}")
        needs_mask, needs_prediction, needs_distribution = required[condition]
        if needs_mask and mask is None:
            raise ValueError(f"Condition {condition} requires mask")
        if needs_prediction and prediction is None:
            raise ValueError(f"Condition {condition} requires prediction")
        if needs_distribution and distribution is None:
            raise ValueError(f"Condition {condition} requires distribution")

    def _model_device(self) -> torch.device | str:
        if hasattr(self.model, "hf_device_map"):
            for device in self.model.hf_device_map.values():
                if device not in {"cpu", "disk", "meta"}:
                    return device
        return next(self.model.parameters()).device

    def generate(
        self,
        condition: str,
        image_raw: np.ndarray,
        mask: np.ndarray | None = None,
        prediction: str | None = None,
        distribution: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        condition = condition.upper()
        self._validate_inputs(condition, mask, prediction, distribution)

        image = self._as_image(image_raw)
        image_was_overlaid = condition in MASK_CONDITIONS
        if image_was_overlaid:
            image = make_overlay(image, mask)

        prompt = PROMPT_TEMPLATES[condition].format(
            prediction=prediction,
            distribution=(
                format_distribution(distribution)
                if distribution is not None
                else None
            ),
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = self._model_device()
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        text = self.processor.decode(
            output[0, input_length:],
            skip_special_tokens=True,
        ).strip()
        return {
            "text": text,
            "condition": condition,
            "prompt_used": prompt,
            "image_was_overlaid": image_was_overlaid,
        }
