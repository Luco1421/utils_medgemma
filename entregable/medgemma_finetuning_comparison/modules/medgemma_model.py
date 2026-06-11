"""Carga e inferencia de MedGemma base o MedGemma con adapter LoRA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from modules.utils import device_index, get_hf_token, torch_dtype_from_config

logger = logging.getLogger(__name__)


class MedGemmaGenerator:
    """Generador deterministico para MedGemma base o MedGemma + LoRA."""

    def __init__(self, config: dict[str, Any], adapter_dir: str | Path | None = None):
        """
        Args:
            config: Config global del experimento.
            adapter_dir: Directorio del adapter LoRA. Si es None, se usa MedGemma base.
        """
        self.config = config
        self.model_config = config["model"]
        self.device = self.model_config["device"]
        self.dtype = torch_dtype_from_config(self.model_config["dtype"], self.device)
        self.max_new_tokens = int(self.model_config["max_new_tokens"])
        self.token = get_hf_token()

        self.processor = AutoProcessor.from_pretrained(
            self.model_config["model_id"],
            token=self.token,
        )
        self.model = self._load_model(adapter_dir=adapter_dir)
        self.model.eval()

    def generate(self, prompt: str, image_path: str | Path, max_new_tokens: int | None = None) -> str:
        """Genera una descripcion clinica para una imagen."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.open(image_path).convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_len = inputs["input_ids"].shape[-1]

        for key, value in inputs.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    inputs[key] = value.to(device=self.device, dtype=self.dtype)
                else:
                    inputs[key] = value.to(device=self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
            )

        generated = output[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def _load_model(self, adapter_dir: str | Path | None) -> torch.nn.Module:
        quantization_config = None
        kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "token": self.token,
        }

        if self.model_config.get("load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.model_config.get("use_double_quant", True),
                bnb_4bit_quant_type=self.model_config.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=self.dtype,
            )
            kwargs["device_map"] = {"": device_index(self.device)}
            kwargs["quantization_config"] = quantization_config

        logger.info("Cargando modelo base: %s", self.model_config["model_id"])
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_config["model_id"],
            **kwargs,
        )

        if adapter_dir is not None:
            logger.info("Cargando adapter LoRA desde: %s", adapter_dir)
            model = PeftModel.from_pretrained(model, str(adapter_dir))

        return model
