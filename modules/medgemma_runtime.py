"""Runtime reutilizable para inferencia con MedGemma."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class MedGemmaRuntime:
    """Carga MedGemma y genera texto con o sin imagen.

    Esta clase centraliza el camino de inferencia para que el baseline y un
    modelo adaptado con LoRA se prueben con el mismo formato de mensajes.
    """

    def __init__(self, config: dict[str, Any]):
        """Inicializa el runtime sin cargar el modelo hasta que sea necesario."""
        self.config = dict(config)
        self.model_id = str(self.config.get("model_id", self.config.get("model_name")))
        if not self.model_id or self.model_id == "None":
            self.model_id = "google/medgemma-1.5-4b-it"

        self.adapter_path = self.config.get("adapter_path")
        self.torch_dtype = str(self.config.get("torch_dtype", "bfloat16"))
        self.device_map = self.config.get("device_map", "auto")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 512))
        self.do_sample = bool(self.config.get("do_sample", False))
        self.temperature = float(self.config.get("temperature", 0.0))

        self.processor: Any | None = None
        self.model: Any | None = None
        self._torch: Any | None = None

    def generate(
        self,
        prompt: str,
        image: Any | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Genera texto a partir de un prompt y, opcionalmente, una imagen."""
        self.load()
        messages = self.build_messages(prompt=prompt, image=image)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs)

        input_len = inputs["input_ids"].shape[-1]
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature

        with self._torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    @classmethod
    def build_messages(cls, prompt: str, image: Any | None = None) -> list[dict[str, Any]]:
        """Construye mensajes compatibles con el chat template de MedGemma."""
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": cls.to_pil_image(image)})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @staticmethod
    def to_pil_image(image: Any) -> Any:
        """Convierte ruta o ndarray RGB a PIL.Image."""
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")

        image_array = np.asarray(image)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("image must be RGB with shape (H, W, 3).")
        return Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8)).convert("RGB")

    def load(self) -> None:
        """Carga modelo, processor y adapter opcional de LoRA."""
        if self.model is not None and self.processor is not None:
            return

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Install torch, transformers, accelerate and pillow before running MedGemma."
            ) from exc

        self._torch = torch
        dtype = getattr(torch, self.torch_dtype)
        LOGGER.info("Loading MedGemma: %s", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device_map,
        )

        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("Install peft before loading a LoRA adapter.") from exc
            LOGGER.info("Loading LoRA adapter: %s", self.adapter_path)
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()

    def _move_inputs(self, inputs: Any) -> Any:
        """Mueve tensores al dispositivo del modelo."""
        if self.device_map == "auto":
            first_device = next(iter(self.model.hf_device_map.values()), self.model.device)
            return inputs.to(first_device, dtype=getattr(self._torch, self.torch_dtype))
        return inputs.to(self.model.device, dtype=getattr(self._torch, self.torch_dtype))
