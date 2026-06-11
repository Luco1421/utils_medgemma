"""Entrenamiento LoRA/QLoRA para MedGemma."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from modules.utils import device_index, ensure_dir, get_hf_token, torch_dtype_from_config

logger = logging.getLogger(__name__)


class MedGemmaLoraTrainer:
    """Entrena un adapter LoRA/QLoRA sobre descripciones expertas."""

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Config global del experimento.
        """
        self.config = config
        self.model_config = config["model"]
        self.lora_config = config["lora"]
        self.training_config = config["training"]
        self.device = self.model_config["device"]
        self.dtype = torch_dtype_from_config(self.model_config["dtype"], self.device)
        self.token = get_hf_token()

        self.processor = AutoProcessor.from_pretrained(
            self.model_config["model_id"],
            token=self.token,
        )
        self.model = self._load_model()

    def train(self, train_rows: list[dict[str, Any]]) -> Path:
        """Entrena y guarda el adapter LoRA."""
        train_dataset = [self._to_description_messages(row) for row in train_rows]
        output_dir = ensure_dir(self.lora_config["output_dir"])

        peft_config = LoraConfig(
            r=int(self.lora_config["r"]),
            lora_alpha=int(self.lora_config["alpha"]),
            lora_dropout=float(self.lora_config["dropout"]),
            bias=self.lora_config["bias"],
            target_modules=self.lora_config["target_modules"],
            task_type="CAUSAL_LM",
        )

        training_args = SFTConfig(
            output_dir=str(output_dir),
            max_steps=int(self.training_config["max_steps"]),
            per_device_train_batch_size=int(
                self.training_config["per_device_train_batch_size"]
            ),
            gradient_accumulation_steps=int(
                self.training_config["gradient_accumulation_steps"]
            ),
            learning_rate=float(self.training_config["learning_rate"]),
            bf16=(self.dtype == torch.bfloat16),
            fp16=(self.dtype == torch.float16),
            logging_steps=int(self.training_config["logging_steps"]),
            save_steps=int(self.training_config["save_steps"]),
            save_total_limit=int(self.training_config["save_total_limit"]),
            remove_unused_columns=False,
            report_to="none",
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            processing_class=self.processor,
            data_collator=self._collate_fn,
        )

        logger.info("Iniciando entrenamiento LoRA con %d ejemplos", len(train_dataset))
        trainer.train()
        trainer.save_model(str(output_dir))
        self.processor.save_pretrained(str(output_dir))
        logger.info("Adapter LoRA guardado en: %s", output_dir)
        return output_dir

    def _load_model(self) -> torch.nn.Module:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bool(self.model_config["load_in_4bit"]),
            bnb_4bit_use_double_quant=bool(self.model_config["use_double_quant"]),
            bnb_4bit_quant_type=self.model_config["quant_type"],
            bnb_4bit_compute_dtype=self.dtype,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            self.model_config["model_id"],
            dtype=self.dtype,
            device_map={"": device_index(self.device)},
            quantization_config=bnb_config,
            token=self.token,
        )
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable()
        except Exception as exc:
            logger.warning("Gradient checkpointing no disponible: %s", exc)
        return model

    def _to_description_messages(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.open(row["image"]).convert("RGB")},
                        {"type": "text", "text": row["prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": row["answer"]}],
                },
            ]
        }

    def _extract_image(self, messages: list[dict[str, Any]]) -> Image.Image:
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                content = [content]
            for element in content:
                if isinstance(element, dict) and element.get("type") == "image":
                    return element["image"].convert("RGB")
        raise ValueError("No image found in training example")

    def _collate_fn(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                ).strip()
            )
            images.append(self._extract_image(messages))

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        for token_name in ("boi_token_id", "image_token_id", "eoi_token_id"):
            token_id = getattr(self.processor.tokenizer, token_name, None)
            if token_id is not None:
                labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch
