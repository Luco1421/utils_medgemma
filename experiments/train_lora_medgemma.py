"""Entrenamiento LoRA/QLoRA para MedGemma.

El modulo expone `train_lora_medgemma(...)` para notebooks y tambien conserva un
CLI delgado. El dataset esperado en JSONL es:
    {"image": "images/001.jpg", "prompt": "...", "answer": "..."}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma with LoRA/QLoRA.")
    parser.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", default="checkpoints/medgemma_lora")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-qlora", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Carga ejemplos JSONL y valida campos minimos."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"image", "prompt", "answer"} - row.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing fields: {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError("Training dataset is empty.")
    return rows


def resolve_image_path(image_value: str, image_root: str | Path | None) -> Path:
    """Resuelve rutas absolutas o relativas al directorio de imagenes."""
    image_path = Path(image_value)
    if image_path.is_absolute() or image_root is None:
        return image_path
    return Path(image_root) / image_path


def build_training_conversations(
    rows: list[dict[str, Any]],
    image_root: str | Path | None,
) -> list[dict[str, Any]]:
    """Convierte JSONL plano a conversaciones multimodales para SFT."""
    conversations = []
    for row in rows:
        resolved = resolve_image_path(row["image"], image_root)
        if not resolved.exists():
            raise FileNotFoundError(f"Image not found: {resolved}")
        conversations.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.open(resolved).convert("RGB")},
                            {"type": "text", "text": row["prompt"]},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": row["answer"]}],
                    },
                ]
            }
        )
    return conversations


def extract_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    """Extrae imagenes PIL desde una conversacion multimodal."""
    images = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and element.get("type") == "image":
                images.append(element["image"].convert("RGB"))
    return images


def make_collate_fn(processor: Any):
    """Crea un collator compatible con SFTTrainer y MedGemma multimodal."""

    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            texts.append(
                processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                ).strip()
            )
            images.append(extract_images(messages))

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        for token_name in ("boi_token_id", "image_token_id", "eoi_token_id"):
            token_id = getattr(processor.tokenizer, token_name, None)
            if token_id is not None:
                labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def train_lora_medgemma(
    model_id: str,
    train_jsonl: str | Path,
    output_dir: str | Path,
    image_root: str | Path | None = None,
    hf_token: str | None = None,
    max_steps: int = 100,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_qlora: bool = False,
) -> str:
    """Entrena un adapter LoRA/QLoRA y retorna su directorio."""
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Install torch, transformers, accelerate, peft, trl, bitsandbytes and pillow."
        ) from exc

    token = hf_token or os.environ.get("HF_TOKEN") or True
    rows = load_jsonl(train_jsonl)
    train_dataset = build_training_conversations(rows, image_root=image_root)

    processor = AutoProcessor.from_pretrained(model_id, token=token)
    quantization_config = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        token=token,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        processing_class=processor,
        data_collator=make_collate_fn(processor),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    return str(output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    output_dir = train_lora_medgemma(
        model_id=args.model_id,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        image_root=args.image_root,
        hf_token=args.hf_token,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_qlora=args.use_qlora,
    )
    print(f"Saved LoRA adapter: {output_dir}")


if __name__ == "__main__":
    main()
