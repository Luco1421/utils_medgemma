"""Non-contractual MedGemma-LoRA training experiment (not QLoRA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MedGemmaLoRAConfig:
    output_dir: str
    max_steps: int = 60
    num_epochs: float | None = None
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 4
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        # num_epochs, when set, takes precedence over max_steps so the schedule
        # is dataset-relative (the right unit for small-data SFT comparisons).
        if self.num_epochs is not None and self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive when provided")


def _extract_image(messages: list[dict[str, Any]]):
    for message in messages:
        for element in message.get("content", []):
            if isinstance(element, dict) and element.get("type") == "image":
                return element["image"].convert("RGB")
    raise ValueError("No image found in messages")


def build_response_only_collator(processor):
    """Mask image, prompt, and padding tokens; train only assistant text."""

    def collate(examples):
        full_texts = []
        prompt_texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            full_texts.append(
                processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                ).strip()
            )
            prompt_texts.append(
                processor.apply_chat_template(
                    messages[:-1],
                    add_generation_prompt=True,
                    tokenize=False,
                ).strip()
            )
            images.append(_extract_image(messages))

        batch = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        prompt_batch = processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            active = batch["attention_mask"][row_index].nonzero(as_tuple=True)[0]
            labels[row_index, active[:int(prompt_length)]] = -100

        for token_name in ("boi_token_id", "image_token_id", "eoi_token_id"):
            token_id = getattr(processor.tokenizer, token_name, None)
            if token_id is not None:
                labels[labels == token_id] = -100
        batch["labels"] = labels
        return batch

    return collate


def train_lora(
    model: Any,
    processor: Any,
    train_dataset: list[dict[str, Any]],
    config: MedGemmaLoRAConfig,
):
    """Train a full-precision base model with PEFT LoRA adapters."""

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dtype = next(model.parameters()).dtype
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    # Either train by epochs (dataset-relative) or by a fixed step budget.
    if config.num_epochs is not None:
        schedule = {"num_train_epochs": float(config.num_epochs), "max_steps": -1}
    else:
        schedule = {"max_steps": config.max_steps}
    training_args = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        **schedule,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=build_response_only_collator(processor),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    trainer.model.eval()
    return trainer.model
