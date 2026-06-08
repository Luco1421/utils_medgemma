# -*- coding: utf-8 -*-
"""Helpers de entrenamiento LoRA: collator y trainer (TRL/SFT)."""


def make_collate_fn(processor):
    """Crea el collator que arma el chat y enmascara labels (imagen/padding)."""
    # id del token de imagen de Gemma 3 (placeholder de imagen en la secuencia)
    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.special_tokens_map["boi_token"]
    )

    def collate_fn(examples):
        texts, images = [], []
        for ex in examples:
            img = ex["image"].convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex["question"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": ex["answer"]},
                ]},
            ]
            text = processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            ).strip()
            texts.append(text)
            images.append([img])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100  # <image_soft_token> de Gemma 3
        batch["labels"] = labels
        return batch

    return collate_fn


def make_trainer(model, processor, train_dataset, collate_fn, *,
                 output_dir="medgemma-glaucoma-lora", use_qlora=True,
                 epochs=3, lr=2e-4):
    """Construye un SFTTrainer configurado para fine-tuning multimodal."""
    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch_fused",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        # Claves para multimodal: no preparar el dataset por su cuenta y no quitar columnas.
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],
    )

    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )
