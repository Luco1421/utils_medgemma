# -*- coding: utf-8 -*-
"""Carga del modelo/processor, LoRA y la funcion de chat (prueba unitaria)."""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from . import config

# bfloat16 funciona en T4/A100. Si tu GPU diera problemas, prueba torch.float16.
DTYPE = torch.bfloat16


def load_model(model_id=None, dtype=DTYPE, use_qlora=False, for_training=False):
    """Carga processor + modelo.

    use_qlora=True     -> cuantiza a 4-bit (entrenamiento eficiente con QLoRA).
    for_training=True  -> desactiva cache (necesario para gradient checkpointing).
    """
    model_id = model_id or config.MODEL_ID

    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",   # recomendado para Gemma 3
    )
    if use_qlora:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if for_training:
        model.config.use_cache = False
    else:
        model.eval()
    return model, processor


def apply_lora(model, use_qlora=True, r=16, alpha=16, dropout=0.05, target_modules=None):
    """Envuelve el modelo con adaptadores LoRA y lo deja listo para entrenar."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if use_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


@torch.inference_mode()
def ask_image(model, processor, image, question,
              system_prompt="You are an expert ophthalmologist.",
              max_new_tokens=300, temperature=0.0, dtype=DTYPE):
    """Chat multimodal de un turno. Sirve como prueba unitaria del modelo."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": question}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=dtype)

    input_len = inputs["input_ids"].shape[-1]
    do_sample = temperature > 0.0
    gen = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
    )
    return processor.decode(gen[0][input_len:], skip_special_tokens=True).strip()
