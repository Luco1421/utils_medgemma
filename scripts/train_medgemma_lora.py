"""Train standard MedGemma LoRA adapters from the shared dataset."""

from __future__ import annotations

import argparse
import os

import torch
from huggingface_hub import get_token
from transformers import AutoModelForImageTextToText, AutoProcessor

from medgemma_utils.dataset import build_sft_examples, load_split_rows
from medgemma_utils.lora_training import MedGemmaLoRAConfig, train_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", default="dataset/split_repetition_1.json")
    parser.add_argument(
        "--output-dir",
        default="checkpoints/official_medgemma_lora_description",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = "google/medgemma-1.5-4b-it"
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    rows = load_split_rows(args.split_file)["train"]
    if args.train_limit:
        rows = rows[: args.train_limit]
    dataset = build_sft_examples(rows)

    dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_name, token=token)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        token=token,
    )
    train_lora(
        model,
        processor,
        dataset,
        MedGemmaLoRAConfig(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        ),
    )


if __name__ == "__main__":
    main()
