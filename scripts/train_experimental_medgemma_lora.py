"""Train the non-contractual MedGemma-LoRA experiment."""

from __future__ import annotations

import argparse
import os

import torch
from huggingface_hub import get_token
from transformers import AutoModelForImageTextToText, AutoProcessor

from medgemma_utils.config import load_project_config
from medgemma_utils.dataset import (
    build_conditioned_sft_examples,
    build_sft_examples,
    load_split_rows,
)
from medgemma_utils.experimental_lora_training import (
    MedGemmaLoRAConfig,
    train_lora,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split-file")
    parser.add_argument(
        "--output-dir",
        default="checkpoints/experimental_medgemma_lora",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--num-epochs",
        type=float,
        help="Train for this many epochs (dataset-relative); overrides max-steps.",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--prompt-mode",
        choices=("generic", "conditioned"),
        default="generic",
        help=(
            "generic: single condition-A prompt per image. "
            "conditioned: one example per applicable condition with overlays "
            "and class hints, matching how the adapter is evaluated."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_config = load_project_config(args.config)
    experiment_config = project_config["experimental_medgemma_lora"]
    model_name = project_config["medgemma"]["model_name"]
    split_file = args.split_file or project_config["data"]["split_file"]
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    rows = load_split_rows(split_file)["train"]
    if args.train_limit:
        rows = rows[: args.train_limit]
    if args.prompt_mode == "conditioned":
        dataset = build_conditioned_sft_examples(
            rows,
            confidence=float(project_config["data"]["oracle_confidence"]),
        )
    else:
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
            max_steps=(
                args.max_steps
                if args.max_steps is not None
                else int(experiment_config["max_steps"])
            ),
            num_epochs=args.num_epochs,
            learning_rate=(
                args.learning_rate
                if args.learning_rate is not None
                else float(experiment_config["learning_rate"])
            ),
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
                if args.gradient_accumulation_steps is not None
                else int(experiment_config["gradient_accumulation_steps"])
            ),
            lora_rank=(
                args.lora_rank
                if args.lora_rank is not None
                else int(experiment_config["lora_rank"])
            ),
            lora_alpha=(
                args.lora_alpha
                if args.lora_alpha is not None
                else int(experiment_config["lora_alpha"])
            ),
            lora_dropout=float(experiment_config["lora_dropout"]),
            seed=(
                args.seed
                if args.seed is not None
                else int(project_config["seed"])
            ),
        ),
    )


if __name__ == "__main__":
    main()
