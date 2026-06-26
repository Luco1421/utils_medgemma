"""Train the non-contractual MedGemma-LoRA experiment."""

from __future__ import annotations

import argparse
import logging
import os
import time

from huggingface_hub import get_token

from medgemma_utils.config import load_project_config
from medgemma_utils.dataset import (
    build_sft_examples,
    load_split_rows,
    summarize_splits,
)
from medgemma_utils.experimental_lora_training import (
    MedGemmaLoRAConfig,
    train_lora,
    write_training_metadata,
)
from medgemma_utils.runtime import format_duration, utc_now_iso

LOGGER = logging.getLogger(__name__)


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
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument(
        "--save-strategy",
        choices=("no", "steps", "epoch"),
        help="Intermediate checkpoint strategy. Final adapter is always saved.",
    )
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        help="Overrides WANDB_MODE/config tracking.wandb_mode.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        help="Optional W&B tags. WANDB_TAGS can also provide comma-separated tags.",
    )
    return parser.parse_args()


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _resolve_tags(args: argparse.Namespace) -> list[str]:
    if args.wandb_tags is not None:
        return [tag for tag in args.wandb_tags if tag]
    env_tags = os.environ.get("WANDB_TAGS")
    if not env_tags:
        return []
    return [tag.strip() for tag in env_tags.split(",") if tag.strip()]


def _init_wandb(
    args: argparse.Namespace,
    project_config: dict,
    metadata: dict,
):
    tracking_config = project_config.get("tracking") or {}
    mode = (
        args.wandb_mode
        or os.environ.get("WANDB_MODE")
        or tracking_config.get("wandb_mode")
        or "online"
    )
    project = _clean_optional(
        args.wandb_project
        or os.environ.get("WANDB_PROJECT")
        or tracking_config.get("wandb_project")
    )
    if mode == "disabled" or not project:
        LOGGER.info("W&B disabled")
        return None, "none"

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B was requested, but the wandb package is not installed. "
            "Install requirements.txt or set WANDB_MODE=disabled."
        ) from exc

    entity = _clean_optional(
        args.wandb_entity
        or os.environ.get("WANDB_ENTITY")
        or tracking_config.get("wandb_entity")
    )
    run_name = _clean_optional(args.wandb_run_name)
    tags = _resolve_tags(args)
    init_kwargs = {
        "project": project,
        "config": metadata,
    }
    if entity:
        init_kwargs["entity"] = entity
    if run_name:
        init_kwargs["name"] = run_name
    if tags:
        init_kwargs["tags"] = tags
    if mode != "online":
        init_kwargs["mode"] = mode
    wandb.init(**init_kwargs)
    LOGGER.info(
        "W&B enabled: project=%s entity=%s mode=%s run=%s",
        project,
        entity or "<default>",
        mode,
        run_name or "<auto>",
    )
    return wandb, "wandb"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_started_at = utc_now_iso()
    run_started = time.perf_counter()
    args = parse_args()
    if args.train_limit is not None and args.train_limit <= 0:
        raise ValueError("--train-limit must be positive when provided")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    project_config = load_project_config(args.config)
    experiment_config = project_config["experimental_medgemma_lora"]
    model_name = project_config["medgemma"]["model_name"]
    split_file = args.split_file or project_config["data"]["split_file"]
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    split_rows = load_split_rows(split_file)
    split_summary = summarize_splits(split_rows)
    rows = split_rows["train"]
    train_rows_total = len(rows)
    if args.train_limit is not None:
        rows = rows[: args.train_limit]
    dataset = build_sft_examples(rows)

    full_train_split = args.train_limit is None
    if full_train_split:
        LOGGER.info(
            "Training uses the full train split: %d/%d rows, %d SFT examples",
            len(rows),
            train_rows_total,
            len(dataset),
        )
    else:
        LOGGER.warning(
            "Training uses a limited subset: %d/%d rows, %d SFT examples",
            len(rows),
            train_rows_total,
            len(dataset),
        )

    seed = args.seed if args.seed is not None else int(project_config["seed"])
    save_strategy = args.save_strategy
    if save_strategy is None:
        save_strategy = "epoch" if args.num_epochs is not None else "steps"
    save_steps = (
        args.save_steps
        if args.save_steps is not None
        else int(experiment_config.get("save_steps", 50))
    )
    save_total_limit = (
        args.save_total_limit
        if args.save_total_limit is not None
        else experiment_config.get("save_total_limit", 2)
    )
    logging_steps = (
        args.logging_steps
        if args.logging_steps is not None
        else int(experiment_config.get("logging_steps", 5))
    )
    training_metadata = {
        "schema_version": "medgemma-lora-training-v1",
        "started_at": run_started_at,
        "model_name": model_name,
        "output_dir": args.output_dir,
        "seed": seed,
        "dataset": {
            "split_file": split_file,
            "split_summary": split_summary,
            "train_rows_total": train_rows_total,
            "train_rows_used": len(rows),
            "train_limit": args.train_limit,
            "full_train_split": full_train_split,
            "sft_examples": len(dataset),
        },
        "training": {
            "max_steps": (
                args.max_steps
                if args.max_steps is not None
                else int(experiment_config["max_steps"])
            ),
            "num_epochs": args.num_epochs,
            "learning_rate": (
                args.learning_rate
                if args.learning_rate is not None
                else float(experiment_config["learning_rate"])
            ),
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
                if args.gradient_accumulation_steps is not None
                else int(experiment_config["gradient_accumulation_steps"])
            ),
            "lora_rank": (
                args.lora_rank
                if args.lora_rank is not None
                else int(experiment_config["lora_rank"])
            ),
            "lora_alpha": (
                args.lora_alpha
                if args.lora_alpha is not None
                else int(experiment_config["lora_alpha"])
            ),
            "lora_dropout": float(experiment_config["lora_dropout"]),
            "logging_steps": logging_steps,
            "save_strategy": save_strategy,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
        },
    }
    wandb_module, report_to = _init_wandb(args, project_config, training_metadata)
    if wandb_module is not None:
        wandb_module.log({
            "data/train_rows_total": train_rows_total,
            "data/train_rows_used": len(rows),
            "data/sft_examples": len(dataset),
            "data/full_train_split": int(full_train_split),
        })

    try:
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
        result = train_lora(
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
                seed=seed,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                report_to=report_to,
                run_name=_clean_optional(args.wandb_run_name),
            ),
        )
        elapsed = time.perf_counter() - run_started
        training_metadata.update({
            "finished_at": utc_now_iso(),
            "runtime": {
                "duration_seconds": round(elapsed, 3),
                "duration": format_duration(elapsed),
            },
            "train_metrics": result.train_metrics,
            "saved_artifacts": result.saved_artifacts,
        })
        write_training_metadata(args.output_dir, training_metadata)
        LOGGER.info(
            "Saved LoRA adapter and metadata to %s (%d files)",
            args.output_dir,
            len(result.saved_artifacts),
        )
        if wandb_module is not None:
            wandb_module.log({
                "runtime/train_job_seconds": round(elapsed, 3),
                "artifacts/final_file_count": len(result.saved_artifacts),
            })
    finally:
        if wandb_module is not None:
            wandb_module.finish()


if __name__ == "__main__":
    main()
