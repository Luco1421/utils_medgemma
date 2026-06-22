"""Validate configuration, dataset, environment and model access prerequisites."""

from __future__ import annotations

import argparse
import json
import logging
import os

from huggingface_hub import get_token

from medgemma_utils.config import load_project_config
from medgemma_utils.dataset import (
    load_split_rows,
    summarize_splits,
    validate_optic_disc_mask_pixels,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-hf-token", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = load_project_config(args.config)
    rows = load_split_rows(config["data"]["split_file"])
    summary = summarize_splits(rows)
    expected = config["data"]["expected_splits"]
    if summary != expected:
        raise RuntimeError(
            "Dataset summary differs from config.yaml: "
            f"actual={summary}, expected={expected}"
        )
    validated_masks = validate_optic_disc_mask_pixels(rows)

    token = os.environ.get("HF_TOKEN") or get_token()
    if args.require_hf_token and not token:
        raise RuntimeError("Hugging Face login missing; run: hf auth login")

    LOGGER.info("Dataset: %s", json.dumps(summary, sort_keys=True))
    LOGGER.info("Mask target: optic_disc")
    LOGGER.info("Validated binary optic-disc masks: %d", validated_masks)
    LOGGER.info("Normal images bypass segmentation: true")
    LOGGER.info("Hugging Face token available: %s", bool(token))
    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        LOGGER.info("CUDA available: true")
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
