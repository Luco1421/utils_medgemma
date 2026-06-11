"""Entrena el adapter LoRA/QLoRA de MedGemma."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from modules.data_module import DataModule
from modules.lora_trainer import MedGemmaLoraTrainer
from modules.utils import load_config, save_json, set_global_seed, setup_logging, utc_timestamp

logger = logging.getLogger(__name__)


def train_lora(config: dict) -> dict:
    """Entrena LoRA y guarda metadatos de entrenamiento."""
    set_global_seed(int(config["seed"]))
    data_module = DataModule(config["data"])
    train_limit = config["training"].get("train_limit")
    train_rows = data_module.get_train_rows(train_limit)

    trainer = MedGemmaLoraTrainer(config)
    adapter_dir = trainer.train(train_rows)

    payload = {
        "timestamp": utc_timestamp(),
        "seed": config["seed"],
        "config": config,
        "model_variant": "lora_adapter",
        "adapter_dir": str(adapter_dir),
        "train_examples": len(train_rows),
    }
    output_path = Path(config["outputs"]["results_dir"]) / "lora_training_metadata.json"
    save_json(output_path, payload)
    logger.info("Metadata de entrenamiento guardada en: %s", output_path)
    return payload


def parse_args() -> argparse.Namespace:
    """Parsea argumentos CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    """Entry point CLI."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    train_lora(config)


if __name__ == "__main__":
    main()
