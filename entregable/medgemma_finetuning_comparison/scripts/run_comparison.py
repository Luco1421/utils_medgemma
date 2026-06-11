"""Corre la comparacion completa: base, entrenamiento LoRA y evaluacion LoRA."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from modules.evaluator import DescriptionEvaluator
from modules.utils import load_config, save_json, setup_logging, utc_timestamp
from scripts.evaluate_base import evaluate_base
from scripts.evaluate_lora import evaluate_lora
from scripts.train_lora import train_lora

logger = logging.getLogger(__name__)


def run_comparison(config: dict) -> dict:
    """Ejecuta el flujo completo y guarda comparacion agregada."""
    base_payload = evaluate_base(config)
    training_payload = train_lora(config)
    lora_payload = evaluate_lora(config, adapter_dir=training_payload["adapter_dir"])

    evaluator = DescriptionEvaluator(config)
    comparison = evaluator.compare(
        base_payload["summary"],
        lora_payload["summary"],
    )
    payload = {
        "timestamp": utc_timestamp(),
        "seed": config["seed"],
        "config": config,
        "train_examples": training_payload["train_examples"],
        "adapter_dir": training_payload["adapter_dir"],
        "split": config["evaluation"]["split"],
        "comparison": comparison,
        "base_results_file": config["outputs"]["base_results_file"],
        "lora_results_file": config["outputs"]["lora_results_file"],
    }

    output_path = (
        Path(config["outputs"]["results_dir"]) / config["outputs"]["comparison_file"]
    )
    save_json(output_path, payload)
    logger.info("Comparacion final guardada en: %s", output_path)
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
    run_comparison(config)


if __name__ == "__main__":
    main()
