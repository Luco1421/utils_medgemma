"""Evalua MedGemma base en el split configurado."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from modules.data_module import DataModule
from modules.evaluator import DescriptionEvaluator
from modules.medgemma_model import MedGemmaGenerator
from modules.utils import load_config, save_json, set_global_seed, setup_logging, utc_timestamp

logger = logging.getLogger(__name__)


def evaluate_base(config: dict) -> dict:
    """Ejecuta evaluacion del modelo base y guarda resultados."""
    set_global_seed(int(config["seed"]))
    data_module = DataModule(config["data"])
    eval_rows = data_module.get_eval_rows(config["evaluation"]["split"])

    generator = MedGemmaGenerator(config)
    outputs = []
    for index, row in enumerate(eval_rows, start=1):
        logger.info("BASE [%d/%d] %s", index, len(eval_rows), row["image"])
        generated = generator.generate(row["prompt"], row["image"])
        outputs.append(
            {
                "image": row["image"],
                "label": row["label"],
                "conditions": row["conditions"],
                "reference": row["reference"],
                "generated": generated,
            }
        )

    evaluator = DescriptionEvaluator(config)
    summary, items = evaluator.evaluate_outputs(outputs)
    payload = {
        "timestamp": utc_timestamp(),
        "seed": config["seed"],
        "config": config,
        "model_variant": "base",
        "split": config["evaluation"]["split"],
        "summary": summary,
        "results": items,
    }

    results_dir = Path(config["outputs"]["results_dir"])
    output_path = results_dir / config["outputs"]["base_results_file"]
    save_json(output_path, payload)
    logger.info("Resultados base guardados en: %s", output_path)
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
    evaluate_base(config)


if __name__ == "__main__":
    main()
