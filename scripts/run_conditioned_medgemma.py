"""Run A-D2 using dataset oracle inputs or real pipeline outputs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import get_token

from medgemma_utils.config import load_project_config
from medgemma_utils.dataset import load_split_rows
from medgemma_utils.evaluation import Evaluator
from medgemma_utils.experiment import run_conditioned_experiment
from medgemma_utils.inputs import load_json_inputs
from medgemma_utils.medgemma_conditioner import MedGemmaConditioner
from medgemma_utils.oracle_inputs import build_dataset_oracle_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--inputs-json")
    parser.add_argument("--split-file")
    parser.add_argument(
        "--split-name",
        choices=("test", "validation", "train"),
        default="test",
        help="Which split to evaluate. Use 'validation' for hyperparameter "
        "selection so 'test' stays untouched until the final report.",
    )
    parser.add_argument("--output", default="results/conditioned_medgemma.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--masked-only", action="store_true")
    parser.add_argument("--adapter-path")
    parser.add_argument("--oracle-confidence", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_config = load_project_config(args.config)
    seed = int(project_config["seed"])
    split_file = args.split_file or project_config["data"]["split_file"]
    oracle_confidence = (
        args.oracle_confidence
        if args.oracle_confidence is not None
        else float(project_config["data"]["oracle_confidence"])
    )
    if args.inputs_json:
        samples = load_json_inputs(args.inputs_json)
    else:
        rows = load_split_rows(split_file)[args.split_name]
        if args.masked_only:
            rows = [row for row in rows if row["has_ground_truth_mask"]]
        samples = build_dataset_oracle_inputs(
            rows,
            confidence=oracle_confidence,
            require_mask=args.masked_only,
        )
    if args.limit:
        samples = samples[: args.limit]

    config = {
        **project_config["medgemma"],
        "seed": seed,
        "token": os.environ.get("HF_TOKEN") or get_token(),
    }
    if args.adapter_path:
        from medgemma_utils.experimental_lora_conditioner import (
            ExperimentalLoRAMedGemmaConditioner,
        )

        conditioner = ExperimentalLoRAMedGemmaConditioner({
            **config,
            "adapter_path": args.adapter_path,
        })
    else:
        conditioner = MedGemmaConditioner(config)
    evaluation_config = {
        **project_config["evaluation"],
        "seed": seed,
        "token": config["token"],
    }
    evaluator = Evaluator(evaluation_config)
    experiment = run_conditioned_experiment(
        conditioner,
        evaluator,
        samples,
        pipeline_name="dataset_oracle",
        model_variant="medgemma_lora" if args.adapter_path else "medgemma_base",
    )
    payload = {
        "schema_version": "m7-m8-v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": config["model_name"],
        "adapter_path": args.adapter_path,
        "split_name": args.split_name,
        "seed": config["seed"],
        "config": {
            "medgemma": {
                key: value for key, value in config.items() if key != "token"
            },
            "evaluation": {
                key: value
                for key, value in evaluation_config.items()
                if key != "token"
            },
        },
        **experiment,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
