"""Run A-D2 from Python using mock dataset inputs or real pipeline JSON."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from medgemma_utils.dataset import load_split_rows
from medgemma_utils.experiment import run_conditioned_experiment
from medgemma_utils.inputs import load_json_inputs
from medgemma_utils.medgemma_conditioner import MedGemmaConditioner
from medgemma_utils.mock_inputs import build_dataset_mock_inputs
from huggingface_hub import get_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-json")
    parser.add_argument("--split-file", default="dataset/split_repetition_1.json")
    parser.add_argument("--output", default="results/conditioned_medgemma.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter-path")
    parser.add_argument("--mock-confidence", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inputs_json:
        samples = load_json_inputs(args.inputs_json)
    else:
        rows = load_split_rows(args.split_file)["test"]
        samples = build_dataset_mock_inputs(rows, confidence=args.mock_confidence)
    if args.limit:
        samples = samples[: args.limit]

    config = {
        "model_name": "google/medgemma-1.5-4b-it",
        "torch_dtype": "auto",
        "max_new_tokens": 384,
        "seed": 42,
        "token": os.environ.get("HF_TOKEN") or get_token(),
    }
    if args.adapter_path:
        config["adapter_path"] = args.adapter_path
    conditioner = MedGemmaConditioner(config)
    experiment = run_conditioned_experiment(
        conditioner,
        samples,
        model_variant="medgemma_lora" if args.adapter_path else "medgemma_base",
        bertscore_model=(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        ),
        bertscore_num_layers=12,
    )
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": config["model_name"],
        "adapter_path": args.adapter_path,
        **experiment,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
