"""Run A-D2 using dataset oracle inputs or real pipeline outputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import get_token

from medgemma_utils.config import load_project_config
from medgemma_utils.dataset import load_split_rows, summarize_splits
from medgemma_utils.evaluation import Evaluator
from medgemma_utils.experiment import run_conditioned_experiment
from medgemma_utils.inputs import load_json_inputs
from medgemma_utils.oracle_inputs import build_dataset_oracle_inputs
from medgemma_utils.runtime import format_duration, timed_stage, utc_now_iso

LOGGER = logging.getLogger(__name__)


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_started_at = utc_now_iso()
    run_started = time.perf_counter()
    stage_timings = []
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")
    project_config = load_project_config(args.config)
    seed = int(project_config["seed"])
    split_file = args.split_file or project_config["data"]["split_file"]
    split_rows = load_split_rows(split_file)
    split_summary = summarize_splits(split_rows)
    oracle_confidence = (
        args.oracle_confidence
        if args.oracle_confidence is not None
        else float(project_config["data"]["oracle_confidence"])
    )
    if args.inputs_json:
        samples = load_json_inputs(args.inputs_json)
        dataset_audit = {
            "input_mode": "inputs_json",
            "inputs_json": args.inputs_json,
            "split_file": split_file,
            "split_summary": split_summary,
            "samples_before_limit": len(samples),
            "limit": args.limit,
            "masked_only": args.masked_only,
            "full_selected_split": False,
        }
    else:
        rows = split_rows[args.split_name]
        selected_rows_before_filters = len(rows)
        if args.masked_only:
            rows = [row for row in rows if row["has_ground_truth_mask"]]
        selected_rows_after_filters = len(rows)
        samples = build_dataset_oracle_inputs(
            rows,
            confidence=oracle_confidence,
            require_mask=args.masked_only,
        )
        dataset_audit = {
            "input_mode": "dataset_oracle",
            "split_file": split_file,
            "split_summary": split_summary,
            "selected_split": args.split_name,
            "selected_rows_before_filters": selected_rows_before_filters,
            "selected_rows_after_filters": selected_rows_after_filters,
            "samples_before_limit": len(samples),
            "limit": args.limit,
            "masked_only": args.masked_only,
            "full_selected_split": args.limit is None and not args.masked_only,
        }
    if args.limit is not None:
        samples = samples[: args.limit]
    dataset_audit["samples_evaluated"] = len(samples)
    if dataset_audit["full_selected_split"]:
        LOGGER.info(
            "Evaluation uses the full %s split: %d samples",
            args.split_name,
            len(samples),
        )
    else:
        LOGGER.warning(
            "Evaluation uses a limited/custom input: mode=%s samples=%d limit=%s masked_only=%s",
            dataset_audit["input_mode"],
            len(samples),
            args.limit,
            args.masked_only,
        )

    config = {
        **project_config["medgemma"],
        "seed": seed,
        "token": os.environ.get("HF_TOKEN") or get_token(),
    }
    with timed_stage(LOGGER, "Model and evaluator load") as stage:
        if args.adapter_path:
            from medgemma_utils.experimental_lora_conditioner import (
                ExperimentalLoRAMedGemmaConditioner,
            )

            conditioner = ExperimentalLoRAMedGemmaConditioner({
                **config,
                "adapter_path": args.adapter_path,
            })
        else:
            from medgemma_utils.medgemma_conditioner import MedGemmaConditioner

            conditioner = MedGemmaConditioner(config)
        evaluation_config = {
            **project_config["evaluation"],
            "seed": seed,
            "token": config["token"],
        }
        evaluator = Evaluator(evaluation_config)
    stage_timings.append(stage)

    with timed_stage(LOGGER, "Conditioned generation and evaluation") as stage:
        experiment = run_conditioned_experiment(
            conditioner,
            evaluator,
            samples,
            pipeline_name="dataset_oracle",
            model_variant=(
                "medgemma_lora" if args.adapter_path else "medgemma_base"
            ),
        )
    stage_timings.append(stage)

    elapsed = time.perf_counter() - run_started
    payload = {
        "schema_version": "m7-m8-v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": config["model_name"],
        "adapter_path": args.adapter_path,
        "split_name": args.split_name,
        "seed": config["seed"],
        "dataset_audit": dataset_audit,
        "runtime": {
            "started_at": run_started_at,
            "finished_at": utc_now_iso(),
            "duration_seconds": round(elapsed, 3),
            "duration": format_duration(elapsed),
            "stages": stage_timings,
        },
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
    payload["generated_result_count"] = len(payload["results"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info(
        "Wrote %s with %d generated rows in %s",
        output,
        payload["generated_result_count"],
        payload["runtime"]["duration"],
    )


if __name__ == "__main__":
    main()
