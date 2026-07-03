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
from medgemma_utils.dataset import load_split_rows
from medgemma_utils.evaluation import Evaluator
from medgemma_utils.experiment import run_conditioned_experiment
from medgemma_utils.inputs import load_json_inputs
from medgemma_utils.oracle_inputs import build_dataset_oracle_inputs
from medgemma_utils.pipeline_inputs import load_pipeline_dir_inputs
from medgemma_utils.runtime import format_duration, timed_stage, utc_now_iso

LOGGER = logging.getLogger(__name__)

# Classifier output encodes the label as 1=glaucoma, 0=normal.
_CLASSIFIER_LABELS = {1: "glaucoma", 0: "normal"}


def load_pipeline_classifications(
    classifications_path: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Read the pipeline classifier file (per_image y_pred/y_true) into two maps.

    Returns ``(predictions, ground_truth_labels)``, both keyed by ``image_id``
    with ``"glaucoma"``/``"normal"`` values, so the classifier prediction and
    the ground-truth label used to judge it share one traceable source.
    """

    data = json.loads(Path(classifications_path).read_text(encoding="utf-8"))
    per_image = data["per_image"]
    predictions: dict[str, str] = {}
    ground_truth_labels: dict[str, str] = {}
    for row in per_image:
        image_id = str(row["image_id"])
        predictions[image_id] = _CLASSIFIER_LABELS[int(row["y_pred"])]
        ground_truth_labels[image_id] = _CLASSIFIER_LABELS[int(row["y_true"])]
    LOGGER.info(
        "Loaded %d classifier predictions and ground-truth labels from %s",
        len(predictions),
        classifications_path,
    )
    return predictions, ground_truth_labels


def _label_counts(samples) -> dict[str, int]:
    """Count evaluated images by ground-truth label."""

    counts = {"total": len(samples), "glaucoma": 0, "normal": 0}
    for sample in samples:
        label = sample.expected_finding
        if label in counts:
            counts[label] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--inputs-json")
    parser.add_argument(
        "--pipeline-dir",
        help="Folder with pipeline-style inputs (image + segmentor overlay), "
        "e.g. dataset/test. Uses GT labels as classifier stand-in unless "
        "--predictions-json is given.",
    )
    parser.add_argument(
        "--predictions-json",
        help="Optional JSON mapping image_id -> 'glaucoma'/'normal' for "
        "--pipeline-dir (real classifier output).",
    )
    parser.add_argument(
        "--pipeline-classifications-json",
        help="Pipeline classifier file (e.g. pipeline_classifications.json) "
        "with a 'per_image' list of {image_id, y_pred, y_true}. Derives BOTH "
        "the classifier prediction (y_pred) and the ground-truth label used to "
        "judge it (y_true) from the same artifact for traceability. "
        "Mapping: 1=glaucoma, 0=normal.",
    )
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
    if args.pipeline_dir:
        dataset_dir = Path(split_file).parent
        predictions = None
        ground_truth_labels = None
        if args.pipeline_classifications_json:
            predictions, ground_truth_labels = load_pipeline_classifications(
                args.pipeline_classifications_json
            )
        if args.predictions_json:
            # Explicit predictions override the metrics-derived y_pred.
            predictions = json.loads(
                Path(args.predictions_json).read_text(encoding="utf-8")
            )
        # Segmentation scoring (predicted mask vs GT disc) is deferred to the
        # analysis phase: the segmentor mask and the GT disc live at different
        # resolutions, so comparing them needs explicit alignment handling.
        # Omitting dataset_root keeps ground_truth_mask=None and skips it here.
        samples = load_pipeline_dir_inputs(
            args.pipeline_dir,
            annotations_file=dataset_dir / "annotations.json",
            predictions=predictions,
            ground_truth_labels=ground_truth_labels,
        )
        dataset_audit = {
            "input_mode": "pipeline_dir",
            "pipeline_dir": args.pipeline_dir,
            "pipeline_classifications_json": args.pipeline_classifications_json,
        }
    elif args.inputs_json:
        samples = load_json_inputs(args.inputs_json)
        dataset_audit = {
            "input_mode": "inputs_json",
            "inputs_json": args.inputs_json,
        }
    else:
        rows = split_rows[args.split_name]
        if args.masked_only:
            rows = [row for row in rows if row["has_ground_truth_mask"]]
        samples = build_dataset_oracle_inputs(
            rows,
            require_mask=args.masked_only,
        )
        dataset_audit = {
            "input_mode": "dataset_oracle",
            "split_file": split_file,
            "selected_split": args.split_name,
        }
    if args.limit is not None:
        samples = samples[: args.limit]
        dataset_audit["limit"] = args.limit
    if args.masked_only:
        dataset_audit["masked_only"] = True
    dataset_audit["samples_evaluated"] = len(samples)
    dataset_audit["evaluated_counts"] = _label_counts(samples)
    LOGGER.info(
        "Evaluation: mode=%s samples=%d %s",
        dataset_audit["input_mode"],
        len(samples),
        dataset_audit["evaluated_counts"],
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
            model_variant=(
                "medgemma_lora" if args.adapter_path else "medgemma_base"
            ),
        )
    stage_timings.append(stage)

    elapsed = time.perf_counter() - run_started
    # ``seed`` is reported once at the top level; it is the single project seed
    # shared by the model and the evaluator, so it is stripped from both config
    # blocks below to avoid three copies of the same number.
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": config["model_name"],
        "adapter_path": args.adapter_path,
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
                key: value
                for key, value in config.items()
                if key not in {"token", "seed"}
            },
            "evaluation": {
                key: value
                for key, value in evaluation_config.items()
                if key not in {"token", "seed"}
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
