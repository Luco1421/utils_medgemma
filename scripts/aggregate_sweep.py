"""Aggregate validation sweep results and select the best LoRA configuration.

Reads every ``val_idx*.json`` produced by the sweep array job, pools the
per-sample text metrics over all conditions and images, and ranks configurations
by a combined score. The combined score z-normalizes each metric across the
sweep (so different scales are comparable) and averages them, as requested:
BERTScore-F1 + calibrated SBERT + finding-mentioned rate.

Raw per-metric means are kept alongside the combined score for transparency.
The winner overall and per prompt mode are printed and written to a summary
JSON, ready to drive the final multi-seed test runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from scripts.sweep_grid import get_point

COMBINED_METRICS = ("bertscore_f1", "sbert_calibrated", "finding_mentioned")
_INDEX_RE = re.compile(r"val_idx(\d+)")


def _pool_metric(results: list[dict[str, Any]], metric: str) -> float:
    values = [
        float(item["text_metrics"][metric])
        for item in results
        if item.get("text_metrics", {}).get(metric) is not None
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _load_records(sweep_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(sweep_dir.glob("val_idx*.json")):
        match = _INDEX_RE.search(path.name)
        if not match:
            continue
        index = int(match.group(1))
        point = get_point(index)
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        record = {
            "index": index,
            "tag": point.tag,
            "prompt_mode": point.prompt_mode,
            "lora_rank": point.lora_rank,
            "lora_alpha": point.lora_alpha,
            "learning_rate": point.learning_rate,
            "num_epochs": point.num_epochs,
            "split_name": payload.get("split_name"),
            "result_count": len(results),
            "metrics": {m: _pool_metric(results, m) for m in COMBINED_METRICS},
        }
        records.append(record)
    return records


def _add_combined_score(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    z_by_metric: dict[str, np.ndarray] = {}
    for metric in COMBINED_METRICS:
        values = np.array([r["metrics"][metric] for r in records], dtype=float)
        std = np.nanstd(values)
        mean = np.nanmean(values)
        z_by_metric[metric] = (
            np.zeros_like(values) if std == 0 else (values - mean) / std
        )
    for position, record in enumerate(records):
        z_values = [z_by_metric[m][position] for m in COMBINED_METRICS]
        record["combined_z"] = float(np.nanmean(z_values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default="results/sweep")
    parser.add_argument("--output", default="results/sweep/summary.json")
    args = parser.parse_args()

    records = _load_records(Path(args.sweep_dir))
    if not records:
        raise SystemExit(f"No val_idx*.json found in {args.sweep_dir}")
    _add_combined_score(records)
    records.sort(key=lambda r: r["combined_z"], reverse=True)

    best_overall = records[0]
    best_by_mode = {}
    for record in records:
        best_by_mode.setdefault(record["prompt_mode"], record)

    header = (
        f"{'rank':>4} {'mode':<12} {'r':>3} {'lr':>7} {'ep':>3} "
        f"{'bertF1':>7} {'sbertC':>7} {'find':>6} {'combZ':>7}"
    )
    print(header)
    print("-" * len(header))
    for position, record in enumerate(records):
        metrics = record["metrics"]
        print(
            f"{position + 1:>4} {record['prompt_mode']:<12} "
            f"{record['lora_rank']:>3} {record['learning_rate']:>7.0e} "
            f"{record['num_epochs']:>3} "
            f"{metrics['bertscore_f1']:>7.4f} {metrics['sbert_calibrated']:>7.4f} "
            f"{metrics['finding_mentioned']:>6.3f} {record['combined_z']:>7.3f}"
        )

    print("\nBest overall:", best_overall["tag"])
    for mode, record in best_by_mode.items():
        print(f"Best [{mode}]:", record["tag"])

    summary = {
        "combined_metrics": list(COMBINED_METRICS),
        "ranked": records,
        "best_overall": best_overall,
        "best_by_prompt_mode": best_by_mode,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
