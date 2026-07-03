"""Export a blinded Likert 1-5 review sheet from an M8 result JSON."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--key-json", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = list(payload["results"])
    random.Random(args.seed).shuffle(rows)

    review_path = Path(args.review_csv)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    key = []
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_id",
                "image_id",
                "reference_text",
                "generated_text",
                "likert_score_1_to_5",
                "reviewer_notes",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            review_id = f"R{index:04d}"
            writer.writerow({
                "review_id": review_id,
                "image_id": row["image_id"],
                "reference_text": (
                    row.get("reference_text") or row.get("reference")
                ),
                "generated_text": (
                    row.get("generated_text")
                    or row.get("generated")
                    or row.get("text")
                ),
                "likert_score_1_to_5": "",
                "reviewer_notes": "",
            })
            key.append({
                "review_id": review_id,
                "image_id": row["image_id"],
                "source_data": row.get("source_data"),
                "model_variant": row.get("model_variant"),
                "condition": row["condition"],
            })

    key_path = Path(args.key_json)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
