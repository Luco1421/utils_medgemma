"""Dataset adapter used for MedGemma SFT and mock experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_annotation(dataset_root: str | Path, annotation_path: str) -> dict[str, Any]:
    data = json.loads(
        (Path(dataset_root) / annotation_path).read_text(encoding="utf-8")
    )
    return data[0] if isinstance(data, list) else data


def load_split_rows(
    split_file: str | Path,
    *,
    dataset_root: str | Path = "dataset",
) -> dict[str, list[dict[str, Any]]]:
    dataset_root = Path(dataset_root)
    split_data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    rows: dict[str, list[dict[str, Any]]] = {}

    for split_name in ("train", "validation", "test"):
        rows[split_name] = []
        for item in split_data[split_name]:
            annotation = read_annotation(dataset_root, item["annotation"])
            conditions = annotation.get("locs_data", {}).get("conditions", []) or []
            rows[split_name].append({
                "split": split_name,
                "image_id": Path(item["image"]).stem,
                "image": str(dataset_root / item["image"]),
                "annotation": str(dataset_root / item["annotation"]),
                "mask_cup_npy": str(dataset_root / item["mask_cup_npy"]),
                "mask_disc_npy": str(dataset_root / item["mask_disc_npy"]),
                "label": annotation.get("label"),
                "conditions": conditions,
                "target_label": (
                    "glaucoma"
                    if any(str(value).lower() == "glaucoma" for value in conditions)
                    else "non_glaucoma"
                ),
                "transcription": annotation.get("transcription", ""),
                "reference": annotation.get("transcription", ""),
            })
    return rows


def build_sft_examples(
    rows: list[dict[str, Any]],
    *,
    prompt: str = "Describe the ophthalmological findings in this fundus image.",
) -> list[dict[str, Any]]:
    from PIL import Image

    examples = []
    for row in rows:
        if not row["transcription"]:
            continue
        examples.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": Image.open(row["image"]).convert("RGB"),
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": row["transcription"]}],
                },
            ]
        })
    return examples
