"""Validated adapter for the current private ophthalmology dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .conditioning import (
    CONDITIONS,
    MASK_CONDITIONS,
    build_prompt,
    make_overlay,
)

SPLIT_NAMES = ("train", "validation", "test")
MASK_TARGET = "optic_disc"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_label(annotation: Mapping[str, Any]) -> str:
    label = str(annotation.get("label") or "").casefold()
    conditions = {
        str(value).casefold()
        for value in (annotation.get("locs_data") or {}).get("conditions", [])
    }
    if label == "pathological" and "glaucoma" in conditions:
        return "glaucoma"
    if label == "normal" and not conditions:
        return "normal"
    raise ValueError(
        f"Unsupported label/condition combination for "
        f"{annotation.get('image_filename')}: "
        f"label={annotation.get('label')!r}, conditions={sorted(conditions)!r}"
    )


def _validate_image_and_disc_mask(
    dataset_root: Path,
    annotation: Mapping[str, Any],
    target_label: str,
) -> tuple[Path, Path | None]:
    image_filename = str(annotation["image_filename"])
    image_path = dataset_root / image_filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Dataset image does not exist: {image_path}")

    mask_path = dataset_root / f"{Path(image_filename).stem}_disc.png"
    if target_label == "normal":
        if mask_path.exists():
            raise ValueError(
                f"Normal image unexpectedly has an optic-disc GT mask: {mask_path}"
            )
        return image_path, None
    if not mask_path.is_file():
        raise FileNotFoundError(
            f"Glaucoma image has no optic-disc GT mask: {mask_path}"
        )

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch for {image_filename}: "
                f"{image.size} vs {mask.size}"
            )
    return image_path, mask_path


def _normalize_row(
    annotation: Mapping[str, Any],
    *,
    split_name: str,
    dataset_root: Path,
) -> dict[str, Any]:
    if "image_filename" not in annotation:
        raise ValueError(
            "The current split schema requires embedded annotation objects "
            "with an 'image_filename' field"
        )
    target_label = _target_label(annotation)
    image_path, mask_path = _validate_image_and_disc_mask(
        dataset_root,
        annotation,
        target_label,
    )
    transcription = str(annotation.get("transcription") or "").strip()
    if not transcription:
        raise ValueError(
            f"Missing expert transcription for {annotation['image_filename']}"
        )
    return {
        "split": split_name,
        "annotation_id": annotation.get("id"),
        "image_id": Path(str(annotation["image_filename"])).stem,
        "image": str(image_path),
        "ground_truth_mask": str(mask_path) if mask_path else None,
        "has_ground_truth_mask": mask_path is not None,
        "segmentation_applicable": target_label == "glaucoma",
        "mask_target": MASK_TARGET,
        "label": annotation.get("label"),
        "target_label": target_label,
        "transcription": transcription,
        "reference": transcription,
        "locs_data": annotation.get("locs_data") or {},
    }


def load_split_rows(
    split_file: str | Path = "dataset/split.json",
    *,
    dataset_root: str | Path | None = None,
    annotations_file: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load and validate the immutable train/validation/test split.

    The current dataset is flat: images, optic-disc masks, ``annotations.json``
    and ``split.json`` live in the same directory. Normal images intentionally
    have no segmentation mask because they bypass SAM in the target pipeline.
    """

    split_path = Path(split_file)
    root = Path(dataset_root) if dataset_root is not None else split_path.parent
    annotations_path = (
        Path(annotations_file)
        if annotations_file is not None
        else root / "annotations.json"
    )
    split_data = _read_json(split_path)
    if not isinstance(split_data, dict):
        raise ValueError("split.json must contain an object keyed by split name")
    missing_splits = set(SPLIT_NAMES) - set(split_data)
    if missing_splits:
        raise ValueError(
            f"Split file is missing required splits: {sorted(missing_splits)}"
        )

    canonical_annotations = _read_json(annotations_path)
    if not isinstance(canonical_annotations, list):
        raise ValueError("annotations.json must contain a JSON array")
    annotations_by_filename = {
        str(annotation["image_filename"]): annotation
        for annotation in canonical_annotations
    }
    if len(annotations_by_filename) != len(canonical_annotations):
        raise ValueError("annotations.json contains duplicate image filenames")

    rows: dict[str, list[dict[str, Any]]] = {}
    seen_filenames: set[str] = set()
    for split_name in SPLIT_NAMES:
        split_rows = split_data[split_name]
        if not isinstance(split_rows, list):
            raise ValueError(f"Split {split_name!r} must contain a list")
        rows[split_name] = []
        for annotation in split_rows:
            image_filename = str(annotation.get("image_filename") or "")
            if not image_filename:
                raise ValueError(
                    f"Split {split_name!r} contains a row without image_filename"
                )
            if image_filename in seen_filenames:
                raise ValueError(
                    f"Image appears in more than one split: {image_filename}"
                )
            seen_filenames.add(image_filename)
            canonical = annotations_by_filename.get(image_filename)
            if canonical is None:
                raise ValueError(
                    f"Split references an image absent from annotations.json: "
                    f"{image_filename}"
                )
            if annotation != canonical:
                raise ValueError(
                    f"Embedded split annotation differs from annotations.json: "
                    f"{image_filename}"
                )
            rows[split_name].append(
                _normalize_row(
                    annotation,
                    split_name=split_name,
                    dataset_root=root,
                )
            )

    missing_from_split = set(annotations_by_filename) - seen_filenames
    if missing_from_split:
        raise ValueError(
            "annotations.json contains images absent from split.json: "
            f"{sorted(missing_from_split)}"
        )
    return rows


def summarize_splits(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, int]]:
    """Return class and optic-disc-mask counts for each fixed split."""

    return {
        split_name: {
            "total": len(split_rows),
            "glaucoma": sum(
                row["target_label"] == "glaucoma" for row in split_rows
            ),
            "normal": sum(
                row["target_label"] == "normal" for row in split_rows
            ),
            "with_optic_disc_mask": sum(
                bool(row["has_ground_truth_mask"]) for row in split_rows
            ),
        }
        for split_name, split_rows in rows.items()
    }


def validate_optic_disc_mask_pixels(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> int:
    """Validate that every contractual GT mask is non-empty and binary."""

    validated = 0
    for split_rows in rows.values():
        for row in split_rows:
            mask_path = row.get("ground_truth_mask")
            if mask_path is None:
                continue
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert("L"))
            if not np.any(mask):
                raise ValueError(f"Optic-disc mask is empty: {mask_path}")
            if np.any((mask != 0) & (mask != 255)):
                raise ValueError(
                    f"Optic-disc mask is not binary 0/255: {mask_path}"
                )
            validated += 1
    return validated


def _sft_example(image: Image.Image, prompt: str, transcription: str) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": transcription}],
            },
        ]
    }


def build_sft_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt: str = "Describe the ophthalmological findings in this fundus image.",
) -> list[dict[str, Any]]:
    """Build generic image-to-expert-description examples (condition-A prompt).

    Every image is paired with the same baseline prompt, so the adapter learns
    the expert writing style but never how to exploit the conditioning hints.
    """

    examples = []
    for row in rows:
        transcription = str(row.get("transcription") or "").strip()
        if not transcription:
            continue
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        examples.append(_sft_example(image, prompt, transcription))
    return examples


def build_conditioned_sft_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    confidence: float = 0.80,
) -> list[dict[str, Any]]:
    """Build SFT examples mirroring the six inference conditions.

    For each image one example is created per applicable condition, reusing the
    exact prompts and red overlay from :mod:`conditioning`. The adapter therefore
    learns to use the mask, predicted class and distribution, matching how it is
    evaluated. Oracle prediction/distribution use the dataset label, consistent
    with :func:`oracle_inputs.build_dataset_oracle_inputs`.
    """

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    examples = []
    for row in rows:
        transcription = str(row.get("transcription") or "").strip()
        if not transcription:
            continue
        prediction = str(row["target_label"])
        other = "normal" if prediction == "glaucoma" else "glaucoma"
        distribution = {prediction: confidence, other: 1.0 - confidence}
        mask = row.get("ground_truth_mask")
        with Image.open(row["image"]) as source:
            base_image = source.convert("RGB")
        overlay_image = make_overlay(base_image, mask) if mask else None

        for condition in CONDITIONS:
            if condition in MASK_CONDITIONS:
                if not mask:
                    continue
                image = overlay_image
            else:
                image = base_image
            prompt = build_prompt(
                condition,
                prediction=prediction if condition in {"C1", "D1"} else None,
                distribution=distribution if condition in {"C2", "D2"} else None,
            )
            examples.append(_sft_example(image, prompt, transcription))
    return examples
