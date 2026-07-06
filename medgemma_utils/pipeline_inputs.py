"""Load conditioning inputs from a pipeline-style folder (classifier + segmentor).

This mirrors what the upstream modules produce: for each fundus image there is
the raw image, the segmentor's object mask (``<id>_obj_0.png``) and the
pre-rendered red overlay delivered by the segmentation module
(``overlays_rojos/Imagenes_overlay/<id>_overlay.jpg``, the disc mask composited
at 50% red). We consume that overlay directly rather than re-render it: the
overlay is an artifact of the upstream module, and the oracle source composites
its own overlay from the expert mask at the same blend, so the two sources
differ only in the provenance of the mask.

The conditioner consumes the same ``ConditioningInput`` contract as the oracle
provider, so M7/M8 are identical regardless of where the data comes from. The
only pipeline-specific piece is the classifier label. Until the real classifier
output is wired in, ground-truth labels from ``annotations.json`` are used as a
stand-in (pass ``predictions`` to override per image).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .dataset import _target_label
from .inputs import ConditioningInput


def load_pipeline_dir_inputs(
    image_dir: str | Path,
    *,
    annotations_file: str | Path | None = None,
    predictions: Mapping[str, str] | None = None,
    ground_truth_labels: Mapping[str, str] | None = None,
    dataset_root: str | Path | None = None,
    overlay_subdir: str = "overlays_rojos/Imagenes_overlay",
    overlay_suffix: str = "_overlay",
    overlay_ext: str = ".jpg",
    mask_suffix: str = "_obj_0",
    mask_ext: str = ".png",
) -> list[ConditioningInput]:
    """Build inputs from a pipeline folder such as ``dataset/test``.

    ``predictions`` maps ``image_id`` to ``"glaucoma"``/``"normal"`` and, when
    given, replaces the ground-truth stand-in. ``ground_truth_labels`` (same
    mapping shape) sets ``expected_finding`` from the pipeline artifact for
    traceability; when omitted it falls back to the label in
    ``annotations.json``. The reference transcription always comes from
    ``annotations.json`` (the pipeline metrics carry no text). ``dataset_root``
    (the flat dataset directory) is only needed to locate GT disc masks for
    segmentation scoring; it is optional.
    """

    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Pipeline image directory not found: {image_dir}")

    annotations_by_id: dict[str, dict] = {}
    if annotations_file is not None:
        records = json.loads(Path(annotations_file).read_text(encoding="utf-8"))
        annotations_by_id = {
            Path(str(record["image_filename"])).stem: record
            for record in records
        }

    overlay_dir = image_dir / overlay_subdir
    root = Path(dataset_root) if dataset_root is not None else None

    inputs: list[ConditioningInput] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        image_id = image_path.stem

        annotation = annotations_by_id.get(image_id)
        if ground_truth_labels is not None and image_id in ground_truth_labels:
            ground_truth_label = str(ground_truth_labels[image_id]).casefold()
            if ground_truth_label not in {"glaucoma", "normal"}:
                raise ValueError(
                    f"Unsupported ground-truth label for {image_id}: "
                    f"{ground_truth_label!r}"
                )
        elif annotation is not None:
            ground_truth_label = _target_label(annotation)
        else:
            ground_truth_label = None

        if predictions is not None and image_id in predictions:
            prediction = str(predictions[image_id]).casefold()
        elif ground_truth_label is not None:
            prediction = ground_truth_label
        else:
            raise ValueError(
                f"No prediction available for {image_id}: provide 'predictions' "
                "or 'annotations_file'"
            )
        if prediction not in {"glaucoma", "normal"}:
            raise ValueError(
                f"Unsupported prediction for {image_id}: {prediction!r}"
            )

        overlay_path = overlay_dir / f"{image_id}{overlay_suffix}{overlay_ext}"
        overlay_image = str(overlay_path) if overlay_path.is_file() else None

        mask_path = image_dir / f"{image_id}{mask_suffix}{mask_ext}"
        mask = str(mask_path) if mask_path.is_file() else None

        ground_truth_mask = None
        if root is not None:
            disc_path = root / f"{image_id}_disc.png"
            ground_truth_mask = str(disc_path) if disc_path.is_file() else None

        reference = ""
        if annotation is not None:
            reference = str(annotation.get("transcription") or "").strip()

        inputs.append(
            ConditioningInput(
                image_id=image_id,
                image=str(image_path),
                overlay_image=overlay_image,
                mask=mask,
                prediction=prediction,
                reference=reference,
                mask_source="pipeline" if mask else "none",
                source_data="pipeline",
                expected_finding=ground_truth_label,
                ground_truth_mask=ground_truth_mask,
            )
        )
    return inputs
