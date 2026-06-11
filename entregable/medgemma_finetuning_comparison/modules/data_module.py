"""Carga de datos para MedGemma base vs LoRA."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DataModule:
    """Carga splits oficiales y anotaciones para descripcion clinica."""

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Seccion `data` del config global.
        """
        self.dataset_root = Path(config["dataset_root"]).resolve()
        self.split_file = Path(config["split_file"]).resolve()
        self.description_prompt = config["description_prompt"]
        self.rows = self._load_rows()

        logger.info("Splits cargados: %s", Counter(row["split"] for row in self.rows))
        logger.info("Labels cargados: %s", Counter(row["label"] for row in self.rows))

    def get_split(self, split_name: str, require_answer: bool = True) -> list[dict[str, Any]]:
        """Retorna filas de un split."""
        rows = [row for row in self.rows if row["split"] == split_name]
        if require_answer:
            rows = [row for row in rows if row["answer"]]
        return rows

    def get_train_rows(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Retorna filas de entrenamiento, opcionalmente limitadas para smoke tests."""
        rows = self.get_split("train")
        return rows if limit is None else rows[:limit]

    def get_eval_rows(self, split_name: str) -> list[dict[str, Any]]:
        """Retorna filas para evaluacion."""
        return self.get_split(split_name)

    def _load_rows(self) -> list[dict[str, Any]]:
        with self.split_file.open("r", encoding="utf-8") as file:
            split_data = json.load(file)

        rows: list[dict[str, Any]] = []
        for split_name in ("train", "validation", "test"):
            rows.extend(
                self._make_row(split_name=split_name, item=item)
                for item in split_data[split_name]
            )
        return rows

    def _read_annotation(self, annotation_path: str) -> dict[str, Any]:
        annotation_file = self.dataset_root / annotation_path
        with annotation_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data[0] if isinstance(data, list) else data

    def _make_row(self, split_name: str, item: dict[str, Any]) -> dict[str, Any]:
        ann = self._read_annotation(item["annotation"])
        conditions = ann.get("locs_data", {}).get("conditions", []) or []
        normalized_conditions = [condition.lower() for condition in conditions]

        return {
            "split": split_name,
            "image": str(self.dataset_root / item["image"]),
            "annotation": str(self.dataset_root / item["annotation"]),
            "label": ann.get("label"),
            "conditions": conditions,
            "target_label": "glaucoma"
            if "glaucoma" in normalized_conditions
            else "non_glaucoma",
            "prompt": self.description_prompt,
            "answer": ann.get("transcription", ""),
            "reference": ann.get("transcription", ""),
            "locs_data": ann.get("locs_data", {}),
        }
