"""Normalized input contract consumed by the MedGemma experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConditioningInput:
    image_id: str
    image: Any
    mask: Any
    prediction: str
    distribution: dict[str, float]
    reference: str
    mask_source: str
    input_source: str


def load_json_inputs(path: str | Path) -> list[ConditioningInput]:
    """Load future real pipeline outputs using one explicit JSON schema."""

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ConditioningInput(**record) for record in records]
