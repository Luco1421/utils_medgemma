"""Project configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = {
    "data",
    "medgemma",
    "evaluation",
    "experimental_medgemma_lora",
}


def load_project_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the centralized YAML configuration."""

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a mapping")
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(
            f"Project config is missing sections: {sorted(missing)}"
        )
    if "seed" not in config:
        raise ValueError("Project config is missing the global seed")
    if config["data"].get("mask_target") != "optic_disc":
        raise ValueError("mask_target must be optic_disc")
    return config
