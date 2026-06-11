"""Utilidades compartidas del experimento."""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Carga un archivo YAML de configuracion."""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML no esta instalado. Instala las dependencias con "
            "`pip install -r requirements.txt` desde la carpeta del experimento."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logging(level: int = logging.INFO) -> None:
    """Configura logging estandar para scripts CLI."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_global_seed(seed: int) -> None:
    """Fija semillas para reproducibilidad."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def utc_timestamp() -> str:
    """Retorna timestamp UTC ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    """Crea un directorio si no existe y retorna su Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Guarda un diccionario como JSON indentado."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_hf_token() -> str | None:
    """Obtiene el token de Hugging Face desde variables de entorno."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def torch_dtype_from_config(dtype_name: str, device: str) -> Any:
    """Resuelve dtype de entrenamiento/inferencia desde config."""
    import torch

    if dtype_name == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"dtype no soportado: {dtype_name}")


def device_index(device: str) -> int:
    """Extrae el indice CUDA desde strings como cuda o cuda:0."""
    if not device.startswith("cuda"):
        raise ValueError(f"QLoRA requiere dispositivo CUDA, recibido: {device}")
    if ":" not in device:
        return 0
    return int(device.split(":", maxsplit=1)[1])
