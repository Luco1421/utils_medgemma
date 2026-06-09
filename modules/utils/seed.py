"""Utilidades de reproducibilidad."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Inicializa las semillas disponibles de forma determinista."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
