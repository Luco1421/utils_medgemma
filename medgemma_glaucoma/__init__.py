# -*- coding: utf-8 -*-
"""MedGemma para glaucoma: 2 modos (classifier / descriptor), modelo base y LoRA."""
from . import config
from .config import (MODEL_ID, CLASS_LABELS, CLASSIFIER_Q, DESCRIPTOR_Q,
                     default_dataset_dir)
from .data import load_records, train_test_split, build_hf_dataset
from .model import load_model, apply_lora, ask_image, DTYPE
from .evaluate import evaluate
from .train import make_collate_fn, make_trainer

__all__ = [
    "config", "MODEL_ID", "CLASS_LABELS", "CLASSIFIER_Q", "DESCRIPTOR_Q",
    "default_dataset_dir", "load_records", "train_test_split", "build_hf_dataset",
    "load_model", "apply_lora", "ask_image", "DTYPE", "evaluate",
    "make_collate_fn", "make_trainer",
]
