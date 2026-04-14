"""Small training and evaluation helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from sklearn.metrics import f1_score


def set_random_seed(seed: int) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch for reproducible runs.

    Args:
        seed: Integer seed shared across ``random``, ``numpy``, and ``torch``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def get_device() -> torch.device:
    """Return ``torch.device`` for CUDA if a GPU is available, otherwise CPU.
    Returns:
        ``cuda`` when ``torch.cuda.is_available()`` is true, else ``cpu``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_macro_f1(y_true, y_pred) -> float:
    """Compute macro-averaged F1 (unweighted mean of per-class F1 scores).

    Args:
        y_true: Ground-truth class labels (array-like).
        y_pred: Predicted class labels (array-like).

    Returns:
        Macro F1 score as a float.
    """
    return float(f1_score(y_true, y_pred, average="macro"))


def save_checkpoint(path: Union[str, Path], state: Dict[str, Any]) -> None:
    """Persist a checkpoint dict with ``torch.save`` (e.g. model and optimizer state).

    Args:
        path: Destination path (``.pt`` / ``.pth``). Parent directories are created if needed.
        state: Mapping to save, typically including ``model_state_dict``,
            ``optimizer_state_dict``, and ``epoch``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
