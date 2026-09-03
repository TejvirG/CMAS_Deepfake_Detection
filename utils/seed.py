"""Reproducibility helpers."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) RNGs for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms where possible; fall back silently otherwise
    # since some ops (e.g. certain conv backwards) lack deterministic kernels.
    torch.backends.cudnn.benchmark = True


def get_device(auto_detect: bool = True) -> torch.device:
    if auto_detect and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
