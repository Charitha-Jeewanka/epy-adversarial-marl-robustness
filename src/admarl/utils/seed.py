"""Seed and RNG state utilities for reproducibility (GEMINI.md §6 & §8)."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds across Python, NumPy, PyTorch CPU and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_states() -> dict[str, Any]:
    """Capture current state of all random number generators for bit-for-bit resume."""
    states: dict[str, Any] = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        states["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict[str, Any]) -> None:
    """Restore state of all random number generators from a checkpoint dictionary."""
    if "python_rng" in states and states["python_rng"] is not None:
        random.setstate(states["python_rng"])
    if "numpy_rng" in states and states["numpy_rng"] is not None:
        np.random.set_state(states["numpy_rng"])
    if "torch_cpu_rng" in states and states["torch_cpu_rng"] is not None:
        torch.set_rng_state(states["torch_cpu_rng"])
    if "torch_cuda_rng" in states and states["torch_cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda_rng"])
