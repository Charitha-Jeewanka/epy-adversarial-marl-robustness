"""Atomic checkpointing and resume utilities (GEMINI.md §6)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from admarl.utils.seed import get_rng_states, set_rng_states

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: str | Path,
    filename: str,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    step: int,
    episode: int,
    config: dict[str, Any],
    lr_scheduler_state: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Save training checkpoint atomically using temporary file + os.replace.

    Guarantees an interrupted write never corrupts the existing checkpoint file.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_path = ckpt_dir / filename
    temp_path = ckpt_dir / f"{filename}.tmp"

    checkpoint_data: dict[str, Any] = {
        "step": step,
        "episode": episode,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "lr_scheduler_state": lr_scheduler_state,
        "rng_states": get_rng_states(),
        "config": config,
        "extra_state": extra_state or {},
    }

    # Save to temp file first
    torch.save(checkpoint_data, temp_path)

    # Atomic replace
    os.replace(temp_path, target_path)
    logger.info("Saved atomic checkpoint to %s at step %d (episode %d)", target_path, step, episode)
    return target_path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint file safely and return dictionary."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")

    logger.info("Loading checkpoint from %s", ckpt_path)
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt_data


def restore_rng_from_checkpoint(checkpoint_data: dict[str, Any]) -> None:
    """Restore RNG states from checkpoint dictionary."""
    if "rng_states" in checkpoint_data:
        set_rng_states(checkpoint_data["rng_states"])
        logger.info("Restored RNG states from checkpoint.")
    else:
        logger.warning("No 'rng_states' found in checkpoint data!")
