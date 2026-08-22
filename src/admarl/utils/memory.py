"""VRAM/RAM memory probe and CUDA OOM guard utilities (GEMINI.md §2 & §7)."""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_memory_info() -> dict[str, float]:
    """Return memory metrics (VRAM allocated/reserved in GB)."""
    info: dict[str, float] = {}
    if torch.cuda.is_available():
        info["vram_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        info["vram_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        info["max_vram_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    return info


def log_memory_status(prefix: str = "") -> None:
    """Log current VRAM/RAM allocation if CUDA is available."""
    info = get_memory_info()
    if info:
        msg = (
            f"{prefix}[Memory Probe] VRAM Allocated: {info.get('vram_allocated_gb', 0.0):.2f} GB | "
            f"Reserved: {info.get('vram_reserved_gb', 0.0):.2f} GB | "
            f"Max Allocated: {info.get('max_vram_allocated_gb', 0.0):.2f} GB"
        )
        logger.info(msg)


def handle_cuda_oom(e: RuntimeError, context: str = "") -> None:
    """Catch CUDA OOM, log current memory state, clear cache, and raise actionable error."""
    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
        if torch.cuda.is_available():
            info = get_memory_info()
            logger.error(
                "CUDA Out of Memory caught during %s. VRAM Allocated: %.2f GB, Reserved: %.2f GB",
                context,
                info.get("vram_allocated_gb", 0.0),
                info.get("vram_reserved_gb", 0.0),
            )
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA OOM in {context}. Reduce 'env.n_parallel_envs' or 'train.batch_size' in config. "
            f"Target VRAM ceiling is 6.0 GB (GEMINI.md §2)."
        ) from e
    raise e
