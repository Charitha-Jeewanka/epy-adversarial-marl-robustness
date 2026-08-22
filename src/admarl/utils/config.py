"""Config loading and validation utilities (GEMINI.md §8)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required sections and fields in config."""
    required_sections = ["seed", "hardware", "env", "train"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config.")

    # Memory guard check
    max_vram = config["hardware"].get("max_vram_gb", 5.0)
    if max_vram > 6.0:
        raise ValueError(
            f"Config max_vram_gb={max_vram} exceeds hardware ceiling 6.0 GB (GEMINI.md §2)."
        )
