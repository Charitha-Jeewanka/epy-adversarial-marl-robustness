"""Defense factory for instantiating BaseCriticRegularizer plugins via configuration (GEMINI.md §4)."""
from __future__ import annotations

from typing import Any

from admarl.defenses.base import BaseCriticRegularizer
from admarl.defenses.grad_penalty import GradientPenaltyRegularizer
from admarl.defenses.no_defense import NoRegularizer


def get_regularizer(config: dict[str, Any]) -> BaseCriticRegularizer:
    """Instantiate and return BaseCriticRegularizer plugin based on config['defense']."""
    defense_cfg = config.get("defense", {})
    name = str(defense_cfg.get("name", "none")).lower()
    coeff = float(defense_cfg.get("penalty_coeff", 0.0))

    if name in ("none", "no_defense", "null", "false") or coeff <= 0.0:
        return NoRegularizer()
    elif name in ("grad_penalty", "gradient_penalty", "lipschitz"):
        return GradientPenaltyRegularizer(
            penalty_coeff=coeff,
            norm=str(defense_cfg.get("norm", "l2")),
        )
    else:
        raise ValueError(f"Unknown defense name: '{name}'. Supported: 'none', 'grad_penalty'")
