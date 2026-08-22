"""Defense factory for instantiating regularizer and training defense plugins via configuration (GEMINI.md §4)."""
from __future__ import annotations

from typing import Any

from admarl.defenses.base import BaseCriticRegularizer
from admarl.defenses.grad_penalty import GradientPenaltyRegularizer
from admarl.defenses.no_defense import NoRegularizer
from admarl.defenses.sa_ppo import SAPPOAdversarialDefense
from admarl.defenses.training_defense import BaseTrainingDefense, NoActorDefense


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


def get_training_defense(config: dict[str, Any]) -> BaseTrainingDefense:
    """Instantiate and return BaseTrainingDefense plugin based on config['adv_training']."""
    adv_cfg = config.get("adv_training", {})
    enabled = bool(adv_cfg.get("enabled", False))
    coeff = float(adv_cfg.get("reg_coeff", 0.0))
    pgd_steps = int(adv_cfg.get("pgd_steps", 0))

    if not enabled or coeff <= 0.0 or pgd_steps <= 0:
        return NoActorDefense()

    return SAPPOAdversarialDefense(
        epsilon=float(adv_cfg.get("epsilon", 0.05)),
        pgd_steps=pgd_steps,
        pgd_step_size=float(adv_cfg.get("pgd_step_size", 0.01)),
        reg_coeff=coeff,
        norm=str(adv_cfg.get("norm", "linf")),
    )
