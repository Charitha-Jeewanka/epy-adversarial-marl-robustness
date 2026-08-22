"""Attack factory for instantiating BaseAttack plugins via configuration (GEMINI.md §4)."""
from __future__ import annotations

from typing import Any

from admarl.attacks.base import BaseAttack
from admarl.attacks.critic_sensitivity import CriticSensitivityAttack
from admarl.attacks.no_attack import NoAttack


def get_attack(config: dict[str, Any]) -> BaseAttack:
    """Instantiate and return BaseAttack plugin based on config['attack']."""
    attack_cfg = config.get("attack", {})
    name = str(attack_cfg.get("name", "none")).lower()

    if name in ("none", "no_attack", "null", "false"):
        return NoAttack()
    elif name in ("critic_sensitivity", "sensitivity"):
        return CriticSensitivityAttack(
            budget_k=int(attack_cfg.get("budget_k", 5)),
            epsilon=float(attack_cfg.get("epsilon", 0.05)),
            norm=str(attack_cfg.get("norm", "linf")),
            sensitivity_threshold=float(attack_cfg.get("sensitivity_threshold", 0.0)),
        )
    else:
        raise ValueError(f"Unknown attack name: '{name}'. Supported: 'none', 'critic_sensitivity'")
