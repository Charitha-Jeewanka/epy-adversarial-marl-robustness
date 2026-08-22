"""No-op attack implementation (BaseAttack plugin)."""
from __future__ import annotations

import torch

from admarl.attacks.base import BaseAttack


class NoAttack(BaseAttack):
    """Null attack plugin that returns unperturbed observations without consuming budget."""

    def __init__(self) -> None:
        super().__init__(budget_k=0, epsilon=0.0, norm="linf")

    def perturb(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None = None,
        critic: torch.nn.Module | None = None,
        actor: torch.nn.Module | None = None,
        step: int = 0,
    ) -> tuple[torch.Tensor, bool]:
        """Return unperturbed observations and False for is_perturbed."""
        return obs.clone(), False
