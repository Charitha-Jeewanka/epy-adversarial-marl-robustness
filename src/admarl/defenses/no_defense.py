"""No-op critic regularizer plugin implementation."""
from __future__ import annotations

import torch

from admarl.defenses.base import BaseCriticRegularizer


class NoRegularizer(BaseCriticRegularizer):
    """Null regularizer plugin returning a zero scalar loss without graph construction."""

    def __init__(self) -> None:
        super().__init__(penalty_coeff=0.0, norm="l2")

    def penalty(
        self,
        critic: torch.nn.Module,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Return zero scalar tensor on the device of states."""
        return torch.tensor(0.0, device=states.device)
