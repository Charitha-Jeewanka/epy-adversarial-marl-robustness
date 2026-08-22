"""Base interface for actor/policy-side robust training defenses (GEMINI.md §4)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseTrainingDefense(ABC):
    """Abstract interface for policy-side training defenses (e.g. SA-PPO adversarial training)."""

    def __init__(self, reg_coeff: float = 0.0, epsilon: float = 0.0) -> None:
        self.reg_coeff = reg_coeff
        self.epsilon = epsilon

    @abstractmethod
    def compute_robust_loss(
        self,
        actor: torch.nn.Module,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return (robustness_loss, metrics_dict)."""
        raise NotImplementedError


class NoActorDefense(BaseTrainingDefense):
    """Null policy defense returning zero scalar loss without inner PGD loop overhead."""

    def __init__(self) -> None:
        super().__init__(reg_coeff=0.0, epsilon=0.0)

    def compute_robust_loss(
        self,
        actor: torch.nn.Module,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return zero scalar loss on device of obs."""
        return torch.tensor(0.0, device=obs.device), {}
