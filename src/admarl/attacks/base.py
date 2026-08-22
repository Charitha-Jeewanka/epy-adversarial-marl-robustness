"""Attack plugin interface. All attacks subclass BaseAttack (GEMINI.md §4)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseAttack(ABC):
    """Common interface for evaluation-time observation attacks.

    Implementations MUST respect the attack budget and the epsilon-ball;
    both are asserted in tests (GEMINI.md §9).
    """

    def __init__(self, budget_k: int = 0, epsilon: float = 0.0, norm: str = "linf") -> None:
        self.budget_k = budget_k
        self.epsilon = epsilon
        self.norm = norm
        self.perturbations_used = 0

    def reset_episode(self) -> None:
        """Reset per-episode perturbation counter."""
        self.perturbations_used = 0

    @abstractmethod
    def perturb(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None = None,
        critic: torch.nn.Module | None = None,
        step: int = 0,
    ) -> tuple[torch.Tensor, bool]:
        """Return (perturbed_obs, is_perturbed).

        Args:
            obs: (batch_size, num_agents, obs_dim) or (num_agents, obs_dim) observation tensor
            state: Optional centralized state tensor
            critic: Optional centralized critic neural network
            step: Timestep index

        Returns:
            perturbed_obs: Tensor of identical shape and dtype as obs, bounded by epsilon-ball
            is_perturbed: True if a budget-consuming perturbation was applied at this step
        """
        raise NotImplementedError
