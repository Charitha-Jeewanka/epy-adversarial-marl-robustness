"""Attack plugin interface. All attacks subclass BaseAttack (GEMINI.md §4)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseAttack(ABC):
    """Common interface for evaluation-time observation attacks.

    Implementations MUST respect the attack budget and the epsilon-ball;
    both are asserted in tests (GEMINI.md §9).
    """

    @abstractmethod
    def perturb(
        self,
        observations: torch.Tensor,
        step: int,
        critic: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Return perturbed observations. Must not exceed the configured budget."""
        raise NotImplementedError
