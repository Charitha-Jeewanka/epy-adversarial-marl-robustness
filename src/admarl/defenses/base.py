"""Defense/regularizer plugin interface (GEMINI.md §4)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseCriticRegularizer(ABC):
    """Common interface for critic regularizers (e.g. gradient penalty).

    Must return a finite scalar tensor; non-finite values raise (GEMINI.md §7).
    """

    def __init__(self, penalty_coeff: float = 0.0, norm: str = "l2") -> None:
        self.penalty_coeff = penalty_coeff
        self.norm = norm

    @abstractmethod
    def penalty(
        self,
        critic: torch.nn.Module,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scalar regularization term to add to the critic loss."""
        raise NotImplementedError
