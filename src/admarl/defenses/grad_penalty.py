"""Gradient penalty / Lipschitz continuity regularizer for centralized critic (GEMINI.md §4 & §7)."""
from __future__ import annotations

import torch

from admarl.defenses.base import BaseCriticRegularizer


class GradientPenaltyRegularizer(BaseCriticRegularizer):
    """Penalizes the norm of the critic gradient w.r.t. global state inputs to enforce Lipschitz continuity."""

    def __init__(self, penalty_coeff: float = 0.1, norm: str = "l2") -> None:
        super().__init__(penalty_coeff=penalty_coeff, norm=norm)

    def penalty(
        self,
        critic: torch.nn.Module,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient-penalty scalar regularization term.

        Args:
            critic: Centralized critic model
            states: (batch_size, state_dim) state tensor

        Returns:
            penalty: Scalar tensor with second-order autograd graph (create_graph=True)
        """
        if self.penalty_coeff <= 0.0:
            return torch.tensor(0.0, device=states.device)

        device = states.device
        states_var = states.clone().detach().to(device).requires_grad_(True)

        with torch.enable_grad():
            values = critic(states_var)
            critic_sum = values.sum()

            grads = torch.autograd.grad(
                outputs=critic_sum,
                inputs=states_var,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        if grads is None:
            return torch.tensor(0.0, device=device)

        if self.norm.lower() == "l2":
            # L2 squared norm of gradient: ||grad||^2
            penalty_per_sample = grads.pow(2).sum(dim=-1)
        elif self.norm.lower() == "l1":
            # L1 norm of gradient: ||grad||_1
            penalty_per_sample = grads.abs().sum(dim=-1)
        else:
            raise ValueError(f"Unsupported norm: '{self.norm}'. Supported norms: 'l2', 'l1'")

        penalty_loss = penalty_per_sample.mean()
        total_penalty = self.penalty_coeff * penalty_loss

        # Defensive numerical check (GEMINI.md §7)
        if not torch.isfinite(total_penalty):
            raise RuntimeError("Non-finite regularizer penalty (NaN/Inf) encountered during critic update!")

        return total_penalty
