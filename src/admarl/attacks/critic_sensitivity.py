"""Budget-constrained centralized-critic sensitivity observation attack (GEMINI.md §4 & §7)."""
from __future__ import annotations

import torch

from admarl.attacks.base import BaseAttack


class CriticSensitivityAttack(BaseAttack):
    """Observation attack targeting centralized critic sensitivity with strict budget and epsilon-ball bounds."""

    def __init__(
        self,
        budget_k: int = 5,
        epsilon: float = 0.05,
        norm: str = "linf",
        sensitivity_threshold: float = 0.0,
    ) -> None:
        super().__init__(budget_k=budget_k, epsilon=epsilon, norm=norm)
        self.sensitivity_threshold = sensitivity_threshold

    def perturb(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None = None,
        critic: torch.nn.Module | None = None,
        step: int = 0,
    ) -> tuple[torch.Tensor, bool]:
        """Perturb observations to degrade state value if budget remains.

        Args:
            obs: Observation tensor of shape (batch_size, num_agents, obs_dim) or (num_agents, obs_dim)
            state: Centralized state tensor of shape (batch_size, state_dim) or (state_dim,)
            critic: Centralized critic model mapping state -> values
            step: Environment step index

        Returns:
            (perturbed_obs, is_perturbed): tuple of (tensor, bool)
        """
        # Strict budget enforcement: never perturb if budget is exhausted
        if self.budget_k <= 0 or self.perturbations_used >= self.budget_k:
            return obs.clone(), False

        # If critic or state is not provided, return unperturbed
        if critic is None:
            return obs.clone(), False

        device = obs.device
        is_batched = obs.dim() == 3

        if not is_batched:
            obs_tensor = obs.unsqueeze(0)
            state_tensor = state.unsqueeze(0) if state is not None else obs_tensor.reshape(1, -1)
        else:
            obs_tensor = obs
            state_tensor = state if state is not None else obs_tensor.reshape(obs_tensor.shape[0], -1)

        # Compute gradient with respect to state/observations
        state_var = state_tensor.clone().detach().to(device).requires_grad_(True)

        with torch.enable_grad():
            values = critic(state_var)
            critic_sum = values.sum()
            critic_sum.backward()

        grad = state_var.grad
        if grad is None:
            return obs.clone(), False

        if not torch.isfinite(grad).all():
            raise RuntimeError("Non-finite value (NaN/Inf) detected in critic gradients during attack!")

        # Calculate sensitivity score (gradient magnitude)
        sensitivity_score = float(grad.abs().sum().item())
        if sensitivity_score < self.sensitivity_threshold:
            return obs.clone(), False

        # Map state gradient back to observation space shape
        grad_obs = grad.reshape(obs_tensor.shape)

        # Generate bounded perturbation
        if self.norm.lower() == "linf":
            delta = -self.epsilon * torch.sign(grad_obs)
        elif self.norm.lower() == "l2":
            norm_val = torch.norm(grad_obs.reshape(grad_obs.shape[0], -1), p=2, dim=-1, keepdim=True)
            norm_val = norm_val.unsqueeze(-1) + 1e-8
            delta = -self.epsilon * (grad_obs / norm_val)
        else:
            raise ValueError(f"Unsupported norm: {self.norm}. Supported norms: 'linf', 'l2'")

        # Projection into strict epsilon-ball
        perturbed_obs_batched = torch.clamp(obs_tensor + delta, obs_tensor - self.epsilon, obs_tensor + self.epsilon)

        # Defensive numerical check (GEMINI.md §7)
        if not torch.isfinite(perturbed_obs_batched).all():
            raise RuntimeError("Non-finite value (NaN/Inf) detected in perturbed observations!")

        # Unbatch if input was single-sample
        perturbed_obs = perturbed_obs_batched.squeeze(0) if not is_batched else perturbed_obs_batched

        # Increment budget counter
        self.perturbations_used += 1

        return perturbed_obs.detach(), True
