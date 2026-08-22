"""Budget-constrained centralized-critic sensitivity observation attack (GEMINI.md §4 & §7)."""
from __future__ import annotations

import torch

from admarl.attacks.base import BaseAttack
from admarl.utils.pgd import pgd_step, project_epsilon_ball


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

    def reset_episode(self) -> None:
        """Reset budget and step tracking at episode start."""
        super().reset_episode()
        self.action_changed_count = getattr(self, "action_changed_count", 0)
        self.total_attacked_count = getattr(self, "total_attacked_count", 0)

    def perturb(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None = None,
        critic: torch.nn.Module | None = None,
        actor: torch.nn.Module | None = None,
        step: int = 0,
    ) -> tuple[torch.Tensor, bool]:
        """Perturb observations to force actor action changes if budget remains.

        Args:
            obs: Observation tensor of shape (batch_size, num_agents, obs_dim) or (num_agents, obs_dim)
            state: Centralized state tensor of shape (batch_size, state_dim) or (state_dim,)
            critic: Centralized critic model mapping state -> values (used for sensitivity check)
            actor: Actor model mapping obs -> action distribution (used for perturbation crafting)
            step: Environment step index

        Returns:
            (perturbed_obs, is_perturbed): tuple of (tensor, bool)
        """
        # Strict budget enforcement: never perturb if budget is exhausted
        if self.budget_k <= 0 or self.perturbations_used >= self.budget_k:
            return obs.clone(), False

        device = obs.device
        is_batched = obs.dim() == 3

        if not is_batched:
            obs_tensor = obs.unsqueeze(0)
            state_tensor = state.unsqueeze(0) if state is not None else obs_tensor.reshape(1, -1)
        else:
            obs_tensor = obs
            state_tensor = state if state is not None else obs_tensor.reshape(obs_tensor.shape[0], -1)

        # Optional critic sensitivity check
        if critic is not None:
            state_var = state_tensor.clone().detach().to(device).requires_grad_(True)
            with torch.enable_grad():
                values = critic(state_var)
                critic_sum = values.sum()
                critic_sum.backward()

            grad_critic = state_var.grad
            if grad_critic is not None:
                sensitivity_score = float(grad_critic.abs().sum().item())
                if sensitivity_score < self.sensitivity_threshold:
                    return obs.clone(), False

        # Craft perturbation targeting actor action probabilities
        obs_var = obs_tensor.clone().detach().to(device).requires_grad_(True)

        if actor is not None:
            with torch.enable_grad():
                dist = actor(obs_var)
                clean_actions = torch.argmax(dist.logits, dim=-1)
                # Loss minimizes log probability of clean argmax actions
                loss = -dist.log_prob(clean_actions).sum()
                loss.backward()
            grad_obs = obs_var.grad
        else:
            # Fallback to critic gradient if actor is not provided
            if critic is None:
                return obs.clone(), False
            state_var = state_tensor.clone().detach().to(device).requires_grad_(True)
            with torch.enable_grad():
                values = critic(state_var)
                values.sum().backward()
            grad_state = state_var.grad
            grad_obs = grad_state.reshape(obs_tensor.shape) if grad_state is not None else None

        if grad_obs is None:
            return obs.clone(), False

        if not torch.isfinite(grad_obs).all():
            raise RuntimeError("Non-finite value (NaN/Inf) detected in gradients during attack!")

        # Generate bounded perturbation using shared PGD primitive (maximize=True for loss)
        unprojected_obs = pgd_step(obs_tensor, grad_obs, self.epsilon, norm=self.norm, maximize=True)

        # Projection into strict epsilon-ball using shared primitive
        perturbed_obs_batched = project_epsilon_ball(obs_tensor, unprojected_obs, self.epsilon, norm=self.norm)

        # Defensive numerical check (GEMINI.md §7)
        if not torch.isfinite(perturbed_obs_batched).all():
            raise RuntimeError("Non-finite value (NaN/Inf) detected in perturbed observations!")

        # Unbatch if input was single-sample
        perturbed_obs = perturbed_obs_batched.squeeze(0) if not is_batched else perturbed_obs_batched

        # Increment budget counter and track action changes
        self.perturbations_used += 1
        if hasattr(self, "total_attacked_count"):
            self.total_attacked_count += 1
            if actor is not None:
                with torch.no_grad():
                    clean_a = torch.argmax(actor(obs_tensor).logits, dim=-1)
                    adv_a = torch.argmax(actor(perturbed_obs_batched).logits, dim=-1)
                    if not torch.equal(clean_a, adv_a):
                        self.action_changed_count += 1

        return perturbed_obs.detach(), True
