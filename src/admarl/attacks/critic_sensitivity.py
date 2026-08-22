"""Budget-constrained centralized-critic sensitivity observation attack (GEMINI.md §4 & §7)."""
from __future__ import annotations

import torch

from admarl.attacks.base import BaseAttack
from admarl.utils.pgd import pgd_step, project_epsilon_ball


class CriticSensitivityAttack(BaseAttack):
    """Observation attack targeting centralized critic sensitivity with multi-step PGD and targeted worst-case actions."""

    def __init__(
        self,
        budget_k: int = 5,
        epsilon: float = 0.05,
        norm: str = "linf",
        sensitivity_threshold: float = 0.0,
        pgd_steps: int = 5,
        step_size: float | None = None,
    ) -> None:
        super().__init__(budget_k=budget_k, epsilon=epsilon, norm=norm)
        self.sensitivity_threshold = sensitivity_threshold
        self.pgd_steps = pgd_steps
        self.step_size = step_size
        self.worst_case_hit_count = 0

    def reset_episode(self) -> None:
        """Reset budget and step tracking at episode start."""
        super().reset_episode()
        self.action_changed_count = getattr(self, "action_changed_count", 0)
        self.total_attacked_count = getattr(self, "total_attacked_count", 0)
        self.worst_case_hit_count = getattr(self, "worst_case_hit_count", 0)

    def perturb(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None = None,
        critic: torch.nn.Module | None = None,
        actor: torch.nn.Module | None = None,
        step: int = 0,
    ) -> tuple[torch.Tensor, bool]:
        """Perturb observations using targeted multi-step PGD in the epsilon-ball.

        Args:
            obs: Observation tensor of shape (batch_size, num_agents, obs_dim) or (num_agents, obs_dim)
            state: Centralized state tensor of shape (batch_size, state_dim) or (state_dim,)
            critic: Centralized critic model (optional sensitivity check)
            actor: Actor model mapping obs -> action distribution
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

        if actor is None:
            return obs.clone(), False

        # Identify worst-case target action per agent (argmin logit under clean observation)
        with torch.no_grad():
            clean_dist = actor(obs_tensor)
            clean_actions = torch.argmax(clean_dist.logits, dim=-1)
            worst_target_actions = torch.argmin(clean_dist.logits, dim=-1)

        # Multi-step targeted PGD in the epsilon-ball
        perturbed_obs_batched = obs_tensor.clone().detach()
        alpha = self.step_size if self.step_size is not None else (self.epsilon / max(1, self.pgd_steps / 2.5))

        for _ in range(self.pgd_steps):
            obs_var = perturbed_obs_batched.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                dist = actor(obs_var)
                # Targeted loss: maximize probability of worst-case target actions
                loss = dist.log_prob(worst_target_actions).sum()
                loss.backward()

            grad_obs = obs_var.grad
            if grad_obs is None:
                break

            if not torch.isfinite(grad_obs).all():
                raise RuntimeError("Non-finite value (NaN/Inf) detected in gradients during PGD attack!")

            # PGD step with maximize=True (increase probability of worst-case action)
            step_obs = pgd_step(perturbed_obs_batched, grad_obs, alpha, norm=self.norm, maximize=True)
            perturbed_obs_batched = project_epsilon_ball(obs_tensor, step_obs, self.epsilon, norm=self.norm)

        # Defensive numerical check (GEMINI.md §7)
        if not torch.isfinite(perturbed_obs_batched).all():
            raise RuntimeError("Non-finite value (NaN/Inf) detected in perturbed observations!")

        # Unbatch if input was single-sample
        perturbed_obs = perturbed_obs_batched.squeeze(0) if not is_batched else perturbed_obs_batched

        # Increment budget counter and track action changes & worst-case hits
        self.perturbations_used += 1
        if hasattr(self, "total_attacked_count"):
            self.total_attacked_count += len(clean_actions.view(-1))
            with torch.no_grad():
                adv_a = torch.argmax(actor(perturbed_obs_batched).logits, dim=-1)
                num_flips = int((clean_actions != adv_a).sum().item())
                num_worst_hits = int((adv_a == worst_target_actions).sum().item())

                self.action_changed_count += num_flips
                self.worst_case_hit_count += num_worst_hits

        return perturbed_obs.detach(), True
