"""Continuous adversarial training defense plugin (SA-PPO-style) (GEMINI.md §4 & §7)."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from admarl.defenses.training_defense import BaseTrainingDefense
from admarl.utils.pgd import pgd_step, project_epsilon_ball


class SAPPOAdversarialDefense(BaseTrainingDefense):
    """SA-PPO-style adversarial training defense applying inner PGD observation perturbations."""

    def __init__(
        self,
        epsilon: float = 0.05,
        pgd_steps: int = 5,
        pgd_step_size: float = 0.01,
        reg_coeff: float = 1.0,
        norm: str = "linf",
    ) -> None:
        super().__init__(reg_coeff=reg_coeff, epsilon=epsilon)
        self.pgd_steps = pgd_steps
        self.pgd_step_size = pgd_step_size
        self.norm = norm

    def compute_robust_loss(
        self,
        actor: nn.Module,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute SA-PPO robustness loss via inner PGD loop.

        Explicit Detach Semantics:
          1. Clean target anchor logits_clean are explicitly detached: logits_clean = actor(obs).logits.detach()
          2. Final perturbed observation final_perturbed_obs is explicitly .detach()ed before computing L_adv_reg.

        Returns:
            (robust_loss, metrics_dict)
        """
        if self.reg_coeff <= 0.0 or self.pgd_steps <= 0 or self.epsilon <= 0.0:
            return torch.tensor(0.0, device=obs.device), {}

        device = obs.device

        # 1. Clean Target Anchor (Explicitly Detached)
        with torch.no_grad():
            dist_clean = actor(obs)
            logits_clean = dist_clean.logits.detach()

        if not torch.isfinite(logits_clean).all():
            raise RuntimeError("Non-finite robustness loss (NaN/Inf) encountered during SA-PPO update!")

        # 2. Inner PGD Loop to find worst-case observation perturbation
        perturbed_obs = obs.clone() + torch.randn_like(obs) * 1e-4

        for _ in range(self.pgd_steps):
            p_var = perturbed_obs.clone().detach().to(device).requires_grad_(True)
            with torch.enable_grad():
                dist_p = actor(p_var)
                anchor_dist = Categorical(logits=logits_clean)
                kl = kl_divergence(anchor_dist, dist_p).sum()
                kl.backward()

            grad = p_var.grad
            if grad is None or not torch.isfinite(grad).all():
                break

            unprojected = pgd_step(p_var, grad, self.pgd_step_size, norm=self.norm, maximize=True)
            perturbed_obs = project_epsilon_ball(obs, unprojected, self.epsilon, norm=self.norm)

        # 3. Explicit Detach of Final Perturbed Observation (Data-level perturbation)
        final_perturbed_obs = perturbed_obs.detach()
        assert final_perturbed_obs.grad_fn is None, "final_perturbed_obs must be detached before outer loss!"

        # 4. Outer Policy Robustness Loss
        dist_perturbed = actor(final_perturbed_obs)
        anchor_dist = Categorical(logits=logits_clean)
        kl_loss = kl_divergence(anchor_dist, dist_perturbed).mean()

        robust_loss = self.reg_coeff * kl_loss

        # Defensive numerical check (GEMINI.md §7)
        if not torch.isfinite(robust_loss):
            raise RuntimeError("Non-finite robustness loss (NaN/Inf) encountered during SA-PPO update!")

        return robust_loss, {
            "adv_reg_loss": float(robust_loss.item()),
            "train_epsilon": float(self.epsilon),
        }
