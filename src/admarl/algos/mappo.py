"""Multi-Agent PPO (MAPPO) baseline algorithm implementation (GEMINI.md §4 & §7)."""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn, optim

from admarl.algos.models import ActorNetwork, CentralizedCriticNetwork
from admarl.defenses.base import BaseCriticRegularizer
from admarl.defenses.training_defense import BaseTrainingDefense

logger = logging.getLogger(__name__)


class MAPPO:
    """MAPPO Policy with Centralized Critic."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        actor_lr: float = 5e-4,
        critic_lr: float = 5e-4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 10.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

        self.critic = CentralizedCriticNetwork(
            state_dim=state_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def get_actions_and_values(
        self, obs: torch.Tensor, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step policy for action selection and value evaluation.

        Args:
            obs: (n_envs, num_agents, obs_dim)
            state: (n_envs, state_dim)
            deterministic: whether to select argmax action

        Returns:
            actions: (n_envs, num_agents)
            log_probs: (n_envs, num_agents)
            values: (n_envs, num_agents)
        """
        obs = obs.to(self.device)
        state = state.to(self.device)

        actions, log_probs = self.actor.get_action(obs, deterministic=deterministic)
        values = self.critic(state)  # (n_envs, num_agents)

        return actions, log_probs, values

    def update(
        self,
        obs_b: torch.Tensor,
        state_b: torch.Tensor,
        actions_b: torch.Tensor,
        old_log_probs_b: torch.Tensor,
        returns_b: torch.Tensor,
        advantages_b: torch.Tensor,
        regularizer: BaseCriticRegularizer | None = None,
        training_defense: BaseTrainingDefense | None = None,
    ) -> dict[str, float]:
        """Execute PPO update on a batch of experience.

        All tensors are assumed to be on self.device.
        """
        obs_b = obs_b.to(self.device)
        state_b = state_b.to(self.device)
        actions_b = actions_b.to(self.device)
        old_log_probs_b = old_log_probs_b.to(self.device)
        returns_b = returns_b.to(self.device)
        advantages_b = advantages_b.to(self.device)

        # Normalize advantages per mini-batch
        adv_mean = advantages_b.mean()
        adv_std = advantages_b.std() + 1e-8
        norm_advantages = (advantages_b - adv_mean) / adv_std

        # Evaluate current actor
        log_probs, entropy = self.actor.evaluate_actions(obs_b, actions_b)

        # PPO ratio & clipped surrogate loss
        ratios = torch.exp(log_probs - old_log_probs_b)
        surr1 = ratios * norm_advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * norm_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy_loss = -entropy.mean()
        actor_loss = policy_loss + self.entropy_coef * entropy_loss

        # Training defense (e.g. SA-PPO adversarial training) if provided
        adv_reg_loss = torch.tensor(0.0, device=self.device)
        adv_metrics: dict[str, float] = {}
        if training_defense is not None:
            adv_reg_loss, adv_metrics = training_defense.compute_robust_loss(self.actor, obs_b, actions_b)
            actor_loss = actor_loss + adv_reg_loss

        # Evaluate critic
        values = self.critic(state_b)
        value_loss = 0.5 * ((values - returns_b) ** 2).mean()

        # Regularizer penalty if provided
        reg_loss = torch.tensor(0.0, device=self.device)
        if regularizer is not None:
            reg_loss = regularizer.penalty(self.critic, state_b)

        critic_loss = self.value_loss_coef * value_loss + reg_loss

        # Defensive numerical checks (GEMINI.md §4)
        for loss_name, loss_val in [
            ("actor_loss", actor_loss),
            ("critic_loss", critic_loss),
            ("reg_loss", reg_loss),
            ("adv_reg_loss", adv_reg_loss),
        ]:
            if not torch.isfinite(loss_val):
                raise RuntimeError(f"Non-finite loss encountered in MAPPO update: {loss_name} = {loss_val.item()}")

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return {
            "actor_loss": policy_loss.item(),
            "entropy": entropy.mean().item(),
            "critic_loss": value_loss.item(),
            "reg_loss": reg_loss.item(),
            "adv_reg_loss": float(adv_reg_loss.item()),
            "train_epsilon": adv_metrics.get("train_epsilon", 0.0),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])

    def optimizer_state_dict(self) -> dict[str, Any]:
        """Return optimizer state dicts."""
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_optimizer_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state dicts."""
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
