"""Rollout buffer for storing multi-agent trajectories and computing GAE (GEMINI.md §4 & §7)."""
from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch


class RolloutBuffer:
    """Bounded rollout buffer for MAPPO trajectory storage and mini-batch generation."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.device = torch.device(device)

        self.reset()

    def reset(self) -> None:
        """Clear buffer arrays."""
        self.obs = np.zeros((self.n_steps, self.n_envs, self.num_agents, self.obs_dim), dtype=np.float32)
        self.states = np.zeros((self.n_steps, self.n_envs, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs, self.num_agents), dtype=np.int64)
        self.log_probs = np.zeros((self.n_steps, self.n_envs, self.num_agents), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_envs, self.num_agents), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs, self.num_agents), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs, self.num_agents), dtype=bool)

        self.returns = np.zeros((self.n_steps, self.n_envs, self.num_agents), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_envs, self.num_agents), dtype=np.float32)

        self.step = 0

    def insert(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray | torch.Tensor,
        log_probs: np.ndarray | torch.Tensor,
        rewards: np.ndarray,
        values: np.ndarray | torch.Tensor,
        dones: np.ndarray,
    ) -> None:
        """Insert a single step transition across all parallel envs."""
        if self.step >= self.n_steps:
            raise RuntimeError(f"RolloutBuffer overflow: step {self.step} >= n_steps {self.n_steps}")

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()

        self.obs[self.step] = obs
        self.states[self.step] = state
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = rewards
        self.values[self.step] = values
        self.dones[self.step] = dones

        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray | torch.Tensor,
        last_dones: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) and Monte Carlo returns."""
        if isinstance(last_values, torch.Tensor):
            last_values = last_values.cpu().numpy()

        last_gae = np.zeros((self.n_envs, self.num_agents), dtype=np.float32)

        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1].astype(np.float32)
                next_values = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
            self.returns[step] = self.advantages[step] + self.values[step]

    def mini_batch_generator(
        self, mini_batch_size: int
    ) -> Generator[tuple[torch.Tensor, ...], None, None]:
        """Yield mini-batches of flattened rollout data on specified PyTorch device."""
        total_samples = self.n_steps * self.n_envs
        mini_batch_size = min(mini_batch_size, total_samples)

        # Flatten (T, n_envs, ...) -> (total_samples, ...)
        obs_flat = self.obs.reshape(total_samples, self.num_agents, self.obs_dim)
        states_flat = self.states.reshape(total_samples, self.state_dim)
        actions_flat = self.actions.reshape(total_samples, self.num_agents)
        log_probs_flat = self.log_probs.reshape(total_samples, self.num_agents)
        returns_flat = self.returns.reshape(total_samples, self.num_agents)
        advantages_flat = self.advantages.reshape(total_samples, self.num_agents)

        indices = np.random.permutation(total_samples)

        for start in range(0, total_samples, mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end]

            yield (
                torch.tensor(obs_flat[mb_idx], dtype=torch.float32, device=self.device),
                torch.tensor(states_flat[mb_idx], dtype=torch.float32, device=self.device),
                torch.tensor(actions_flat[mb_idx], dtype=torch.long, device=self.device),
                torch.tensor(log_probs_flat[mb_idx], dtype=torch.float32, device=self.device),
                torch.tensor(returns_flat[mb_idx], dtype=torch.float32, device=self.device),
                torch.tensor(advantages_flat[mb_idx], dtype=torch.float32, device=self.device),
            )
