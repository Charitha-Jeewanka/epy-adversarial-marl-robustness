"""Environment interface for cooperative MARL (GEMINI.md §4)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseMARLEnv(ABC):
    """Abstract interface for multi-agent RL environments."""

    num_agents: int
    agents: list[str]
    obs_dim: int
    state_dim: int
    action_dim: int

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reset environment and return (obs, state).

        obs: np.ndarray of shape (num_agents, obs_dim)
        state: np.ndarray of shape (state_dim,)
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step environment with joint actions.

        actions: np.ndarray of shape (num_agents,) containing discrete action indices

        Returns:
            obs: (num_agents, obs_dim)
            state: (state_dim,)
            rewards: (num_agents,)
            terminations: (num_agents,)
            truncations: (num_agents,)
            infos: list of info dicts per agent
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        raise NotImplementedError
