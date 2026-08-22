"""Vectorized multi-agent environment wrapper for parallel rollout collection (GEMINI.md §2)."""
from __future__ import annotations

from typing import Any

import numpy as np

from admarl.envs.base import BaseMARLEnv
from admarl.envs.mpe import MPEEnv


class VectorMARLEnv:
    """CPU-vectorized container managing n_parallel_envs instances."""

    def __init__(self, env_name: str, n_envs: int = 8, max_cycles: int = 25) -> None:
        self.n_envs = n_envs
        self.envs: list[BaseMARLEnv] = [
            MPEEnv(env_name=env_name, max_cycles=max_cycles) for _ in range(n_envs)
        ]
        self.num_agents = self.envs[0].num_agents
        self.obs_dim = self.envs[0].obs_dim
        self.state_dim = self.envs[0].state_dim
        self.action_dim = self.envs[0].action_dim

    def reset(self, seeds: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reset all environments and return stacked (obs, state).

        obs: shape (n_envs, num_agents, obs_dim)
        state: shape (n_envs, state_dim)
        """
        all_obs = []
        all_states = []
        for i, env in enumerate(self.envs):
            s = seeds[i] if seeds is not None and i < len(seeds) else None
            o, st = env.reset(seed=s)
            all_obs.append(o)
            all_states.append(st)

        return np.stack(all_obs, axis=0), np.stack(all_states, axis=0)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[dict[str, Any]]]]:
        """Step all environments with joint actions (n_envs, num_agents).

        Auto-resets individual environments when episode finishes.

        Returns:
            next_obs: (n_envs, num_agents, obs_dim)
            next_state: (n_envs, state_dim)
            rewards: (n_envs, num_agents)
            dones: (n_envs, num_agents)
            infos: list of info dict lists per env
        """
        all_obs = []
        all_states = []
        all_rewards = []
        all_dones = []
        all_infos = []

        for i, env in enumerate(self.envs):
            o, st, r, term, trunc, info = env.step(actions[i])
            done = term | trunc
            all_rewards.append(r)
            all_dones.append(done)

            # Auto-reset if all agents are done
            if np.all(done):
                reset_obs, reset_state = env.reset()
                all_obs.append(reset_obs)
                all_states.append(reset_state)
            else:
                all_obs.append(o)
                all_states.append(st)

            all_infos.append(info)

        return (
            np.stack(all_obs, axis=0),
            np.stack(all_states, axis=0),
            np.stack(all_rewards, axis=0),
            np.stack(all_dones, axis=0),
            all_infos,
        )

    def close(self) -> None:
        """Close all underlying environments."""
        for env in self.envs:
            env.close()
