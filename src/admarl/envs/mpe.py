"""PettingZoo MPE / MPE2 environment wrapper (GEMINI.md §3 & §4)."""
from __future__ import annotations

from typing import Any

import numpy as np

from admarl.envs.base import BaseMARLEnv

try:
    from mpe2 import simple_spread_v3
except ImportError:
    try:
        from pettingzoo.mpe import simple_spread_v3
    except ImportError as err:
        raise ImportError("Neither mpe2 nor pettingzoo.mpe is installed.") from err


class MPEEnv(BaseMARLEnv):
    """Wrapper for PettingZoo MPE2 environments (e.g. simple_spread_v3)."""

    def __init__(self, env_name: str = "simple_spread", max_cycles: int = 25) -> None:
        self.env_name = env_name
        self.max_cycles = max_cycles

        if env_name in ("simple_spread", "simple_spread_v3"):
            self._env = simple_spread_v3.parallel_env(
                max_cycles=max_cycles, continuous_actions=False
            )
        else:
            raise ValueError(f"Unsupported MPE env_name: {env_name}")

        obs_dict, _ = self._env.reset()
        self.agents = list(self._env.agents)
        self.num_agents = len(self.agents)

        sample_obs = obs_dict[self.agents[0]]
        self.obs_dim = sample_obs.shape[0]
        self.action_dim = self._env.action_space(self.agents[0]).n

        state = self._env.state()
        self.state_dim = state.shape[0] if state is not None else self.obs_dim * self.num_agents

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        obs_dict, _ = self._env.reset(seed=seed)
        self.agents = list(self._env.agents)

        obs = np.stack([obs_dict[a] for a in self.agents], axis=0).astype(np.float32)
        state = self._env.state()
        if state is None:
            state = obs.reshape(-1)
        else:
            state = state.astype(np.float32)

        return obs, state

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        action_dict = {a: int(actions[i]) for i, a in enumerate(self.agents)}
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(action_dict)

        # Handle case where env auto-resets or agents list changes
        active_agents = self.agents

        obs = np.stack([obs_dict.get(a, np.zeros(self.obs_dim, dtype=np.float32)) for a in active_agents], axis=0)
        state = self._env.state()
        if state is None:
            state = obs.reshape(-1)
        else:
            state = state.astype(np.float32)

        rewards = np.array([rew_dict.get(a, 0.0) for a in active_agents], dtype=np.float32)
        terminations = np.array([term_dict.get(a, False) for a in active_agents], dtype=bool)
        truncations = np.array([trunc_dict.get(a, False) for a in active_agents], dtype=bool)
        infos = [info_dict.get(a, {}) for a in active_agents]

        return obs, state, rewards, terminations, truncations, infos

    def close(self) -> None:
        self._env.close()
