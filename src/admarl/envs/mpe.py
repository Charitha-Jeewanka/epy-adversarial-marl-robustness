"""PettingZoo MPE / MPE2 environment wrapper (GEMINI.md §3 & §4)."""
from __future__ import annotations

from typing import Any

import numpy as np

from admarl.envs.base import BaseMARLEnv

try:
    from mpe2 import simple_spread_v3, simple_tag_v3
except ImportError:
    try:
        from pettingzoo.mpe import simple_spread_v3, simple_tag_v3
    except ImportError as err:
        raise ImportError("Neither mpe2 nor pettingzoo.mpe is installed.") from err


class MPEEnv(BaseMARLEnv):
    """Wrapper for PettingZoo MPE2 environments (simple_spread_v3, simple_tag_v3)."""

    def __init__(
        self,
        env_name: str = "simple_spread",
        max_cycles: int = 25,
        num_obstacles: int = 3,
        prey_max_speed: float = 1.05,
        prey_accel: float = 3.1,
        shaping_coeff: float = 0.0,
    ) -> None:
        self.env_name = env_name
        self.max_cycles = max_cycles
        self.num_obstacles = num_obstacles
        self.prey_max_speed = prey_max_speed
        self.prey_accel = prey_accel
        self.shaping_coeff = shaping_coeff
        self.is_simple_tag = env_name in ("simple_tag", "simple_tag_v3")

        if env_name in ("simple_spread", "simple_spread_v3"):
            self._env = simple_spread_v3.parallel_env(
                max_cycles=max_cycles, continuous_actions=False
            )
        elif self.is_simple_tag:
            self._env = simple_tag_v3.parallel_env(
                num_obstacles=num_obstacles,
                max_cycles=max_cycles,
                continuous_actions=False,
            )
        else:
            raise ValueError(f"Unsupported MPE env_name: {env_name}")

        obs_dict, _ = self._env.reset()
        self.all_agents = list(self._env.agents)

        self.prey_agent: str | None = None
        if self.is_simple_tag:
            # CTDE re-framing: train the 3 adversaries as the cooperative team
            self.agents = [a for a in self.all_agents if "adversary" in a]
            self.prey_agent = "agent_0"
            self._apply_prey_speed_tuning()
        else:
            self.agents = list(self.all_agents)
            self.prey_agent = None

        self.num_agents = len(self.agents)
        sample_obs = obs_dict[self.agents[0]]
        self.obs_dim = sample_obs.shape[0]
        self.action_dim = self._env.action_space(self.agents[0]).n

        state = self._env.state()
        self.state_dim = state.shape[0] if state is not None else self.obs_dim * self.num_agents

        self._latest_obs_dict = obs_dict

        # Assert environment dimensions post-initialization
        assert self.num_agents == 3, f"Expected 3 cooperative agents, got {self.num_agents}"

    def _apply_prey_speed_tuning(self) -> None:
        """Apply custom prey max_speed and accel to world object."""
        world = getattr(self._env.unwrapped, "world", None)
        if world is not None:
            for agent in world.agents:
                if not getattr(agent, "adversary", False):
                    agent.max_speed = self.prey_max_speed
                    agent.accel = self.prey_accel

    def _flee_heuristic(self, prey_obs: np.ndarray) -> int:
        """Flee-nearest-predator heuristic policy for frozen prey (agent_0)."""
        calc_n = (len(prey_obs) - 10) // 2
        start = 4 + 2 * calc_n
        rel_positions = [
            prey_obs[start : start + 2],
            prey_obs[start + 2 : start + 4],
            prey_obs[start + 4 : start + 6],
        ]
        dists = [np.linalg.norm(rp) for rp in rel_positions]
        nearest_rp = rel_positions[int(np.argmin(dists))]
        dx, dy = float(nearest_rp[0]), float(nearest_rp[1])
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 2
        else:
            return 3 if dy > 0 else 4

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        obs_dict, _ = self._env.reset(seed=seed)
        self.all_agents = list(self._env.agents)
        if self.is_simple_tag:
            self._apply_prey_speed_tuning()

        self._latest_obs_dict = obs_dict

        obs = np.stack([obs_dict[a] for a in self.agents], axis=0).astype(np.float32)
        state = self._env.state()
        if state is None:
            state = obs.reshape(-1)
        else:
            state = state.astype(np.float32)

        assert obs.shape == (self.num_agents, self.obs_dim), f"Expected obs shape ({self.num_agents}, {self.obs_dim}), got {obs.shape}"
        assert state.shape == (self.state_dim,), f"Expected state shape ({self.state_dim},), got {state.shape}"

        return obs, state

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        action_dict = {a: int(actions[i]) for i, a in enumerate(self.agents)}

        if self.is_simple_tag and self.prey_agent is not None and self.prey_agent in self.all_agents:
            prey_obs = self._latest_obs_dict.get(self.prey_agent)
            if prey_obs is not None:
                action_dict[self.prey_agent] = self._flee_heuristic(prey_obs)

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(action_dict)
        self._latest_obs_dict = obs_dict

        active_agents = self.agents

        obs = np.stack([obs_dict.get(a, np.zeros(self.obs_dim, dtype=np.float32)) for a in active_agents], axis=0)
        state = self._env.state()
        if state is None:
            state = obs.reshape(-1)
        else:
            state = state.astype(np.float32)

        rewards = np.array([rew_dict.get(a, 0.0) for a in active_agents], dtype=np.float32)

        # Dense distance-to-prey reward shaping for predators (Stage 1 trainability lever)
        if self.is_simple_tag and self.shaping_coeff > 0.0:
            world = getattr(self._env.unwrapped, "world", None)
            if world is not None:
                prey = next((a for a in world.agents if not getattr(a, "adversary", False)), None)
                if prey is not None:
                    advs = [a for a in world.agents if getattr(a, "adversary", False)]
                    for i, adv in enumerate(advs):
                        if i < len(rewards):
                            dist = float(np.linalg.norm(adv.state.p_pos - prey.state.p_pos))
                            rewards[i] -= self.shaping_coeff * dist

        terminations = np.array([term_dict.get(a, False) for a in active_agents], dtype=bool)
        truncations = np.array([trunc_dict.get(a, False) for a in active_agents], dtype=bool)
        infos = [info_dict.get(a, {}) for a in active_agents]

        return obs, state, rewards, terminations, truncations, infos

    def close(self) -> None:
        self._env.close()
