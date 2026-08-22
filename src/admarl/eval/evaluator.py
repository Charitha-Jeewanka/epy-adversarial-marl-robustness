"""Policy evaluator under observation attacks (GEMINI.md §4)."""
from __future__ import annotations

import numpy as np
import torch

from admarl.algos.mappo import MAPPO
from admarl.attacks.base import BaseAttack
from admarl.envs.mpe import MPEEnv


def evaluate_policy(
    mappo: MAPPO,
    env_name: str = "simple_spread",
    episode_length: int = 25,
    attack: BaseAttack | None = None,
    num_episodes: int = 10,
    eval_seeds: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate a trained MAPPO policy under an observation attack.

    Args:
        mappo: Trained MAPPO algorithm instance
        env_name: Name of MPE environment
        episode_length: Maximum steps per episode
        attack: BaseAttack plugin instance
        num_episodes: Number of evaluation episodes
        eval_seeds: List of random seeds for evaluation episodes

    Returns:
        Dictionary containing mean and std of post-attack episode returns
    """
    env = MPEEnv(env_name=env_name, max_cycles=episode_length)
    episode_returns: list[float] = []
    episode_catches: list[float] = []

    if eval_seeds is None:
        eval_seeds = [1000 + i for i in range(num_episodes)]

    mappo.actor.eval()
    if mappo.critic is not None:
        mappo.critic.eval()

    device = mappo.device

    for ep in range(min(num_episodes, len(eval_seeds))):
        seed = eval_seeds[ep]
        obs_np, state_np = env.reset(seed=seed)
        if attack is not None:
            attack.reset_episode()

        ep_return = 0.0
        ep_catches = 0.0

        for step in range(episode_length):
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device)

            if attack is not None:
                obs_tensor, _is_perturbed = attack.perturb(
                    obs=obs_tensor,
                    state=state_tensor,
                    critic=mappo.critic,
                    actor=mappo.actor,
                    step=step,
                )

            with torch.no_grad():
                actions_tensor, _ = mappo.actor.get_action(obs_tensor, deterministic=True)

            actions_np = actions_tensor.cpu().numpy()
            next_obs_np, next_state_np, rewards_np, term_np, trunc_np, _info = env.step(actions_np)

            ep_return += float(rewards_np.mean())
            if (rewards_np >= 10.0).any():
                ep_catches += 1.0

            obs_np = next_obs_np
            state_np = next_state_np

            if term_np.all() or trunc_np.all():
                break

        episode_returns.append(ep_return)
        episode_catches.append(ep_catches)

    env.close()

    mean_ret = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_ret = float(np.std(episode_returns)) if episode_returns else 0.0
    mean_catches = float(np.mean(episode_catches)) if episode_catches else 0.0
    std_catches = float(np.std(episode_catches)) if episode_catches else 0.0

    action_changed_frac = 0.0
    if attack is not None and hasattr(attack, "total_attacked_count") and attack.total_attacked_count > 0:
        action_changed_frac = float(attack.action_changed_count / attack.total_attacked_count)

    return {
        "post_attack_return_mean": mean_ret,
        "post_attack_return_std": std_ret,
        "catch_rate_mean": mean_catches,
        "catch_rate_std": std_catches,
        "action_changed_fraction": action_changed_frac,
    }
