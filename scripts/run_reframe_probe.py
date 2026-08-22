"""Eval-Only Research Reframe Probe (GEMINI.md §8 & §11).

Evaluates the 3 existing trained undefended base models (speed parity, 3 obstacles, distance shaping ON)
under actor-targeted CriticSensitivityAttack across k in [0, 5, 10, 15, 20, 25] at eps=0.05.

Measures:
1. Shaped Return (coordination signal)
2. Mean Distance-to-Prey (tighter geometric positioning metric)
3. Action-Change Fraction
4. True Catches / Episode (reference)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from admarl.algos.mappo import MAPPO
from admarl.attacks.critic_sensitivity import CriticSensitivityAttack
from admarl.envs.mpe import MPEEnv

logger = logging.getLogger(__name__)


def evaluate_reframe_cell(
    mappo: MAPPO,
    env_name: str = "simple_tag",
    episode_length: int = 25,
    attack: CriticSensitivityAttack | None = None,
    num_episodes: int = 30,
    eval_seeds: list[int] | None = None,
    shaping_coeff: float = 0.1,
) -> dict[str, float]:
    """Evaluate policy on an eval cell, measuring shaped return and mean distance to prey."""
    if eval_seeds is None:
        eval_seeds = [3000 + i for i in range(num_episodes)]

    env = MPEEnv(
        env_name=env_name,
        max_cycles=episode_length,
        num_obstacles=3,
        prey_max_speed=1.00,
        prey_accel=3.0,
        shaping_coeff=shaping_coeff,
    )

    ep_shaped_returns: list[float] = []
    ep_mean_distances: list[float] = []
    ep_true_catches: list[float] = []

    mappo.actor.eval()
    if mappo.critic is not None:
        mappo.critic.eval()

    device = mappo.device

    for ep in range(min(num_episodes, len(eval_seeds))):
        seed = eval_seeds[ep]
        obs_np, state_np = env.reset(seed=seed)
        if attack is not None:
            attack.reset_episode()

        ep_shaped_ret = 0.0
        ep_dists: list[float] = []
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

            # Record distance to prey before step or during step
            world = env._env.unwrapped.world
            prey = next(a for a in world.agents if not getattr(a, "adversary", False))
            advs = [a for a in world.agents if getattr(a, "adversary", False)]
            step_dists = [float(np.linalg.norm(a.state.p_pos - prey.state.p_pos)) for a in advs]
            ep_dists.append(float(np.mean(step_dists)))

            next_obs_np, next_state_np, rewards_np, term_np, trunc_np, _info = env.step(actions_np)

            # rewards_np contains distance shaping penalty -0.1 * dist
            ep_shaped_ret += float(rewards_np.mean())

            # Check true physical catches (sparse tag reward +10.0)
            if (rewards_np >= 5.0).any():
                ep_catches += 1.0

            obs_np = next_obs_np
            state_np = next_state_np

            if term_np.all() or trunc_np.all():
                break

        ep_shaped_returns.append(ep_shaped_ret)
        ep_mean_distances.append(float(np.mean(ep_dists)))
        ep_true_catches.append(ep_catches)

    env.close()

    act_frac = 0.0
    if attack is not None and hasattr(attack, "total_attacked_count") and attack.total_attacked_count > 0:
        act_frac = float(attack.action_changed_count / attack.total_attacked_count)

    return {
        "shaped_return_mean": float(np.mean(ep_shaped_returns)),
        "mean_distance_mean": float(np.mean(ep_mean_distances)),
        "catches_mean": float(np.mean(ep_true_catches)),
        "action_changed_frac": act_frac,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    models_dir = Path("runs/spike_simple_tag/models")
    seeds = [0, 1, 2]
    num_episodes = 30
    eval_seeds = [3000 + i for i in range(num_episodes)]
    budgets = [0, 5, 10, 15, 20, 25]
    epsilon = 0.05

    print("\nExecuting Eval-Only Reframe Probe across k in [0, 5, 10, 15, 20, 25]...")

    table_results: list[dict[str, Any]] = []

    for k in budgets:
        seed_shaped_returns = []
        seed_distances = []
        seed_catches = []
        total_attacked = 0
        total_changed = 0

        for seed in seeds:
            ckpt_path = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
            ckpt_data = torch.load(ckpt_path, weights_only=False)

            mappo = MAPPO(obs_dim=18, state_dim=70, action_dim=5, num_agents=3)
            mappo.load_state_dict(ckpt_data["model_state"])

            attack = CriticSensitivityAttack(budget_k=k, epsilon=epsilon) if k > 0 else None

            res = evaluate_reframe_cell(
                mappo=mappo,
                env_name="simple_tag",
                episode_length=25,
                attack=attack,
                num_episodes=num_episodes,
                eval_seeds=eval_seeds,
                shaping_coeff=0.1,
            )

            seed_shaped_returns.append(res["shaped_return_mean"])
            seed_distances.append(res["mean_distance_mean"])
            seed_catches.append(res["catches_mean"])

            if attack is not None:
                total_attacked += attack.total_attacked_count
                total_changed += attack.action_changed_count

        mean_shaped = float(np.mean(seed_shaped_returns))
        sem_shaped = float(np.std(seed_shaped_returns, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0

        mean_dist = float(np.mean(seed_distances))
        sem_dist = float(np.std(seed_distances, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0

        mean_cat = float(np.mean(seed_catches))
        sem_cat = float(np.std(seed_catches, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0

        act_frac = float(total_changed / max(1, total_attacked)) if k > 0 else 0.0

        table_results.append({
            "k": k,
            "shaped_return_mean": mean_shaped,
            "shaped_return_sem": sem_shaped,
            "mean_dist_mean": mean_dist,
            "mean_dist_sem": sem_dist,
            "catches_mean": mean_cat,
            "catches_sem": sem_cat,
            "action_frac": act_frac,
            "seed_returns": seed_shaped_returns,
            "seed_dists": seed_distances,
        })

    print("\n==========================================================================================")
    print("REFRAME PROBE SUMMARY TABLE (eps=0.05, 30 ep/seed, 3 seeds)")
    print("==========================================================================================")
    print("| Attack Budget k | Shaped Return +/- SEM | Mean Distance to Prey +/- SEM | Action-Change Frac | Catches/Ep +/- SEM |")
    print("|---|---|---|---|---|")
    for r in table_results:
        print(f"| {r['k']:2d} | {r['shaped_return_mean']:6.2f} +/- {r['shaped_return_sem']:4.2f} | {r['mean_dist_mean']:5.3f} +/- {r['mean_dist_sem']:5.3f} | {r['action_frac']*100:5.1f}% | {r['catches_mean']:4.2f} +/- {r['catches_sem']:4.2f} |")
    print("==========================================================================================\n")


if __name__ == "__main__":
    main()
