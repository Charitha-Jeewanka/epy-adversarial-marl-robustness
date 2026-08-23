"""Targeted Multi-Step PGD Observation Attack Evaluation Harness (GEMINI.md §8 & §11).

Evaluates existing simple_spread_v3 base models across:
- k in [0, 5, 10, 15, 20, 25]
- eps in [0.01, 0.05, 0.10, 0.15] (realism ceiling = 0.15)
- Reference line: Brute-Force Minimizing Oracle

Reports:
1. Post-attack return +/- SEM over 3 seed means
2. Action-flip fraction & Worst-case-hit fraction
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from admarl.algos.mappo import MAPPO
from admarl.attacks.critic_sensitivity import CriticSensitivityAttack
from admarl.envs.mpe import MPEEnv

logger = logging.getLogger(__name__)


def evaluate_pgd_cell(
    mappo: MAPPO,
    env_name: str = "simple_spread",
    episode_length: int = 25,
    budget_k: int = 5,
    epsilon: float = 0.05,
    num_episodes: int = 30,
    eval_seeds: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate policy under targeted multi-step PGD attack."""
    if eval_seeds is None:
        eval_seeds = [3000 + i for i in range(num_episodes)]

    env = MPEEnv(env_name=env_name, max_cycles=episode_length)
    ep_returns: list[float] = []

    attack = CriticSensitivityAttack(budget_k=budget_k, epsilon=epsilon, pgd_steps=5) if budget_k > 0 else None

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
            obs_np = next_obs_np
            state_np = next_state_np

            if term_np.all() or trunc_np.all():
                break

        ep_returns.append(ep_return)

    env.close()

    act_frac = 0.0
    worst_hit_frac = 0.0
    if attack is not None and attack.total_attacked_count > 0:
        act_frac = float(attack.action_changed_count / attack.total_attacked_count)
        worst_hit_frac = float(attack.worst_case_hit_count / attack.total_attacked_count)

    return {
        "return_mean": float(np.mean(ep_returns)),
        "action_changed_frac": act_frac,
        "worst_hit_frac": worst_hit_frac,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    models_dir = Path("runs/sweep_phase6/models")
    seeds = [0, 1, 2]
    num_episodes = 30
    eval_seeds = [3000 + i for i in range(num_episodes)]

    budgets = [0, 5, 10, 15, 20, 25]
    epsilons = [0.01, 0.05, 0.10, 0.15]  # 0.15 is realism ceiling

    print("\n==========================================================================================")
    print("TARGETED MULTI-STEP PGD OBSERVATION ATTACK EVALUATION (simple_spread_v3)")
    print("==========================================================================================")

    for eps in epsilons:
        print(f"\n--- PERTURBATION RADIUS eps = {eps:.2f} (Realism Ceiling = 0.15) ---")
        print("| Budget k | PGD Return +/- SEM | Action Flips (%) | Worst-Case Hits (%) |")
        print("|---|---|---|---|")

        for k in budgets:
            if k == 0:
                # Clean baseline
                seed_rets = []
                for seed in seeds:
                    ckpt_path = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
                    ckpt_data = torch.load(ckpt_path, weights_only=False)
                    mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
                    mappo.load_state_dict(ckpt_data["model_state"])
                    res = evaluate_pgd_cell(mappo, budget_k=0, epsilon=0.0, num_episodes=num_episodes, eval_seeds=eval_seeds)
                    seed_rets.append(res["return_mean"])
                mean_r = float(np.mean(seed_rets))
                sem_r = float(np.std(seed_rets, ddof=1) / np.sqrt(len(seeds)))
                print(f"| {k:2d} (Clean) | {mean_r:6.2f} +/- {sem_r:4.2f} |   0.0% |   0.0% |")
                continue

            seed_rets = []
            seed_flips = []
            seed_worst_hits = []

            for seed in seeds:
                ckpt_path = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
                ckpt_data = torch.load(ckpt_path, weights_only=False)
                mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
                mappo.load_state_dict(ckpt_data["model_state"])

                res = evaluate_pgd_cell(mappo, budget_k=k, epsilon=eps, num_episodes=num_episodes, eval_seeds=eval_seeds)
                seed_rets.append(res["return_mean"])
                seed_flips.append(res["action_changed_frac"])
                seed_worst_hits.append(res["worst_hit_frac"])

            mean_r = float(np.mean(seed_rets))
            sem_r = float(np.std(seed_rets, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
            mean_flip = float(np.mean(seed_flips))
            mean_worst = float(np.mean(seed_worst_hits))

            print(f"| {k:2d} | {mean_r:6.2f} +/- {sem_r:4.2f} | {mean_flip*100:5.1f}% | {mean_worst*100:5.1f}% |")


if __name__ == "__main__":
    main()
