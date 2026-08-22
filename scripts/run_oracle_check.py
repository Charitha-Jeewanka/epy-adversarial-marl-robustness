"""Directed Worst-Case Oracle vs Critic Sensitivity Attack Evaluation (GEMINI.md §11).

Evaluates the 3 existing simple_spread_v3 base models (runs/sweep_phase6/models/none_seed0,1,2)
under two attack modes:
1. CriticSensitivityAttack (gradient-based observation attack at eps=0.05)
2. BruteForceMinimizingOracle (forces the least-likely / worst-case action argmin_a pi(a|o) for targeted agents)

Across budgets k in [0, 5, 10, 15, 20, 25] over 30 episodes/seed.
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


def evaluate_oracle_vs_attack(
    mappo: MAPPO,
    env_name: str = "simple_spread",
    episode_length: int = 25,
    attack_mode: str = "sensitivity",  # "sensitivity" or "oracle"
    budget_k: int = 5,
    epsilon: float = 0.05,
    num_episodes: int = 30,
    eval_seeds: list[int] | None = None,
) -> dict[str, float]:
    if eval_seeds is None:
        eval_seeds = [3000 + i for i in range(num_episodes)]

    env = MPEEnv(env_name=env_name, max_cycles=episode_length)
    ep_returns: list[float] = []
    total_attacked = 0
    total_action_flips = 0

    mappo.actor.eval()
    if mappo.critic is not None:
        mappo.critic.eval()

    device = mappo.device

    for ep in range(min(num_episodes, len(eval_seeds))):
        seed = eval_seeds[ep]
        obs_np, state_np = env.reset(seed=seed)

        # Attack budget tracking
        attack_budget_remaining = budget_k
        ep_return = 0.0

        for step in range(episode_length):
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                clean_dist = mappo.actor(obs_tensor)
                clean_actions = torch.argmax(clean_dist.logits, dim=-1)
            clean_actions_np = clean_actions.cpu().numpy()

            executed_actions_np = clean_actions_np.copy()

            # Decide whether to attack this step based on budget
            should_attack = (budget_k >= 25) or (attack_budget_remaining > 0 and (step % (episode_length // max(1, budget_k)) == 0))

            if should_attack and budget_k > 0:
                attack_budget_remaining -= 1
                total_attacked += len(clean_actions_np)

                if attack_mode == "sensitivity":
                    # Gradient-based observation attack
                    attack = CriticSensitivityAttack(budget_k=budget_k, epsilon=epsilon)
                    perturbed_obs, _ = attack.perturb(
                        obs=obs_tensor, state=state_tensor, critic=mappo.critic, actor=mappo.actor, step=step
                    )
                    with torch.no_grad():
                        adv_actions, _ = mappo.actor.get_action(perturbed_obs, deterministic=True)
                    executed_actions_np = adv_actions.cpu().numpy()

                elif attack_mode == "oracle":
                    # Brute-Force Oracle: pick argmin_a pi(a|o) for each agent (forcing opposite/least likely action)
                    with torch.no_grad():
                        worst_actions = torch.argmin(clean_dist.logits, dim=-1)
                    executed_actions_np = worst_actions.cpu().numpy()

                total_action_flips += int((executed_actions_np != clean_actions_np).sum())

            next_obs_np, next_state_np, rewards_np, term_np, trunc_np, _info = env.step(executed_actions_np)
            ep_return += float(rewards_np.mean())

            obs_np = next_obs_np
            state_np = next_state_np

            if term_np.all() or trunc_np.all():
                break

        ep_returns.append(ep_return)

    env.close()

    mean_ret = float(np.mean(ep_returns))
    std_ret = float(np.std(ep_returns))
    flip_frac = float(total_action_flips / max(1, total_attacked)) if budget_k > 0 else 0.0

    return {
        "return_mean": mean_ret,
        "return_std": std_ret,
        "action_flip_frac": flip_frac,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    models_dir = Path("runs/sweep_phase6/models")
    seeds = [0, 1, 2]
    num_episodes = 30
    eval_seeds = [3000 + i for i in range(num_episodes)]
    budgets = [0, 5, 10, 15, 20, 25]
    epsilon = 0.05

    print("\nExecuting Directed Worst-Case Oracle vs Critic-Sensitivity Check on simple_spread_v3...")

    print("\n=== SENSITIVITY ATTACK vs BRUTE-FORCE ORACLE ATTACK SUMMARY TABLE ===")
    print("| Budget k | Sensitivity Return +/- SEM | Oracle Return +/- SEM | Sens Flips (%) | Oracle Flips (%) |")
    print("|---|---|---|---|---|")

    for k in budgets:
        sens_returns, sens_flips = [], []
        orac_returns, orac_flips = [], []

        for seed in seeds:
            ckpt_path = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
            ckpt_data = torch.load(ckpt_path, weights_only=False)

            # simple_spread_v3 dims: obs_dim=18, state_dim=54, action_dim=5, num_agents=3
            mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
            mappo.load_state_dict(ckpt_data["model_state"])

            # Sensitivity Attack
            sens_res = evaluate_oracle_vs_attack(
                mappo=mappo, env_name="simple_spread", episode_length=25, attack_mode="sensitivity", budget_k=k, epsilon=epsilon, num_episodes=num_episodes, eval_seeds=eval_seeds
            )
            sens_returns.append(sens_res["return_mean"])
            sens_flips.append(sens_res["action_flip_frac"])

            # Oracle Attack (argmin_a pi(a|o))
            orac_res = evaluate_oracle_vs_attack(
                mappo=mappo, env_name="simple_spread", episode_length=25, attack_mode="oracle", budget_k=k, epsilon=epsilon, num_episodes=num_episodes, eval_seeds=eval_seeds
            )
            orac_returns.append(orac_res["return_mean"])
            orac_flips.append(orac_res["action_flip_frac"])

        sens_mean = float(np.mean(sens_returns))
        sens_sem = float(np.std(sens_returns, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
        sens_flip_mean = float(np.mean(sens_flips))

        orac_mean = float(np.mean(orac_returns))
        orac_sem = float(np.std(orac_returns, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
        orac_flip_mean = float(np.mean(orac_flips))

        print(f"| {k:2d} | {sens_mean:6.2f} +/- {sens_sem:4.2f} | {orac_mean:6.2f} +/- {orac_sem:4.2f} | {sens_flip_mean*100:5.1f}% | {orac_flip_mean*100:5.1f}% |")


if __name__ == "__main__":
    main()
