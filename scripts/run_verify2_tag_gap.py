"""VERIFY 2: Accessibility Gap Replication in simple_tag_v3 (GEMINI.md §11).

Evaluates existing simple_tag_v3 base models (runs/spike_simple_tag/models/none_seed0,1,2)
comparing:
1. Brute-Force Minimizing Oracle (direct action-level attack)
2. Targeted Multi-Step PGD Attack (observation attack at eps=0.15, pgd_steps=10)

Across budgets k in [0, 5, 10, 15, 20, 25] over 30 episodes/seed.
Reports:
- PGD Return (unshaped) +/- SEM
- Oracle Return (unshaped) +/- SEM
- Action-flip %
- Worst-case-hit %
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


def evaluate_tag_cell(
    mappo: MAPPO,
    env_name: str = "simple_tag",
    episode_length: int = 25,
    attack_mode: str = "pgd",  # "pgd" or "oracle"
    budget_k: int = 5,
    epsilon: float = 0.15,
    num_episodes: int = 30,
    eval_seeds: list[int] | None = None,
) -> dict[str, float]:
    if eval_seeds is None:
        eval_seeds = [3000 + i for i in range(num_episodes)]

    env = MPEEnv(env_name=env_name, max_cycles=episode_length, num_obstacles=3, shaping_coeff=0.0)
    ep_returns: list[float] = []

    attack = CriticSensitivityAttack(budget_k=budget_k, epsilon=epsilon, pgd_steps=10) if (budget_k > 0 and attack_mode == "pgd") else None

    mappo.actor.eval()
    if mappo.critic is not None:
        mappo.critic.eval()

    device = mappo.device

    for ep in range(min(num_episodes, len(eval_seeds))):
        seed = eval_seeds[ep]
        obs_np, state_np = env.reset(seed=seed)
        if attack is not None:
            attack.reset_episode()

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

            should_attack = (budget_k >= 25) or (attack_budget_remaining > 0 and (step % (episode_length // max(1, budget_k)) == 0))

            if should_attack and budget_k > 0:
                attack_budget_remaining -= 1

                if attack_mode == "pgd" and attack is not None:
                    perturbed_obs, _ = attack.perturb(
                        obs=obs_tensor, state=state_tensor, critic=mappo.critic, actor=mappo.actor, step=step
                    )
                    with torch.no_grad():
                        adv_actions, _ = mappo.actor.get_action(perturbed_obs, deterministic=True)
                    executed_actions_np = adv_actions.cpu().numpy()

                elif attack_mode == "oracle":
                    with torch.no_grad():
                        worst_actions = torch.argmin(clean_dist.logits, dim=-1)
                    executed_actions_np = worst_actions.cpu().numpy()

            next_obs_np, next_state_np, rewards_np, term_np, trunc_np, _info = env.step(executed_actions_np)

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
    models_dir = Path("runs/spike_simple_tag/models")
    seeds = [0, 1, 2]
    num_episodes = 30
    eval_seeds = [3000 + i for i in range(num_episodes)]

    budgets = [0, 5, 10, 15, 20, 25]
    eps = 0.15

    print("\n==========================================================================================")
    print("VERIFY 2: ACCESSIBILITY GAP REPLICATION IN simple_tag_v3 (eps=0.15 Ceiling)")
    print("==========================================================================================")
    print("| Budget k | PGD Return +/- SEM | Oracle Return +/- SEM | PGD Flips (%) | PGD Worst Hits (%) |")
    print("|---|---|---|---|---|")

    for k in budgets:
        pgd_rets, oracle_rets = [], []
        pgd_flips, pgd_worst_hits = [], []

        for seed in seeds:
            ckpt_path = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
            ckpt_data = torch.load(ckpt_path, weights_only=False)
            # simple_tag_v3 predator dims: obs_dim=18, state_dim=70, action_dim=5, num_agents=3
            mappo = MAPPO(obs_dim=18, state_dim=70, action_dim=5, num_agents=3)
            mappo.load_state_dict(ckpt_data["model_state"])

            # PGD attack
            res_pgd = evaluate_tag_cell(mappo, env_name="simple_tag", budget_k=k, epsilon=eps, attack_mode="pgd", num_episodes=num_episodes, eval_seeds=eval_seeds)
            pgd_rets.append(res_pgd["return_mean"])
            pgd_flips.append(res_pgd["action_changed_frac"])
            pgd_worst_hits.append(res_pgd["worst_hit_frac"])

            # Oracle attack
            res_oracle = evaluate_tag_cell(mappo, env_name="simple_tag", budget_k=k, epsilon=eps, attack_mode="oracle", num_episodes=num_episodes, eval_seeds=eval_seeds)
            oracle_rets.append(res_oracle["return_mean"])

        pgd_mean = float(np.mean(pgd_rets))
        pgd_sem = float(np.std(pgd_rets, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0

        oracle_mean = float(np.mean(oracle_rets))
        oracle_sem = float(np.std(oracle_rets, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0

        flip_mean = float(np.mean(pgd_flips))
        worst_mean = float(np.mean(pgd_worst_hits))

        print(f"| {k:2d} | {pgd_mean:6.2f} +/- {pgd_sem:4.2f} | {oracle_mean:6.2f} +/- {oracle_sem:4.2f} | {flip_mean*100:5.1f}% | {worst_mean*100:5.1f}% |")


if __name__ == "__main__":
    main()
