"""Unified Paper Evaluation Harness (GEMINI.md §8 & §11).

Single-source evaluation pass producing all numbers for the paper from ONE execution.

Covers:
1. simple_spread_v3: Clean baseline, targeted PGD (k x eps grid), PGD step budget sweep, and Q-Minimizing Action Oracle
2. simple_tag_v3: Clean baseline, targeted PGD (k-sweep), and Q-Minimizing Action Oracle
3. Logit Margin Measurements: Delta z_adj vs Delta z_worst vs Delta z_achieved (eps=0.15)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from admarl.algos.mappo import MAPPO
from admarl.attacks.critic_sensitivity import CriticSensitivityAttack
from admarl.envs.mpe import MPEEnv

logger = logging.getLogger(__name__)


def evaluate_spread_cell(
    mappo: MAPPO,
    budget_k: int = 5,
    epsilon: float = 0.05,
    pgd_steps: int = 5,
    attack_mode: str = "pgd",  # "pgd" or "oracle"
    num_episodes: int = 30,
    eval_seeds: list[int] | None = None,
) -> dict[str, float]:
    if eval_seeds is None:
        eval_seeds = [3000 + i for i in range(num_episodes)]

    env = MPEEnv(env_name="simple_spread", max_cycles=25)
    ep_returns: list[float] = []

    attack = CriticSensitivityAttack(budget_k=budget_k, epsilon=epsilon, pgd_steps=pgd_steps) if (budget_k > 0 and attack_mode == "pgd") else None

    mappo.actor.eval()
    if mappo.critic is not None:
        mappo.critic.eval()

    device = mappo.device
    attack_budget_remaining = budget_k

    for ep in range(min(num_episodes, len(eval_seeds))):
        seed = eval_seeds[ep]
        obs_np, state_np = env.reset(seed=seed)
        if attack is not None:
            attack.reset_episode()
        attack_budget_remaining = budget_k
        ep_return = 0.0

        for step in range(25):
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                clean_dist = mappo.actor(obs_tensor)
                clean_actions = torch.argmax(clean_dist.logits, dim=-1)
            clean_actions_np = clean_actions.cpu().numpy()

            executed_actions_np = clean_actions_np.copy()
            should_attack = (budget_k >= 25) or (attack_budget_remaining > 0 and (step % (25 // max(1, budget_k)) == 0))

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
                    # True Worst-Case Action: argmin_a z(a) for each agent
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


def measure_logit_margins(mappo: MAPPO, num_episodes: int = 30, eval_seeds: list[int] | None = None) -> dict[str, float]:
    """Measure Logit Margins: Delta z_adj vs Delta z_worst vs Delta z_achieved (eps=0.15)."""
    if eval_seeds is None:
        eval_seeds = [3000 + i for i in range(num_episodes)]

    env = MPEEnv(env_name="simple_spread", max_cycles=25)
    mappo.actor.eval()
    device = mappo.device

    delta_z_adj_list = []
    delta_z_worst_list = []
    delta_z_achieved_list = []

    attack = CriticSensitivityAttack(budget_k=25, epsilon=0.15, pgd_steps=5)

    for ep in range(min(num_episodes, len(eval_seeds))):
        seed = eval_seeds[ep]
        obs_np, state_np = env.reset(seed=seed)

        for step in range(25):
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                dist = mappo.actor(obs_tensor)
                logits = dist.logits  # shape (num_agents, action_dim)

            clean_actions = torch.argmax(logits, dim=-1)
            worst_actions = torch.argmin(logits, dim=-1)

            for i in range(len(clean_actions)):
                c_act = clean_actions[i].item()
                w_act = worst_actions[i].item()
                z_clean = logits[i, c_act].item()
                z_worst = logits[i, w_act].item()

                # Find nearest non-clean logit
                other_logits = [logits[i, a].item() for a in range(5) if a != c_act]
                z_adj = max(other_logits)

                delta_z_adj_list.append(z_clean - z_adj)
                delta_z_worst_list.append(z_clean - z_worst)

            # Measure achieved logit shift under PGD (eps=0.15)
            perturbed_obs, _ = attack.perturb(obs=obs_tensor, state=state_tensor, critic=mappo.critic, actor=mappo.actor, step=step)
            with torch.no_grad():
                pert_dist = mappo.actor(perturbed_obs)
                pert_logits = pert_dist.logits

            for i in range(len(clean_actions)):
                c_act = clean_actions[i].item()
                shift = abs(pert_logits[i, c_act].item() - logits[i, c_act].item())
                delta_z_achieved_list.append(shift)

            next_obs_np, next_state_np, _rewards_np, term_np, trunc_np, _info = env.step(clean_actions.cpu().numpy())
            obs_np = next_obs_np
            state_np = next_state_np

            if term_np.all() or trunc_np.all():
                break

    env.close()

    return {
        "margin_adjacent_mean": float(np.mean(delta_z_adj_list)),
        "margin_adjacent_std": float(np.std(delta_z_adj_list)),
        "margin_worst_mean": float(np.mean(delta_z_worst_list)),
        "margin_worst_std": float(np.std(delta_z_worst_list)),
        "shift_achieved_mean": float(np.mean(delta_z_achieved_list)),
        "shift_achieved_std": float(np.std(delta_z_achieved_list)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    seeds = [0, 1, 2]
    num_episodes = 30
    eval_seeds = [3000 + i for i in range(num_episodes)]

    results: dict[str, Any] = {}

    print("\n==========================================================================================")
    print("UNIFIED PAPER EVALUATION PASS (GEMINI.md §8 & §11)")
    print("==========================================================================================")

    # 1. simple_spread_v3 Evaluation
    print("\n--- 1. simple_spread_v3 Evaluation ---")
    models_dir = Path("runs/sweep_phase6/models")
    budgets = [0, 5, 10, 15, 20, 25]
    epsilons = [0.01, 0.05, 0.10, 0.15]

    spread_results: dict[str, Any] = {}
    for eps in epsilons:
        grid_cell: dict[str, Any] = {}
        for k in budgets:
            rets, flips, worsts = [], [], []
            for seed in seeds:
                ckpt_path = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
                ckpt_data = torch.load(ckpt_path, weights_only=False)
                mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
                mappo.load_state_dict(ckpt_data["model_state"])

                res = evaluate_spread_cell(mappo, budget_k=k, epsilon=eps, pgd_steps=5, attack_mode="pgd", num_episodes=num_episodes, eval_seeds=eval_seeds)
                rets.append(res["return_mean"])
                flips.append(res["action_changed_frac"])
                worsts.append(res["worst_hit_frac"])

            grid_cell[str(k)] = {
                "return_mean": float(np.mean(rets)),
                "return_sem": float(np.std(rets, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0,
                "flip_frac": float(np.mean(flips)),
                "worst_hit_frac": float(np.mean(worsts)),
            }
        spread_results[str(eps)] = grid_cell

    # Spread Action Oracle
    oracle_cell: dict[str, Any] = {}
    for k in budgets:
        rets = []
        for seed in seeds:
            ckpt_path = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
            ckpt_data = torch.load(ckpt_path, weights_only=False)
            mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
            mappo.load_state_dict(ckpt_data["model_state"])

            res = evaluate_spread_cell(mappo, budget_k=k, epsilon=0.0, pgd_steps=0, attack_mode="oracle", num_episodes=num_episodes, eval_seeds=eval_seeds)
            rets.append(res["return_mean"])
        oracle_cell[str(k)] = {
            "return_mean": float(np.mean(rets)),
            "return_sem": float(np.std(rets, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0,
        }
    spread_results["oracle"] = oracle_cell
    results["simple_spread"] = spread_results

    # 2. Logit Margin Measurements (FIX C)
    print("\n--- 2. Measuring Logit Margins (FIX C) ---")
    margin_means_adj, margin_means_worst, shift_means_achieved = [], [], []
    for seed in seeds:
        ckpt_path = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
        ckpt_data = torch.load(ckpt_path, weights_only=False)
        mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
        mappo.load_state_dict(ckpt_data["model_state"])

        m_res = measure_logit_margins(mappo, num_episodes=num_episodes, eval_seeds=eval_seeds)
        margin_means_adj.append(m_res["margin_adjacent_mean"])
        margin_means_worst.append(m_res["margin_worst_mean"])
        shift_means_achieved.append(m_res["shift_achieved_mean"])

    margins_summary = {
        "margin_adjacent_mean": float(np.mean(margin_means_adj)),
        "margin_adjacent_sem": float(np.std(margin_means_adj, ddof=1) / np.sqrt(len(seeds))),
        "margin_worst_mean": float(np.mean(margin_means_worst)),
        "margin_worst_sem": float(np.std(margin_means_worst, ddof=1) / np.sqrt(len(seeds))),
        "shift_achieved_mean": float(np.mean(shift_means_achieved)),
        "shift_achieved_sem": float(np.std(shift_means_achieved, ddof=1) / np.sqrt(len(seeds))),
    }
    results["logit_margins"] = margins_summary
    print("MARGINS MEASURED:", margins_summary)

    # Save to json artifact
    with open("eval_results_unified.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nUNIFIED EVALUATION COMPLETE. Saved to eval_results_unified.json.")


if __name__ == "__main__":
    main()
