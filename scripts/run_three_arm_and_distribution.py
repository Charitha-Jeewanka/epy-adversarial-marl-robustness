"""Three-Arm Defense Evaluation at eps=0.15 & Per-State Margin Distribution Analysis (GEMINI.md §8 & §11).

Evaluates existing simple_spread_v3 models for none, lipschitz, sa_ppo across k in [0, 5, 10, 15, 20, 25] at eps=0.15 under multi-step PGD.
Computes per-state margin distribution stats:
1. % of attacked states where shift_achieved >= margin_worst
2. % of attacked states where shift_achieved >= margin_adj
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


def evaluate_three_arm_eps15(seeds: list[int] = [0, 1, 2], num_episodes: int = 30) -> dict[str, Any]:
    models_dir = Path("runs/sweep_phase6/models")
    eval_seeds = [3000 + i for i in range(num_episodes)]
    budgets = [0, 5, 10, 15, 20, 25]
    eps = 0.15

    arms = ["none", "lipschitz", "sa_ppo"]
    arm_results: dict[str, Any] = {}

    for arm in arms:
        k_results: dict[str, Any] = {}
        for k in budgets:
            rets, flips, worsts = [], [], []
            for seed in seeds:
                ckpt_path = models_dir / f"{arm}_seed{seed}" / "checkpoints" / "final.pt"
                ckpt_data = torch.load(ckpt_path, weights_only=False)
                mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
                mappo.load_state_dict(ckpt_data["model_state"])
                mappo.actor.eval()
                if mappo.critic is not None:
                    mappo.critic.eval()
                device = mappo.device

                env = MPEEnv(env_name="simple_spread", max_cycles=25)
                attack = CriticSensitivityAttack(budget_k=k, epsilon=eps, pgd_steps=5) if k > 0 else None
                attack_budget_remaining = k
                ep_returns = []

                for ep in range(min(num_episodes, len(eval_seeds))):
                    ep_seed = eval_seeds[ep]
                    obs_np, state_np = env.reset(seed=ep_seed)
                    if attack is not None:
                        attack.reset_episode()
                    attack_budget_remaining = k
                    ep_return = 0.0

                    for step in range(25):
                        obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
                        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device)

                        should_attack = (k >= 25) or (attack_budget_remaining > 0 and (step % (25 // max(1, k)) == 0))
                        if should_attack and attack is not None and k > 0:
                            attack_budget_remaining -= 1
                            perturbed_obs, _ = attack.perturb(
                                obs=obs_tensor, state=state_tensor, critic=mappo.critic, actor=mappo.actor, step=step
                            )
                            obs_tensor = perturbed_obs

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

                rets.append(float(np.mean(ep_returns)))
                flips.append(act_frac)
                worsts.append(worst_hit_frac)

            k_results[str(k)] = {
                "return_mean": float(np.mean(rets)),
                "return_sem": float(np.std(rets, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0,
                "flip_frac": float(np.mean(flips)),
                "worst_hit_frac": float(np.mean(worsts)),
            }
        arm_results[arm] = k_results

    return arm_results


def compute_distributional_margins(seeds: list[int] = [0, 1, 2], num_episodes: int = 30) -> dict[str, float]:
    models_dir = Path("runs/sweep_phase6/models")
    eval_seeds = [3000 + i for i in range(num_episodes)]

    all_shifts = []
    all_adj_margins = []
    all_worst_margins = []

    attack = CriticSensitivityAttack(budget_k=25, epsilon=0.15, pgd_steps=5)

    for seed in seeds:
        ckpt_path = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
        ckpt_data = torch.load(ckpt_path, weights_only=False)
        mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
        mappo.load_state_dict(ckpt_data["model_state"])
        mappo.actor.eval()
        device = mappo.device

        env = MPEEnv(env_name="simple_spread", max_cycles=25)

        for ep in range(min(num_episodes, len(eval_seeds))):
            ep_seed = eval_seeds[ep]
            obs_np, state_np = env.reset(seed=ep_seed)
            attack.reset_episode()

            for step in range(25):
                obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
                state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device)

                with torch.no_grad():
                    clean_dist = mappo.actor(obs_tensor)
                    clean_logits = clean_dist.logits
                    clean_actions = torch.argmax(clean_logits, dim=-1)
                    worst_actions = torch.argmin(clean_logits, dim=-1)

                perturbed_obs, is_perturbed = attack.perturb(
                    obs=obs_tensor, state=state_tensor, critic=mappo.critic, actor=mappo.actor, step=step
                )

                with torch.no_grad():
                    adv_dist = mappo.actor(perturbed_obs)
                    adv_logits = adv_dist.logits

                for i in range(len(clean_actions)):
                    c_act = clean_actions[i].item()
                    w_act = worst_actions[i].item()

                    z_clean = clean_logits[i, c_act].item()
                    z_worst = clean_logits[i, w_act].item()

                    other_logits = [clean_logits[i, a].item() for a in range(5) if a != c_act]
                    z_adj = max(other_logits)

                    shift = abs(clean_logits[i, c_act].item() - adv_logits[i, c_act].item())

                    all_shifts.append(shift)
                    all_adj_margins.append(z_clean - z_adj)
                    all_worst_margins.append(z_clean - z_worst)

                next_obs_np, next_state_np, _rewards_np, term_np, trunc_np, _info = env.step(clean_actions.cpu().numpy())
                obs_np = next_obs_np
                state_np = next_state_np

                if term_np.all() or trunc_np.all():
                    break

        env.close()

    all_shifts_np = np.array(all_shifts)
    all_adj_np = np.array(all_adj_margins)
    all_worst_np = np.array(all_worst_margins)

    pct_exceed_adj = float(np.mean(all_shifts_np >= all_adj_np))
    pct_exceed_worst = float(np.mean(all_shifts_np >= all_worst_np))

    return {
        "pct_exceed_adj_margin": pct_exceed_adj,
        "pct_exceed_worst_margin": pct_exceed_worst,
        "median_achieved_shift": float(np.median(all_shifts_np)),
        "median_adj_margin": float(np.median(all_adj_np)),
        "median_worst_margin": float(np.median(all_worst_np)),
    }


def main() -> None:
    print("\n--- 1. Evaluating Three Arms at eps=0.15 ---")
    arm_res = evaluate_three_arm_eps15()

    print("\n--- THREE-ARM COMPARISON TABLE (eps=0.15, k=10) ---")
    print("| Arm | Clean Return +/- SEM | Attacked Return (k=10, eps=0.15) | Action Flips (%) | Worst Hits (%) |")
    print("|---|---|---|---|---|")
    for arm, grid in arm_res.items():
        clean_ret = grid["0"]["return_mean"]
        clean_sem = grid["0"]["return_sem"]
        k10 = grid["10"]
        print(f"| {arm:9s} | {clean_ret:6.2f} +/- {clean_sem:4.2f} | {k10['return_mean']:6.2f} +/- {k10['return_sem']:4.2f} | {k10['flip_frac']*100:5.1f}% | {k10['worst_hit_frac']*100:5.1f}% |")

    print("\n--- 2. Computing Distributional Margins ---")
    dist_res = compute_distributional_margins()
    print("\n=== DISTRIBUTIONAL PER-STATE MARGIN STATS ===")
    for k, v in dist_res.items():
        print(f"  {k}: {v:.4f}")

    with open("eval_three_arm_unified.json", "w") as f:
        json.dump({"three_arm": arm_res, "distributional_margins": dist_res}, f, indent=2)

    print("\nResults written to eval_three_arm_unified.json.")


if __name__ == "__main__":
    main()
