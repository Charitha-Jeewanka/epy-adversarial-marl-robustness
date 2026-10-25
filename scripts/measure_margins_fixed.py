"""Fix Logit Margin Measurement (RECONCILE 1).

Measures:
1. Achievable logit shift strictly on attacked timesteps under eps=0.15 PGD
2. Adjacent logit margin on flipped timesteps
3. True worst-case logit margin to return-minimizing target action
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


def measure_margins_on_attacked_steps(seeds: list[int] = [0, 1, 2], num_episodes: int = 30) -> dict[str, float]:
    models_dir = Path("runs/sweep_phase6/models")
    eval_seeds = [3000 + i for i in range(num_episodes)]

    all_z_achieved_attacked = []
    all_z_adj_flipped = []
    all_z_adj_all = []
    all_z_worst_all = []

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
                    clean_logits = clean_dist.logits  # shape (3, 5)
                    clean_actions = torch.argmax(clean_logits, dim=-1)
                    worst_actions = torch.argmin(clean_logits, dim=-1)

                # Execute PGD perturbation
                perturbed_obs, is_perturbed = attack.perturb(
                    obs=obs_tensor, state=state_tensor, critic=mappo.critic, actor=mappo.actor, step=step
                )

                with torch.no_grad():
                    adv_dist = mappo.actor(perturbed_obs)
                    adv_logits = adv_dist.logits
                    adv_actions = torch.argmax(adv_logits, dim=-1)

                for i in range(len(clean_actions)):
                    c_act = clean_actions[i].item()
                    w_act = worst_actions[i].item()
                    adv_act = adv_actions[i].item()

                    z_clean = clean_logits[i, c_act].item()
                    z_worst = clean_logits[i, w_act].item()

                    # Margin to worst
                    all_z_worst_all.append(z_clean - z_worst)

                    # Adjacent margin
                    other_logits = [clean_logits[i, a].item() for a in range(5) if a != c_act]
                    z_adj = max(other_logits)
                    adj_margin = z_clean - z_adj
                    all_z_adj_all.append(adj_margin)

                    if is_perturbed:
                        # Achieved shift on clean logit
                        shift = abs(clean_logits[i, c_act].item() - adv_logits[i, c_act].item())
                        all_z_achieved_attacked.append(shift)

                        # Margin on flipped timesteps
                        if adv_act != c_act:
                            all_z_adj_flipped.append(adj_margin)

                next_obs_np, next_state_np, _rewards_np, term_np, trunc_np, _info = env.step(clean_actions.cpu().numpy())
                obs_np = next_obs_np
                state_np = next_state_np

                if term_np.all() or trunc_np.all():
                    break

        env.close()

    return {
        "shift_achieved_on_attacked_mean": float(np.mean(all_z_achieved_attacked)),
        "shift_achieved_on_attacked_std": float(np.std(all_z_achieved_attacked)),
        "margin_adj_all_mean": float(np.mean(all_z_adj_all)),
        "margin_adj_all_std": float(np.std(all_z_adj_all)),
        "margin_adj_flipped_mean": float(np.mean(all_z_adj_flipped)),
        "margin_adj_flipped_std": float(np.std(all_z_adj_flipped)),
        "margin_worst_all_mean": float(np.mean(all_z_worst_all)),
        "margin_worst_all_std": float(np.std(all_z_worst_all)),
    }


def main() -> None:
    res = measure_margins_on_attacked_steps()
    print("\n=== RECONCILED LOGIT MARGIN MEASUREMENTS ===")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
