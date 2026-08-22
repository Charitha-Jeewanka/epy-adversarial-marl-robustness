"""Script to execute simple_tag_v3 experimental spike probe (GEMINI.md §8).

1. Train 3 seeds of undefended MAPPO on simple_tag_v3 with 400k steps, 3 obstacles, prey speed 1.05.
2. Evaluate relative nominal competence gate (Random Predators vs Trained MAPPO Predators).
3. If competent, run attack-surface probe across k in [5, 10, 15, 20, 25] at eps=0.05 over 30 episodes/seed.
"""
from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from admarl.algos.mappo import MAPPO
from admarl.attacks.critic_sensitivity import CriticSensitivityAttack
from admarl.envs.mpe import MPEEnv
from admarl.eval.evaluator import evaluate_policy
from admarl.training.train import Trainer
from admarl.utils.config import load_config
from admarl.utils.logger import ExperimentLogger
from admarl.utils.memory import handle_cuda_oom

logger = logging.getLogger(__name__)


def evaluate_random_predators(
    env_name: str = "simple_tag",
    episode_length: int = 25,
    num_episodes: int = 30,
    seeds: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate random predator baseline performance on the exact same config."""
    if seeds is None:
        seeds = [0, 1, 2]

    all_seed_catches = []
    all_seed_returns = []

    for seed in seeds:
        env = MPEEnv(env_name=env_name, max_cycles=episode_length, num_obstacles=3, prey_max_speed=1.00, prey_accel=3.0)
        ep_returns = []
        ep_catches = []

        for ep in range(num_episodes):
            ep_seed = 4000 + seed * 100 + ep
            _obs_np, _state_np = env.reset(seed=ep_seed)
            ret = 0.0
            catches = 0.0

            for _step in range(episode_length):
                # Uniform random actions for 3 predators
                actions_np = np.random.randint(0, 5, size=3)
                _next_obs_np, _next_state_np, rewards_np, term_np, trunc_np, _info = env.step(actions_np)

                ret += float(rewards_np.mean())
                if (rewards_np >= 10.0).any():
                    catches += 1.0

                if term_np.all() or trunc_np.all():
                    break

            ep_returns.append(ret)
            ep_catches.append(catches)

        env.close()
        all_seed_returns.append(float(np.mean(ep_returns)))
        all_seed_catches.append(float(np.mean(ep_catches)))

    mean_ret = float(np.mean(all_seed_returns))
    sem_ret = float(np.std(all_seed_returns, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
    mean_cat = float(np.mean(all_seed_catches))
    sem_cat = float(np.std(all_seed_catches, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0

    return {
        "return_mean": mean_ret,
        "return_sem": sem_ret,
        "catch_rate_mean": mean_cat,
        "catch_rate_sem": sem_cat,
        "seed_catches": all_seed_catches,
    }


def run_spike(config_path: Path | str) -> None:
    raw_cfg = load_config(config_path)
    spike_cfg = raw_cfg.get("spike", raw_cfg)

    base_out_dir = Path(spike_cfg.get("output_dir", "runs/spike_simple_tag"))
    models_dir = base_out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    seeds: list[int] = spike_cfg.get("seeds", [0, 1, 2])
    eval_cfg = spike_cfg.get("eval", {})
    env_cfg = spike_cfg.get("env", {})
    train_cfg = spike_cfg.get("train", {})

    num_episodes = eval_cfg.get("num_episodes", 30)
    eval_seeds = eval_cfg.get("eval_seeds", [3000 + i for i in range(num_episodes)])

    # =========================================================================
    # STAGE 1: Train 3 undefended base models on simple_tag (400k steps)
    # =========================================================================
    for seed in seeds:
        model_folder = models_dir / f"none_seed{seed}"
        final_ckpt = model_folder / "checkpoints" / "final.pt"

        if final_ckpt.exists():
            logger.info("Base model ready: %s", model_folder)
            continue

        logger.info("Training base model: arm=none, seed=%d (400k steps)", seed)

        base_config: dict[str, Any] = {
            "seed": seed,
            "deterministic": True,
            "hardware": {"device": "cuda" if torch.cuda.is_available() else "cpu", "max_vram_gb": 5.0},
            "env": env_cfg,
            "model": {"hidden_dim": 64, "num_layers": 2},
            "train": train_cfg,
            "defense": {"name": "none", "penalty_coeff": 0.0},
            "adv_training": {"enabled": False},
            "attack": {"name": "none", "budget_k": 0, "epsilon": 0.0},
            "sweep_meta": {"arm": "none", "budget_k": 0, "epsilon": 0.0},
        }

        try:
            model_folder.mkdir(parents=True, exist_ok=True)
            with open(model_folder / "resolved_config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(base_config, f)

            exp_logger = ExperimentLogger(config=base_config, base_output_dir=base_out_dir)
            trainer = Trainer(config=base_config, exp_logger=exp_logger)
            trainer.train()

            if exp_logger.run_dir and (exp_logger.run_dir / "checkpoints" / "final.pt").exists():
                (model_folder / "checkpoints").mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(exp_logger.run_dir / "checkpoints" / "final.pt", final_ckpt)

        except RuntimeError as e:
            handle_cuda_oom(e, context=f"Base Model Training none_seed{seed}")
            logger.error("Error training base model none_seed%d: %s", seed, e)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 2: Relative Nominal Competence Gate (Random vs Trained)
    # =========================================================================
    logger.info("--- EVALUATING RELATIVE COMPETENCE GATE ---")
    
    # 1. Random Predator Baseline
    rand_res = evaluate_random_predators(env_name="simple_tag", episode_length=25, num_episodes=num_episodes, seeds=seeds)
    logger.info("Random Predator Catch Rate: %.2f +/- %.2f catches/ep", rand_res["catch_rate_mean"], rand_res["catch_rate_sem"])

    # 2. Trained MAPPO Predators
    clean_returns = []
    clean_catches = []

    for seed in seeds:
        final_ckpt = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
        ckpt_data = torch.load(final_ckpt, weights_only=False)

        # Dynamic dimension spec from env (obs_dim=18, state_dim=70)
        sample_env = MPEEnv(env_name="simple_tag", max_cycles=25, num_obstacles=3, prey_max_speed=1.00, prey_accel=3.0)
        obs_dim, state_dim = sample_env.obs_dim, sample_env.state_dim
        sample_env.close()

        mappo = MAPPO(obs_dim=obs_dim, state_dim=state_dim, action_dim=5, num_agents=3)
        mappo.load_state_dict(ckpt_data["model_state"])

        clean_res = evaluate_policy(
            mappo=mappo,
            env_name="simple_tag",
            episode_length=25,
            attack=None,
            num_episodes=num_episodes,
            eval_seeds=eval_seeds,
        )

        clean_returns.append(clean_res["post_attack_return_mean"])
        clean_catches.append(clean_res["catch_rate_mean"])
        logger.info("Seed %d Trained Return: %.2f, Catches/Ep: %.2f", seed, clean_res["post_attack_return_mean"], clean_res["catch_rate_mean"])

    clean_mean = float(np.mean(clean_returns))
    clean_sem = float(np.std(clean_returns, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
    catch_mean = float(np.mean(clean_catches))
    catch_sem = float(np.std(clean_catches, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0

    print("\n==================================================")
    print("RELATIVE COMPETENCE GATE RESULTS")
    print(f"Random Predator Catch Rate:  {rand_res['catch_rate_mean']:.2f} +/- {rand_res['catch_rate_sem']:.2f} catches/ep (Seeds: {rand_res['seed_catches']})")
    print(f"Trained MAPPO Catch Rate:    {catch_mean:.2f} +/- {catch_sem:.2f} catches/ep (Seeds: {[round(c, 2) for c in clean_catches]})")
    print(f"Trained MAPPO Return:        {clean_mean:.2f} +/- {clean_sem:.2f}")

    # Pass bar check: Trained catch rate must be substantially above random floor (> 2x random catch rate AND >= 0.50 catches/ep)
    competence_passed = (catch_mean >= 0.50) and (catch_mean >= 2.0 * max(0.01, rand_res["catch_rate_mean"]))
    print(f"COMPETENCE GATE STATUS:      {'PASSED' if competence_passed else 'FAILED'}")
    print("==================================================\n")

    if not competence_passed:
        print("COMPETENCE GATE FAILED: Trained policy catch rate is not sufficiently above random floor.")
        print("Nominal gate FAILED; attack probe INCONCLUSIVE because policy is not competent.")
        return

    # =========================================================================
    # STAGE 3: Attack-Surface Probe (Only if Competence Passed)
    # =========================================================================
    budgets = [5, 10, 15, 20, 25]
    epsilon = 0.05
    table_rows = []

    for k in budgets:
        seed_returns = []
        seed_catches = []
        total_attacked = 0
        total_changed = 0

        for seed in seeds:
            final_ckpt = models_dir / f"none_seed{seed}" / "checkpoints" / "final.pt"
            ckpt_data = torch.load(final_ckpt, weights_only=False)

            mappo = MAPPO(obs_dim=18, state_dim=70, action_dim=5, num_agents=3)
            mappo.load_state_dict(ckpt_data["model_state"])

            attack = CriticSensitivityAttack(budget_k=k, epsilon=epsilon)

            adv_res = evaluate_policy(
                mappo=mappo,
                env_name="simple_tag",
                episode_length=25,
                attack=attack,
                num_episodes=num_episodes,
                eval_seeds=eval_seeds,
            )

            seed_returns.append(adv_res["post_attack_return_mean"])
            seed_catches.append(adv_res["catch_rate_mean"])
            total_attacked += attack.total_attacked_count
            total_changed += attack.action_changed_count

        mean_ret = float(np.mean(seed_returns))
        sem_ret = float(np.std(seed_returns, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
        mean_cat = float(np.mean(seed_catches))
        sem_cat = float(np.std(seed_catches, ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
        act_frac = float(total_changed / max(1, total_attacked))

        table_rows.append({
            "k": k,
            "mean_ret": mean_ret,
            "sem_ret": sem_ret,
            "mean_cat": mean_cat,
            "sem_cat": sem_cat,
            "action_frac": act_frac,
            "seed_returns": seed_returns,
        })

        logger.info(
            "k=%2d | Return: %6.2f +/- %4.2f | Catches: %4.2f +/- %4.2f | Action Frac: %5.1f%%",
            k, mean_ret, sem_ret, mean_cat, sem_cat, act_frac * 100
        )

    print("\n=== ATTACK-SURFACE PROBE SUMMARY TABLE (eps=0.05, 30 ep/seed) ===")
    print("| k | Mean Return +/- SEM | Catches/Ep +/- SEM | Action-Change Fraction |")
    print("|---|---|---|---|")
    print(f"| 0 (Clean) | {clean_mean:.2f} +/- {clean_sem:.2f} | {catch_mean:.2f} +/- {catch_sem:.2f} | 0.0% |")
    for r in table_rows:
        print(f"| {r['k']:2d} | {r['mean_ret']:.2f} +/- {r['sem_ret']:.2f} | {r['mean_cat']:.2f} +/- {r['sem_cat']:.2f} | {r['action_frac']*100:.1f}% |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple_tag_v3 experimental spike probe.")
    parser.add_argument("--config", type=str, default="configs/spike_simple_tag.yaml", help="Path to spike config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_spike(args.config)


if __name__ == "__main__":
    main()
