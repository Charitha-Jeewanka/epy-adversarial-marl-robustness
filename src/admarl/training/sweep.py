"""Resumable sequential experiment sweep harness runner (GEMINI.md §6 & §8).

Usage:
    python -m admarl.training.sweep --config configs/sweep.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from admarl.algos.mappo import MAPPO
from admarl.attacks.critic_sensitivity import CriticSensitivityAttack
from admarl.eval.evaluator import evaluate_policy
from admarl.training.train import Trainer
from admarl.utils.config import load_config
from admarl.utils.logger import ExperimentLogger
from admarl.utils.memory import handle_cuda_oom

logger = logging.getLogger(__name__)


def run_sweep(sweep_config_path: Path | str) -> list[Path]:
    """Execute resumable experiment sweep grid sequentially with train/eval separation (GEMINI.md §8).

    Stage 1: Train base models once per arm and seed (3 arms x N seeds).
    Stage 2: Evaluate fixed base model weights across the test attack grid (k x epsilon).

    Args:
        sweep_config_path: Path to sweep YAML configuration file

    Returns:
        List of completed evaluation directory paths
    """
    raw_cfg = load_config(sweep_config_path)
    sweep_cfg = raw_cfg.get("sweep", raw_cfg)

    base_out_dir = Path(sweep_cfg.get("output_dir", "runs/sweep_phase6"))
    models_dir = base_out_dir / "models"
    eval_dir = base_out_dir / "eval"

    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    arms: dict[str, Any] = sweep_cfg.get("arms", {})
    budgets: list[int] = sweep_cfg.get("budgets", [0, 1, 3, 5, 10])
    epsilons: list[float] = sweep_cfg.get("epsilons", [0.01, 0.05, 0.10])
    seeds: list[int] = sweep_cfg.get("seeds", [0, 1, 2])

    eval_cfg = sweep_cfg.get("eval", {})
    env_cfg = sweep_cfg.get("env", {})
    train_cfg = sweep_cfg.get("train", {})

    # =========================================================================
    # STAGE 1: Train Base Models (3 arms x N seeds)
    # =========================================================================
    total_models = len(arms) * len(seeds)
    model_idx = 0

    for arm_name, arm_cfg in arms.items():
        for seed in seeds:
            model_idx += 1
            model_folder = models_dir / f"{arm_name}_seed{seed}"
            final_ckpt = model_folder / "checkpoints" / "final.pt"

            # Check if pre-existing trained checkpoint exists in legacy root sweep directory
            if not final_ckpt.exists():
                legacy_candidates = list(base_out_dir.rglob(f"{arm_name}_k0_*seed{seed}/checkpoints/final.pt"))
                if not legacy_candidates:
                    legacy_candidates = list(base_out_dir.rglob(f"*{arm_name}*seed{seed}/checkpoints/final.pt"))
                if legacy_candidates:
                    model_folder.mkdir(parents=True, exist_ok=True)
                    (model_folder / "checkpoints").mkdir(parents=True, exist_ok=True)
                    shutil.copy(legacy_candidates[0], final_ckpt)
                    logger.info("Migrated existing trained checkpoint from %s -> %s", legacy_candidates[0], final_ckpt)

            if final_ckpt.exists():
                logger.info("[%d/%d] Base model ready: %s", model_idx, total_models, model_folder)
                continue

            logger.info("[%d/%d] Training base model: arm=%s, seed=%d", model_idx, total_models, arm_name, seed)

            base_config: dict[str, Any] = {
                "seed": seed,
                "deterministic": True,
                "hardware": {"device": "cuda" if torch.cuda.is_available() else "cpu", "max_vram_gb": 5.0},
                "env": env_cfg,
                "model": {"hidden_dim": 64, "num_layers": 2},
                "train": train_cfg,
                "defense": arm_cfg.get("defense", {"name": "none", "penalty_coeff": 0.0}),
                "adv_training": arm_cfg.get("adv_training", {"enabled": False}),
                "attack": {"name": "none", "budget_k": 0, "epsilon": 0.0},
                "sweep_meta": {"arm": arm_name, "budget_k": 0, "epsilon": 0.0},
            }

            try:
                model_folder.mkdir(parents=True, exist_ok=True)
                with open(model_folder / "resolved_config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(base_config, f)

                exp_logger = ExperimentLogger(config=base_config, base_output_dir=base_out_dir)
                trainer = Trainer(config=base_config, exp_logger=exp_logger)

                latest_ckpt = model_folder / "checkpoints" / "latest.pt"
                if latest_ckpt.exists() and not final_ckpt.exists():
                    logger.info("Resuming interrupted training run from %s", latest_ckpt)

                trainer.train()
                # Copy final checkpoint to model_folder
                if exp_logger.run_dir and (exp_logger.run_dir / "checkpoints" / "final.pt").exists():
                    (model_folder / "checkpoints").mkdir(parents=True, exist_ok=True)
                    shutil.copy(exp_logger.run_dir / "checkpoints" / "final.pt", final_ckpt)

            except RuntimeError as e:
                handle_cuda_oom(e, context=f"Base Model Training {arm_name}_seed{seed}")
                logger.error("Error training base model %s_seed%d: %s", arm_name, seed, e)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 2: Evaluate Fixed Base Models Across Test Attack Grid
    # =========================================================================
    completed_eval_dirs: list[Path] = []
    total_eval_cells = len(arms) * len(budgets) * len(epsilons) * len(seeds)
    eval_idx = 0

    num_episodes = eval_cfg.get("num_episodes", 10)
    eval_seeds = eval_cfg.get("eval_seeds", [1000 + i for i in range(num_episodes)])

    for arm_name in arms:
        for seed in seeds:
            model_folder = models_dir / f"{arm_name}_seed{seed}"
            final_ckpt = model_folder / "checkpoints" / "final.pt"

            if not final_ckpt.exists():
                logger.error("Missing trained checkpoint for %s_seed%d. Skipping eval.", arm_name, seed)
                continue

            # Load fixed base model weights once
            ckpt_data = torch.load(final_ckpt, weights_only=False)
            mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
            mappo.load_state_dict(ckpt_data["model_state"])

            for k in budgets:
                for eps in epsilons:
                    eval_idx += 1
                    cell_eval_dir = eval_dir / f"{arm_name}_k{k}_eps{eps}_seed{seed}"
                    eval_json_path = cell_eval_dir / "eval_results.json"

                    if eval_json_path.exists():
                        logger.info(
                            "[%d/%d] Skipping completed eval: %s",
                            eval_idx,
                            total_eval_cells,
                            cell_eval_dir,
                        )
                        completed_eval_dirs.append(cell_eval_dir)
                        continue

                    logger.info(
                        "[%d/%d] Evaluating fixed model %s_seed%d at k=%d, eps=%.3f",
                        eval_idx,
                        total_eval_cells,
                        arm_name,
                        seed,
                        k,
                        eps,
                    )

                    attack = CriticSensitivityAttack(budget_k=k, epsilon=eps) if k > 0 else None

                    eval_res = evaluate_policy(
                        mappo=mappo,
                        env_name=env_cfg.get("name", "simple_spread"),
                        episode_length=env_cfg.get("episode_length", 25),
                        attack=attack,
                        num_episodes=num_episodes,
                        eval_seeds=eval_seeds,
                    )

                    cell_eval_dir.mkdir(parents=True, exist_ok=True)
                    cell_config = {
                        "seed": seed,
                        "sweep_meta": {
                            "arm": arm_name,
                            "budget_k": k,
                            "epsilon": eps,
                        },
                        "attack": {
                            "name": "critic_sensitivity" if k > 0 else "none",
                            "budget_k": k,
                            "epsilon": eps,
                        },
                    }
                    with open(cell_eval_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
                        yaml.dump(cell_config, f)

                    eval_payload = {
                        "post_attack_return_mean": eval_res["post_attack_return_mean"],
                        "post_attack_return_std": eval_res["post_attack_return_std"],
                        "action_changed_fraction": eval_res.get("action_changed_fraction", 0.0),
                        "budget_k": k,
                        "epsilon": eps,
                    }
                    with open(eval_json_path, "w", encoding="utf-8") as f:
                        json.dump(eval_payload, f, indent=2)

                    completed_eval_dirs.append(cell_eval_dir)

    logger.info("Sweep evaluation completed cleanly! Evaluated %d cells.", len(completed_eval_dirs))
    return completed_eval_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment sweep grid sequentially.")
    parser.add_argument("--config", type=str, default="configs/sweep.yaml", help="Path to sweep config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_sweep(args.config)


if __name__ == "__main__":
    main()
