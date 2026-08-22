"""Resumable sequential experiment sweep harness runner (GEMINI.md §6 & §8).

Usage:
    python -m admarl.training.sweep --config configs/sweep.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from admarl.attacks.factory import get_attack
from admarl.eval.evaluator import evaluate_policy
from admarl.training.train import Trainer
from admarl.utils.config import load_config
from admarl.utils.logger import ExperimentLogger
from admarl.utils.memory import handle_cuda_oom

logger = logging.getLogger(__name__)


def run_sweep(sweep_config_path: Path | str) -> list[Path]:
    """Execute resumable experiment sweep grid sequentially.

    Args:
        sweep_config_path: Path to sweep YAML configuration file

    Returns:
        List of completed run directory paths
    """
    raw_cfg = load_config(sweep_config_path)
    sweep_cfg = raw_cfg.get("sweep", raw_cfg)

    base_out_dir = Path(sweep_cfg.get("output_dir", "runs/sweep_phase6"))
    base_out_dir.mkdir(parents=True, exist_ok=True)

    arms: dict[str, Any] = sweep_cfg.get("arms", {})
    budgets: list[int] = sweep_cfg.get("budgets", [0, 5])
    epsilons: list[float] = sweep_cfg.get("epsilons", [0.05])
    seeds: list[int] = sweep_cfg.get("seeds", [0])

    eval_cfg = sweep_cfg.get("eval", {})
    env_cfg = sweep_cfg.get("env", {})
    train_cfg = sweep_cfg.get("train", {})

    completed_run_dirs: list[Path] = []

    total_cells = len(arms) * len(budgets) * len(epsilons) * len(seeds)
    cell_index = 0

    for arm_name, arm_cfg in arms.items():
        for k in budgets:
            for eps in epsilons:
                for seed in seeds:
                    cell_index += 1
                    run_folder_name = f"{arm_name}_k{k}_eps{eps}_seed{seed}"
                    cell_run_dir = base_out_dir / run_folder_name

                    final_ckpt = cell_run_dir / "checkpoints" / "final.pt"

                    # 1. Skip logic for completed runs (GEMINI.md §6)
                    if final_ckpt.exists():
                        logger.info(
                            "[%d/%d] Skipping already completed run: %s",
                            cell_index,
                            total_cells,
                            cell_run_dir,
                        )
                        completed_run_dirs.append(cell_run_dir)
                        continue

                    logger.info(
                        "[%d/%d] Starting cell: arm=%s, k=%d, eps=%.3f, seed=%d",
                        cell_index,
                        total_cells,
                        arm_name,
                        k,
                        eps,
                        seed,
                    )

                    # Build concrete cell config
                    cell_config: dict[str, Any] = {
                        "seed": seed,
                        "deterministic": True,
                        "hardware": {"device": "cuda" if torch.cuda.is_available() else "cpu", "max_vram_gb": 5.0},
                        "env": env_cfg,
                        "model": {"hidden_dim": 64, "num_layers": 2},
                        "train": train_cfg,
                        "defense": arm_cfg.get("defense", {"name": "none", "penalty_coeff": 0.0}),
                        "adv_training": arm_cfg.get("adv_training", {"enabled": False}),
                        "attack": {
                            "name": "critic_sensitivity" if k > 0 else "none",
                            "budget_k": k,
                            "epsilon": eps,
                            "norm": "linf",
                        },
                        "sweep_meta": {
                            "arm": arm_name,
                            "budget_k": k,
                            "epsilon": eps,
                        },
                    }

                    try:
                        # 2. Initialize ExperimentLogger for this cell
                        cell_run_dir.mkdir(parents=True, exist_ok=True)
                        cell_config_path = cell_run_dir / "resolved_config.yaml"

                        with open(cell_config_path, "w", encoding="utf-8") as f:
                            yaml.dump(cell_config, f)

                        exp_logger = ExperimentLogger(config=cell_config, base_output_dir=base_out_dir)

                        # 3. Train policy
                        trainer = Trainer(config=cell_config, exp_logger=exp_logger)

                        # Resume support if interrupted
                        latest_ckpt = cell_run_dir / "checkpoints" / "latest.pt"
                        if latest_ckpt.exists() and not final_ckpt.exists():
                            logger.info("Resuming interrupted run from %s", latest_ckpt)

                        trainer.train()

                        # 4. Evaluate policy under Phase 3 attack
                        attack = get_attack(cell_config)
                        eval_res = evaluate_policy(
                            mappo=trainer.mappo,
                            env_name=env_cfg.get("name", "simple_spread"),
                            episode_length=env_cfg.get("episode_length", 25),
                            attack=attack,
                            num_episodes=eval_cfg.get("num_episodes", 10),
                            eval_seeds=eval_cfg.get("eval_seeds", [1000 + i for i in range(10)]),
                        )

                        # Save evaluation results to eval_results.json
                        eval_json_path = cell_run_dir / "eval_results.json"
                        eval_payload = {
                            "post_attack_return_mean": eval_res["post_attack_return_mean"],
                            "post_attack_return_std": eval_res["post_attack_return_std"],
                            "budget_k": k,
                            "epsilon": eps,
                        }
                        with open(eval_json_path, "w", encoding="utf-8") as f:
                            json.dump(eval_payload, f, indent=2)

                        completed_run_dirs.append(cell_run_dir)

                    except RuntimeError as e:
                        handle_cuda_oom(e, context=f"Sweep Cell {run_folder_name}")
                        logger.error("Error executing cell %s: %s", run_folder_name, e)

                    finally:
                        # Memory cleanup between sequential GPU runs (GEMINI.md §2 & §7)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    logger.info("Sweep completed cleanly! Ran/validated %d runs.", len(completed_run_dirs))
    return completed_run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment sweep grid sequentially.")
    parser.add_argument("--config", type=str, default="configs/sweep.yaml", help="Path to sweep config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_sweep(args.config)


if __name__ == "__main__":
    main()
