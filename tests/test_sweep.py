"""Unit tests for sweep harness, resume/skip logic, aggregation, plotting, and smoke sweep (GEMINI.md §9)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from admarl.eval.aggregate import aggregate_sweep_results
from admarl.eval.plot import generate_robustness_plots
from admarl.training.sweep import run_sweep
from admarl.utils.config import load_config


def test_sweep_config_validation() -> None:
    """Test sweep config loading and structural validity."""
    config = load_config("configs/sweep.yaml")
    assert "sweep" in config
    sweep_cfg = config["sweep"]

    assert "arms" in sweep_cfg
    assert "none" in sweep_cfg["arms"]
    assert "lipschitz" in sweep_cfg["arms"]
    assert "sa_ppo" in sweep_cfg["arms"]

    assert isinstance(sweep_cfg["budgets"], list)
    assert 0 in sweep_cfg["budgets"]  # Clean anchor requirement
    assert isinstance(sweep_cfg["epsilons"], list)
    assert len(sweep_cfg["seeds"]) >= 3  # Multi-seed requirement


def test_aggregation_correctness() -> None:
    """Test results aggregation from synthetic run directories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Create synthetic run directories for 2 seeds of 'none' arm
        for seed in [0, 1]:
            run_dir = tmp_path / f"none_k5_eps0.05_seed{seed}"
            run_dir.mkdir(parents=True)

            # Write config
            cfg = {
                "seed": seed,
                "sweep_meta": {"arm": "none"},
                "attack": {"budget_k": 5, "epsilon": 0.05},
            }
            with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(cfg, f)

            # Write metrics.csv
            df = pd.DataFrame([
                {
                    "step": 100,
                    "episode_return_mean": -10.0 + seed,
                    "critic_loss": 0.5,
                    "policy_loss": 0.2,
                    "entropy": 0.1,
                    "regularizer_penalty": 0.0,
                    "post_attack_return": -15.0 + seed,
                }
            ])
            df.to_csv(run_dir / "metrics.csv", index=False)

        # Aggregate results
        aggregated = aggregate_sweep_results(tmp_path)
        summary = aggregated.get("summary", {})

        key = "none_k5_eps0.05"
        assert key in summary
        stat = summary[key]
        assert stat["num_seeds"] == 2
        assert stat["mean"] == pytest.approx(-14.5)  # Mean of -15.0 and -14.0
        assert stat["std"] > 0.0


def test_plot_generation_synthetic() -> None:
    """Test generate_robustness_plots end-to-end on synthetic run metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        fig_dir = tmp_path / "figures"
        sweep_dir = tmp_path / "runs"

        # Create synthetic runs for none, lipschitz, sa_ppo
        for arm in ["none", "lipschitz", "sa_ppo"]:
            for k in [0, 5]:
                run_dir = sweep_dir / f"{arm}_k{k}_eps0.05_seed0"
                run_dir.mkdir(parents=True)

                cfg = {
                    "seed": 0,
                    "sweep_meta": {"arm": arm},
                    "attack": {"budget_k": k, "epsilon": 0.05},
                }
                with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)

                df = pd.DataFrame([
                    {
                        "step": 100,
                        "episode_return_mean": -10.0,
                        "critic_loss": 0.5,
                        "policy_loss": 0.2,
                        "entropy": 0.1,
                        "regularizer_penalty": 0.0,
                        "post_attack_return": -12.0 if k == 0 else -18.0,
                    }
                ])
                df.to_csv(run_dir / "metrics.csv", index=False)

        generated = generate_robustness_plots(sweep_dir=sweep_dir, output_dir=fig_dir)
        assert len(generated) >= 3

        for fig_file in generated:
            assert fig_file.exists()


def test_resume_skip_logic() -> None:
    """Test that completed runs with final.pt are skipped by the sweep harness."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create mock sweep config with 1 run
        sweep_cfg = {
            "sweep": {
                "name": "test_skip",
                "output_dir": str(tmp_path),
                "arms": {
                    "none": {
                        "defense": {"name": "none", "penalty_coeff": 0.0},
                        "adv_training": {"enabled": False},
                    }
                },
                "budgets": [0],
                "epsilons": [0.05],
                "seeds": [0],
                "eval": {"num_episodes": 2, "eval_seeds": [100]},
                "env": {"name": "simple_spread", "n_parallel_envs": 1, "episode_length": 5},
                "train": {"total_steps": 20, "n_steps": 10, "batch_size": 16},
            }
        }

        sweep_cfg_file = tmp_path / "test_sweep.yaml"
        with open(sweep_cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(sweep_cfg, f)

        # Create mock completed run folder
        completed_folder = tmp_path / "none_k0_eps0.05_seed0" / "checkpoints"
        completed_folder.mkdir(parents=True)
        (completed_folder / "final.pt").touch()

        # Run sweep
        completed_dirs = run_sweep(sweep_cfg_file)
        assert len(completed_dirs) == 1
        assert completed_dirs[0].name == "none_k0_eps0.05_seed0"


def test_minimal_2arm_smoke_sweep() -> None:
    """End-to-end smoke sweep running a minimal 2-arm grid on CPU."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        sweep_cfg = {
            "sweep": {
                "name": "smoke_sweep",
                "output_dir": str(tmp_path / "sweep_out"),
                "arms": {
                    "none": {
                        "defense": {"name": "none", "penalty_coeff": 0.0},
                        "adv_training": {"enabled": False},
                    },
                    "lipschitz": {
                        "defense": {"name": "grad_penalty", "penalty_coeff": 0.1, "norm": "l2"},
                        "adv_training": {"enabled": False},
                    },
                },
                "budgets": [0],
                "epsilons": [0.05],
                "seeds": [0],
                "eval": {"num_episodes": 2, "eval_seeds": [1000, 1001]},
                "env": {"name": "simple_spread", "n_parallel_envs": 1, "episode_length": 5},
                "train": {
                    "total_steps": 20,
                    "n_steps": 10,
                    "batch_size": 16,
                    "n_epochs": 1,
                    "log_interval_steps": 10,
                },
            }
        }

        sweep_cfg_file = tmp_path / "smoke_sweep.yaml"
        with open(sweep_cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(sweep_cfg, f)

        completed = run_sweep(sweep_cfg_file)
        assert len(completed) == 2

        # Verify figures generation
        fig_files = generate_robustness_plots(sweep_dir=tmp_path / "sweep_out", output_dir=tmp_path / "figs")
        assert len(fig_files) >= 3
