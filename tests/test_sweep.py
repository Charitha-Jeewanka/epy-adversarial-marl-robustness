"""Unit tests for sweep harness, resume/skip logic, aggregation, plotting, and smoke sweep (GEMINI.md §9)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
    """Test results aggregation from synthetic run directories with completed final.pt checkpoints."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Create synthetic run directories for 2 seeds of 'none' arm
        for seed in [0, 1]:
            run_dir = tmp_path / f"none_k5_eps0.05_seed{seed}"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "final.pt").touch()

            # Write config
            cfg = {
                "seed": seed,
                "sweep_meta": {"arm": "none", "budget_k": 5, "epsilon": 0.05},
                "attack": {"budget_k": 5, "epsilon": 0.05},
            }
            with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(cfg, f)

            # Write eval_results.json
            eval_res = {
                "post_attack_return_mean": -15.0 + seed,
                "post_attack_return_std": 1.0,
                "budget_k": 5,
                "epsilon": 0.05,
            }
            with open(run_dir / "eval_results.json", "w", encoding="utf-8") as f:
                json.dump(eval_res, f)

        # Aggregate results
        aggregated = aggregate_sweep_results(tmp_path)
        summary = aggregated.get("summary", {})

        key = "none_k5_eps0.05"
        assert key in summary
        stat = summary[key]
        assert stat["num_seeds"] == 2
        assert stat["mean"] == pytest.approx(-14.5)  # Mean of -15.0 and -14.0
        assert stat["std"] > 0.0


def test_aggregation_excludes_incomplete_runs() -> None:
    """Test that partial/incomplete runs missing eval_results.json are excluded from aggregation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Valid completed eval run with eval_results.json
        valid_dir = tmp_path / "eval" / "none_k5_eps0.05_seed0"
        valid_dir.mkdir(parents=True)
        with open(valid_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"seed": 0, "sweep_meta": {"arm": "none", "budget_k": 5, "epsilon": 0.05}}, f)
        with open(valid_dir / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump({"post_attack_return_mean": -22.0}, f)

        # 2. Incomplete run missing eval_results.json
        incomplete_dir = tmp_path / "eval" / "none_k5_eps0.05_seed1"
        incomplete_dir.mkdir(parents=True)
        with open(incomplete_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"seed": 1, "sweep_meta": {"arm": "none", "budget_k": 5, "epsilon": 0.05}}, f)

        aggregated = aggregate_sweep_results(tmp_path)
        summary = aggregated.get("summary", {})

        key = "none_k5_eps0.05"
        assert key in summary
        stat = summary[key]
        assert stat["num_seeds"] == 1
        assert stat["seeds"] == [0]
        assert stat["mean"] == pytest.approx(-22.0)


def test_plot_generation_synthetic() -> None:
    """Test generate_robustness_plots end-to-end on synthetic run metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        fig_dir = tmp_path / "figures"
        sweep_dir = tmp_path / "runs"

        # Create synthetic eval runs for none, lipschitz, sa_ppo
        for arm in ["none", "lipschitz", "sa_ppo"]:
            for k in [0, 5]:
                run_dir = sweep_dir / "eval" / f"{arm}_k{k}_eps0.05_seed0"
                run_dir.mkdir(parents=True)

                cfg = {
                    "seed": 0,
                    "sweep_meta": {"arm": arm, "budget_k": k, "epsilon": 0.05},
                    "attack": {"budget_k": k, "epsilon": 0.05},
                }
                with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)

                eval_data = {
                    "post_attack_return_mean": -12.0 if k == 0 else -18.0,
                    "post_attack_return_std": 1.0,
                    "budget_k": k,
                    "epsilon": 0.05,
                }
                with open(run_dir / "eval_results.json", "w", encoding="utf-8") as f:
                    json.dump(eval_data, f)

        generated = generate_robustness_plots(sweep_dir=sweep_dir, output_dir=fig_dir)
        assert len(generated) >= 3

        for fig_file in generated:
            assert fig_file.exists()


def test_resume_skip_logic() -> None:
    """Test that completed runs with eval_results.json are skipped by the sweep harness."""
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

        # Create mock completed model folder and eval folder
        model_folder = tmp_path / "models" / "none_seed0" / "checkpoints"
        model_folder.mkdir(parents=True)
        import torch

        from admarl.algos.mappo import MAPPO
        mock_mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
        torch.save({"model_state": mock_mappo.state_dict()}, model_folder / "final.pt")

        eval_folder = tmp_path / "eval" / "none_k0_eps0.05_seed0"
        eval_folder.mkdir(parents=True)
        with open(eval_folder / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump({"post_attack_return_mean": -20.0}, f)

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
