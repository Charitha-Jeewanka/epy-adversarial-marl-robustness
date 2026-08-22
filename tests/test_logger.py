"""Unit tests for ExperimentLogger and run directory artifact generation (GEMINI.md §8 & §9)."""
from __future__ import annotations

import csv
import json
import re
import tempfile
from pathlib import Path

import yaml

from admarl.utils.config import load_config
from admarl.utils.logger import ExperimentLogger, get_git_metadata


def test_git_metadata() -> None:
    """Test retrieving git metadata dictionary."""
    git_meta = get_git_metadata()
    assert "commit" in git_meta
    assert "short_commit" in git_meta
    assert "is_dirty" in git_meta


def test_experiment_logger_artifacts() -> None:
    """Verify that ExperimentLogger produces all required run_dir artifacts and headers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        exp_md_path = tmp_path / "EXPERIMENTS.md"

        config = {
            "seed": 42,
            "hardware": {"device": "cpu", "max_vram_gb": 5.0},
            "env": {"name": "simple_spread", "n_parallel_envs": 2},
            "train": {"total_steps": 100},
            "attack": {"budget_fraction": 0.1, "epsilon": 0.05},
        }

        exp_logger = ExperimentLogger(
            config=config,
            base_output_dir=tmp_path,
            experiment_log_file=exp_md_path,
        )

        run_dir = exp_logger.run_dir
        assert run_dir.exists()
        assert re.search(r"\d{8}-\d{6}-.+-seed42", run_dir.name) is not None

        # 1. Verify resolved_config.yaml
        resolved_cfg_file = run_dir / "resolved_config.yaml"
        assert resolved_cfg_file.exists()
        with open(resolved_cfg_file, "r", encoding="utf-8") as f:
            loaded_cfg = yaml.safe_load(f)
        assert loaded_cfg["seed"] == 42
        assert loaded_cfg["env"]["name"] == "simple_spread"

        # 2. Verify meta.json
        meta_file = run_dir / "meta.json"
        assert meta_file.exists()
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        required_meta_keys = [
            "git_commit",
            "is_dirty",
            "seed",
            "device",
            "gpu_name",
            "torch_version",
            "cuda_version",
            "n_parallel_envs",
            "start_time",
            "command_line",
        ]
        for key in required_meta_keys:
            assert key in meta, f"Missing key '{key}' in meta.json"

        # 3. Verify metrics.csv header
        metrics_csv = run_dir / "metrics.csv"
        assert metrics_csv.exists()
        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        expected_header = ExperimentLogger.CSV_HEADER
        assert header == expected_header

        # 4. Log step and finish
        exp_logger.log_step(
            step=10,
            metrics={
                "episode_return_mean": -15.5,
                "episode_return_std": 2.1,
                "actor_loss": 0.12,
                "critic_loss": 0.45,
                "entropy": 1.1,
            },
        )
        exp_logger.finish(final_notes="Test completed")

        # Verify logged row in CSV
        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert int(rows[0]["step"]) == 10
            assert float(rows[0]["episode_return_mean"]) == -15.5

        # 5. Verify EXPERIMENTS.md row appended
        assert exp_md_path.exists()
        with open(exp_md_path, "r", encoding="utf-8") as f:
            exp_content = f.read()
        assert run_dir.name in exp_content
        assert "Test completed" in exp_content


def test_trainer_integration_artifacts() -> None:
    """Test that a short training run produces the expected run directory artifacts via Trainer."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        exp_md_path = tmp_path / "EXPERIMENTS.md"

        config = load_config("configs/base.yaml")
        config["hardware"]["device"] = "cpu"
        config["env"]["n_parallel_envs"] = 2
        config["train"]["total_steps"] = 40
        config["train"]["n_steps"] = 10
        config["train"]["batch_size"] = 16
        config["train"]["n_epochs"] = 1
        config["train"]["log_interval_steps"] = 20

        exp_logger = ExperimentLogger(
            config=config, base_output_dir=tmp_path, experiment_log_file=exp_md_path
        )
        from admarl.training.train import Trainer

        trainer = Trainer(config=config, exp_logger=exp_logger)
        trainer.train()

        run_dir = exp_logger.run_dir
        assert (run_dir / "resolved_config.yaml").exists()
        assert (run_dir / "meta.json").exists()
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "checkpoints" / "final.pt").exists()
        assert exp_md_path.exists()
