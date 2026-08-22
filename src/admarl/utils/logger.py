"""Structured experiment logging and run directory artifact management (GEMINI.md §8)."""
from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from admarl.utils.memory import get_memory_info, log_memory_status

logger = logging.getLogger(__name__)


def get_git_metadata() -> dict[str, Any]:
    """Retrieve current git commit hash and dirty status for reproducibility."""
    try:
        import subprocess

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        dirty_str = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return {
            "commit": commit,
            "short_commit": commit[:7],
            "is_dirty": len(dirty_str) > 0,
        }
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return {
            "commit": "unknown",
            "short_commit": "nohash",
            "is_dirty": False,
        }


class ExperimentLogger:
    """Manages run directory artifacts, dual metrics logging (CSV + TensorBoard), and experiment tracking."""

    CSV_HEADER: ClassVar[list[str]] = [
        "step",
        "episode_return_mean",
        "episode_return_std",
        "critic_loss",
        "policy_loss",
        "entropy",
        "regularizer_penalty",
        "adv_reg_loss",
        "train_epsilon",
        "post_attack_return",
        "attack_budget",
        "epsilon",
        "vram_allocated_gb",
        "vram_reserved_gb",
    ]

    def __init__(
        self,
        config: dict[str, Any],
        base_output_dir: str | Path = "runs",
        experiment_log_file: str | Path = "EXPERIMENTS.md",
    ) -> None:
        self.config = config
        self.base_output_dir = Path(base_output_dir)
        self.experiment_log_file = Path(experiment_log_file)

        git_meta = get_git_metadata()
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        seed = config.get("seed", 0)
        short_hash = git_meta["short_commit"]
        if git_meta["is_dirty"]:
            short_hash += "-dirty"

        # Directory structure: runs/<YYYYMMDD-HHMMSS>-<short-git-hash>-seed<N>/
        run_name = f"{timestamp}-{short_hash}-seed{seed}"
        self.run_dir = self.base_output_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._setup_run_log()
        self._write_meta_json(git_meta, timestamp)
        self._write_resolved_config()
        self._snapshot_lockfile()

        # CSV Logging Setup (file kept open until finish() is called)
        self.csv_path = self.run_dir / "metrics.csv"
        self._csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")  # noqa: SIM115
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_HEADER)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        # TensorBoard Logging Setup
        self.tb_dir = self.run_dir / "tensorboard"
        self.tb_writer = SummaryWriter(log_dir=str(self.tb_dir))

        self.last_metrics: dict[str, float] = {}

    def _setup_run_log(self) -> None:
        """Configure Python logging output to run.log in the run directory."""
        log_file = self.run_dir / "run.log"
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

        self._file_handler = logging.FileHandler(log_file, encoding="utf-8")
        self._file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(self._file_handler)

    def _write_meta_json(self, git_meta: dict[str, Any], timestamp: str) -> None:
        """Write meta.json containing hardware, software, and run metadata."""
        gpu_name = "N/A"
        cuda_version = "N/A"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda or "N/A"

        meta_data = {
            "git_commit": git_meta["commit"],
            "is_dirty": git_meta["is_dirty"],
            "seed": self.config.get("seed", 0),
            "device": self.config.get("hardware", {}).get("device", "cpu"),
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "cuda_version": cuda_version,
            "n_parallel_envs": self.config.get("env", {}).get("n_parallel_envs", 1),
            "start_time": datetime.now(UTC).isoformat(),
            "command_line": " ".join(sys.argv),
        }

        meta_path = self.run_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

    def _write_resolved_config(self) -> None:
        """Write resolved_config.yaml snapshot."""
        config_path = self.run_dir / "resolved_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def _snapshot_lockfile(self) -> None:
        """Snapshot dependency lockfile (uv.lock) if present in root."""
        lockfile_path = Path("uv.lock")
        if lockfile_path.exists():
            shutil.copy(lockfile_path, self.run_dir / "uv.lock")

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        """Log metrics to metrics.csv, TensorBoard, and Python logger (GEMINI.md §8)."""
        self.last_metrics = metrics.copy()
        mem_info = get_memory_info()

        row: dict[str, Any] = {
            "step": step,
            "episode_return_mean": metrics.get("episode_return_mean", 0.0),
            "episode_return_std": metrics.get("episode_return_std", 0.0),
            "critic_loss": metrics.get("critic_loss", 0.0),
            "policy_loss": metrics.get("policy_loss", metrics.get("actor_loss", 0.0)),
            "entropy": metrics.get("entropy", 0.0),
            "regularizer_penalty": metrics.get("regularizer_penalty", metrics.get("reg_loss", 0.0)),
            "post_attack_return": metrics.get("post_attack_return", 0.0),
            "attack_budget": metrics.get("attack_budget", self.config.get("attack", {}).get("budget_fraction", 0.0)),
            "epsilon": metrics.get("epsilon", self.config.get("attack", {}).get("epsilon", 0.0)),
            "vram_allocated_gb": mem_info.get("vram_allocated_gb", 0.0),
            "vram_reserved_gb": mem_info.get("vram_reserved_gb", 0.0),
        }

        # CSV Logging
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # TensorBoard Logging
        for k, v in row.items():
            if k != "step":
                self.tb_writer.add_scalar(f"train/{k}", v, step)

        # Python Console Logger
        logger.info(
            "Step: %d | Return: %.2f ± %.2f | Policy Loss: %.4f | Critic Loss: %.4f | Entropy: %.4f",
            step,
            row["episode_return_mean"],
            row["episode_return_std"],
            row["policy_loss"],
            row["critic_loss"],
            row["entropy"],
        )

        log_memory_status(prefix="Periodic: ")

    def finish(self, final_notes: str = "Completed successfully") -> None:
        """Close log handles and append summary row to EXPERIMENTS.md."""
        if hasattr(self, "_file_handler"):
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()

        if not self._csv_file.closed:
            self._csv_file.close()
        self.tb_writer.close()

        self._append_to_experiments_md(final_notes)

    def _append_to_experiments_md(self, notes: str) -> None:
        """Append run summary row to EXPERIMENTS.md."""
        meta_path = self.run_dir / "meta.json"
        git_commit = "unknown"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                git_commit = meta.get("git_commit", "unknown")[:7]

        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        env_name = self.config.get("env", {}).get("name", "unknown")
        seed = self.config.get("seed", 0)
        attack_budget = self.config.get("attack", {}).get("budget_fraction", 0.0)

        ret_mean = self.last_metrics.get("episode_return_mean", 0.0)
        ret_std = self.last_metrics.get("episode_return_std", 0.0)
        metric_str = f"{ret_mean:.2f} ± {ret_std:.2f}"

        row_str = (
            f"| {self.run_dir.name} | {date_str} | {git_commit} | "
            f"base.yaml | {seed} | {env_name} | {attack_budget} | {metric_str} | {notes} |\n"
        )

        if not self.experiment_log_file.exists():
            with open(self.experiment_log_file, "w", encoding="utf-8") as f:
                f.write("# Experiment Log\n\n")
                f.write("| Run ID | Date | Branch/Commit | Config | Seeds | Env | Attack budget | Key metric (mean ± std) | Notes |\n")
                f.write("|--------|------|---------------|--------|-------|-----|---------------|-------------------------|-------|\n")

        with open(self.experiment_log_file, "a", encoding="utf-8") as f:
            f.write(row_str)

        logger.info("Appended summary row to %s", self.experiment_log_file)
