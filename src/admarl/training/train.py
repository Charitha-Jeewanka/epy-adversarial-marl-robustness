"""Training loop for MAPPO with atomic checkpointing and resume support (GEMINI.md §6 & §8)."""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from admarl.algos.mappo import MAPPO
from admarl.envs.vector_env import VectorMARLEnv
from admarl.training.rollout import RolloutBuffer
from admarl.utils.checkpoint import load_checkpoint, restore_rng_from_checkpoint, save_checkpoint
from admarl.utils.config import load_config
from admarl.utils.memory import handle_cuda_oom, log_memory_status
from admarl.utils.seed import set_seed

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path) -> None:
    """Configure logging to stdout and run log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_git_commit_hash() -> dict[str, Any]:
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
        return {"commit": commit, "is_dirty": len(dirty_str) > 0}
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return {"commit": "unknown", "is_dirty": False}


class Trainer:
    """MAPPO Trainer managing training loop, checkpointing, and evaluation logging."""

    def __init__(self, config: dict[str, Any], run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.interrupted = False

        # Set seed & determinism
        set_seed(config["seed"], deterministic=config.get("deterministic", False))

        # Device selection
        req_device = config["hardware"].get("device", "cpu")
        if req_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info("Using compute device: %s", self.device)

        # Environments
        env_cfg = config["env"]
        self.n_envs = env_cfg.get("n_parallel_envs", 8)
        self.episode_length = env_cfg.get("episode_length", 25)
        self.vec_env = VectorMARLEnv(
            env_name=env_cfg["name"],
            n_envs=self.n_envs,
            max_cycles=self.episode_length,
        )

        self.num_agents = self.vec_env.num_agents
        self.obs_dim = self.vec_env.obs_dim
        self.state_dim = self.vec_env.state_dim
        self.action_dim = self.vec_env.action_dim

        # Algorithm
        train_cfg = config["train"]
        model_cfg = config.get("model", {})
        self.mappo = MAPPO(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            actor_lr=train_cfg.get("actor_lr", 5e-4),
            critic_lr=train_cfg.get("critic_lr", 5e-4),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_layers=model_cfg.get("num_layers", 2),
            clip_param=train_cfg.get("clip_param", 0.2),
            value_loss_coef=train_cfg.get("value_loss_coef", 0.5),
            entropy_coef=train_cfg.get("entropy_coef", 0.01),
            max_grad_norm=train_cfg.get("grad_clip", 10.0),
            device=self.device,
        )

        # Rollout Buffer
        self.n_steps = train_cfg.get("n_steps", 128)
        self.buffer = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            device=self.device,
        )

        # Tracking state
        self.total_steps = train_cfg.get("total_steps", 1000000)
        self.batch_size = train_cfg.get("batch_size", 1024)
        self.n_epochs = train_cfg.get("n_epochs", 10)
        self.gamma = train_cfg.get("gamma", 0.99)
        self.gae_lambda = train_cfg.get("gae_lambda", 0.95)

        self.checkpoint_every_steps = train_cfg.get("checkpoint_every_steps", 20000)
        self.checkpoint_every_minutes = train_cfg.get("checkpoint_every_minutes", 15)
        self.log_interval_steps = train_cfg.get("log_interval_steps", 1000)

        self.current_step = 0
        self.current_episode = 0
        self.last_ckpt_time = time.time()
        self.last_ckpt_step = 0

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    def register_signal_handlers(self) -> None:
        """Register graceful interruption handlers (GEMINI.md §6)."""
        def handle_signal(sig: int, frame: Any) -> None:
            logger.warning("Received signal %d. Triggering graceful shutdown and checkpointing...", sig)
            self.interrupted = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def save_checkpoint(self, is_final: bool = False) -> Path:
        """Save training state atomically."""
        filename = "final.pt" if is_final else "latest.pt"
        ckpt_path = save_checkpoint(
            checkpoint_dir=self.ckpt_dir,
            filename=filename,
            model_state=self.mappo.state_dict(),
            optimizer_state=self.mappo.optimizer_state_dict(),
            step=self.current_step,
            episode=self.current_episode,
            config=self.config,
        )
        if not is_final and self.current_step > 0:
            step_filename = f"step_{self.current_step}.pt"
            save_checkpoint(
                checkpoint_dir=self.ckpt_dir,
                filename=step_filename,
                model_state=self.mappo.state_dict(),
                optimizer_state=self.mappo.optimizer_state_dict(),
                step=self.current_step,
                episode=self.current_episode,
                config=self.config,
            )
        self.last_ckpt_step = self.current_step
        self.last_ckpt_time = time.time()
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume training bit-for-bit from checkpoint."""
        data = load_checkpoint(checkpoint_path, device=self.device)
        self.mappo.load_state_dict(data["model_state"])
        self.mappo.load_optimizer_state_dict(data["optimizer_state"])
        restore_rng_from_checkpoint(data)

        self.current_step = data["step"]
        self.current_episode = data.get("episode", 0)
        self.last_ckpt_step = self.current_step
        self.last_ckpt_time = time.time()
        logger.info(
            "Resumed training from step %d, episode %d (checkpoint: %s)",
            self.current_step,
            self.current_episode,
            checkpoint_path,
        )

    def train(self) -> None:
        """Main training loop."""
        self.register_signal_handlers()
        logger.info("Starting training loop up to %d total steps...", self.total_steps)
        log_memory_status("Start of training: ")

        obs, state = self.vec_env.reset()
        episode_returns = np.zeros(self.n_envs, dtype=np.float32)
        recent_returns: list[float] = []

        last_log_step = self.current_step

        try:
            while self.current_step < self.total_steps and not self.interrupted:
                self.buffer.reset()

                # Rollout Collection
                for _ in range(self.n_steps):
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

                    actions, log_probs, values = self.mappo.get_actions_and_values(
                        obs_tensor, state_tensor
                    )

                    actions_np = actions.cpu().numpy()
                    next_obs, next_state, rewards, dones, _infos = self.vec_env.step(actions_np)

                    # Track episode returns (sum reward of first agent as cooperative return metric)
                    episode_returns += rewards[:, 0]
                    for env_idx, d in enumerate(dones):
                        if np.all(d):
                            recent_returns.append(float(episode_returns[env_idx]))
                            episode_returns[env_idx] = 0.0
                            self.current_episode += 1

                    self.buffer.insert(
                        obs=obs,
                        state=state,
                        actions=actions,
                        log_probs=log_probs,
                        rewards=rewards,
                        values=values,
                        dones=dones,
                    )

                    obs = next_obs
                    state = next_state
                    self.current_step += self.n_envs * self.num_agents

                # GAE computation
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                _, _, last_values = self.mappo.get_actions_and_values(obs_tensor, state_tensor)

                last_dones = np.zeros((self.n_envs, self.num_agents), dtype=bool)
                self.buffer.compute_returns_and_advantages(
                    last_values=last_values,
                    last_dones=last_dones,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                )

                # MAPPO Update
                update_metrics: dict[str, float] = {}
                for epoch in range(self.n_epochs):
                    for batch in self.buffer.mini_batch_generator(self.batch_size):
                        obs_b, state_b, actions_b, old_log_probs_b, returns_b, advantages_b = batch
                        metrics = self.mappo.update(
                            obs_b=obs_b,
                            state_b=state_b,
                            actions_b=actions_b,
                            old_log_probs_b=old_log_probs_b,
                            returns_b=returns_b,
                            advantages_b=advantages_b,
                        )
                        update_metrics = metrics

                # Logging
                if self.current_step - last_log_step >= self.log_interval_steps:
                    mean_return = np.mean(recent_returns) if len(recent_returns) > 0 else 0.0
                    recent_returns.clear()

                    logger.info(
                        "Step: %d/%d | Episode: %d | Mean Return: %.2f | Actor Loss: %.4f | Critic Loss: %.4f",
                        self.current_step,
                        self.total_steps,
                        self.current_episode,
                        mean_return,
                        update_metrics.get("actor_loss", 0.0),
                        update_metrics.get("critic_loss", 0.0),
                    )

                    self.writer.add_scalar("train/mean_episode_return", mean_return, self.current_step)
                    for k, v in update_metrics.items():
                        self.writer.add_scalar(f"train/{k}", v, self.current_step)

                    last_log_step = self.current_step

                # Checkpointing logic
                elapsed_minutes = (time.time() - self.last_ckpt_time) / 60.0
                steps_since_ckpt = self.current_step - self.last_ckpt_step
                if (
                    steps_since_ckpt >= self.checkpoint_every_steps
                    or elapsed_minutes >= self.checkpoint_every_minutes
                ):
                    self.save_checkpoint()

        except RuntimeError as e:
            handle_cuda_oom(e, context="MAPPO Training Loop")

        finally:
            # Atomic checkpoint on exit / interrupt
            logger.info("Cleaning up training resources...")
            self.save_checkpoint(is_final=True)
            self.writer.close()
            self.vec_env.close()
            log_memory_status("End of training: ")
            logger.info("Training completed cleanly at step %d.", self.current_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO baseline in cooperative MARL.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to run dir or checkpoint file to resume")
    parser.add_argument("--output-dir", type=str, default="runs", help="Output directory for runs")
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve run directory
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            run_dir = resume_path
            ckpt_file = run_dir / "checkpoints" / "latest.pt"
        else:
            run_dir = resume_path.parent.parent
            ckpt_file = resume_path
    else:
        env_name = config["env"]["name"]
        seed = config["seed"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / f"{env_name}_seed{seed}_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")

    # Write reproducibility artifacts (GEMINI.md §8)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    git_info = get_git_commit_hash()
    with open(run_dir / "git_info.json", "w", encoding="utf-8") as f:
        json.dump(git_info, f, indent=2)

    trainer = Trainer(config=config, run_dir=run_dir)

    if args.resume:
        trainer.load_checkpoint(ckpt_file)

    trainer.train()


if __name__ == "__main__":
    main()
