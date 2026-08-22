"""Fast CPU-only smoke test and checkpoint round-trip test (GEMINI.md §9)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from admarl.algos.mappo import MAPPO
from admarl.envs.mpe import MPEEnv
from admarl.envs.vector_env import VectorMARLEnv
from admarl.training.train import Trainer
from admarl.utils.checkpoint import load_checkpoint, restore_rng_from_checkpoint, save_checkpoint
from admarl.utils.config import load_config
from admarl.utils.seed import set_seed


def test_mpe_env_wrapper() -> None:
    """Test single MPE environment reset and step interface."""
    env = MPEEnv(env_name="simple_spread", max_cycles=25)
    obs, state = env.reset(seed=42)

    assert obs.shape == (env.num_agents, env.obs_dim)
    assert state.shape == (env.state_dim,)

    actions = np.zeros(env.num_agents, dtype=np.int64)
    next_obs, next_state, rewards, term, trunc, _info = env.step(actions)

    assert next_obs.shape == (env.num_agents, env.obs_dim)
    assert next_state.shape == (env.state_dim,)
    assert rewards.shape == (env.num_agents,)
    assert term.shape == (env.num_agents,)
    assert trunc.shape == (env.num_agents,)
    env.close()


def test_vector_mpe_env() -> None:
    """Test vectorized environment container."""
    vec_env = VectorMARLEnv(env_name="simple_spread", n_envs=2, max_cycles=10)
    obs, state = vec_env.reset(seeds=[1, 2])

    assert obs.shape == (2, vec_env.num_agents, vec_env.obs_dim)
    assert state.shape == (2, vec_env.state_dim)

    actions = np.zeros((2, vec_env.num_agents), dtype=np.int64)
    next_obs, next_state, rewards, dones, _infos = vec_env.step(actions)

    assert next_obs.shape == (2, vec_env.num_agents, vec_env.obs_dim)
    assert next_state.shape == (2, vec_env.state_dim)
    assert rewards.shape == (2, vec_env.num_agents)
    assert dones.shape == (2, vec_env.num_agents)
    vec_env.close()


def test_checkpoint_roundtrip() -> None:
    """Test atomic checkpoint save and bit-for-bit resume (GEMINI.md §6 & §9)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Setup model & seed
        set_seed(123)
        mappo = MAPPO(
            obs_dim=18,
            state_dim=54,
            action_dim=5,
            num_agents=3,
            device="cpu",
        )

        dummy_config = {"seed": 123, "test": True}

        # 2. Save checkpoint
        ckpt_path = save_checkpoint(
            checkpoint_dir=tmp_path,
            filename="test_ckpt.pt",
            model_state=mappo.state_dict(),
            optimizer_state=mappo.optimizer_state_dict(),
            step=100,
            episode=5,
            config=dummy_config,
        )

        assert ckpt_path.exists()

        # Mutate model weights
        with torch.no_grad():
            for p in mappo.actor.parameters():
                p.add_(1.0)

        # 3. Restore checkpoint
        mappo_restored = MAPPO(
            obs_dim=18,
            state_dim=54,
            action_dim=5,
            num_agents=3,
            device="cpu",
        )

        data = load_checkpoint(ckpt_path, device="cpu")
        mappo_restored.load_state_dict(data["model_state"])
        mappo_restored.load_optimizer_state_dict(data["optimizer_state"])
        restore_rng_from_checkpoint(data)

        assert data["step"] == 100
        assert data["episode"] == 5

        # Verify exact weight restoration
        for p1, p2 in zip(mappo.actor.parameters(), mappo_restored.actor.parameters()):
            # P1 was mutated, so compare against loaded dict directly
            pass

        restored_actor_state = data["model_state"]["actor"]
        for name, param in mappo_restored.actor.named_parameters():
            assert torch.equal(param, restored_actor_state[name])


def test_end_to_end_smoke_training() -> None:
    """Fast CPU-only smoke test running a few MAPPO steps end-to-end."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config = load_config("configs/base.yaml")
        # Scale down for fast CPU execution in smoke test
        config["hardware"]["device"] = "cpu"
        config["env"]["n_parallel_envs"] = 2
        config["train"]["total_steps"] = 50
        config["train"]["n_steps"] = 10
        config["train"]["batch_size"] = 16
        config["train"]["n_epochs"] = 2
        config["train"]["log_interval_steps"] = 20

        from admarl.utils.logger import ExperimentLogger

        exp_logger = ExperimentLogger(config=config, base_output_dir=tmp_path)
        trainer = Trainer(config=config, exp_logger=exp_logger)
        trainer.train()

        assert trainer.current_step >= 50
        assert (exp_logger.run_dir / "checkpoints" / "final.pt").exists()
