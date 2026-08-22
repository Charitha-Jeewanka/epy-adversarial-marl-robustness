"""Unit tests for GradientPenaltyRegularizer, no-op equivalence, and non-finite guards (GEMINI.md §9)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from admarl.algos.mappo import MAPPO
from admarl.algos.models import CentralizedCriticNetwork
from admarl.defenses.factory import get_regularizer
from admarl.defenses.grad_penalty import GradientPenaltyRegularizer
from admarl.defenses.no_defense import NoRegularizer
from admarl.training.train import Trainer
from admarl.utils.config import load_config
from admarl.utils.logger import ExperimentLogger
from admarl.utils.seed import set_seed


def test_factory_instantiation() -> None:
    """Test get_regularizer factory creation."""
    cfg_none = {"defense": {"name": "none", "penalty_coeff": 0.0}}
    reg_none = get_regularizer(cfg_none)
    assert isinstance(reg_none, NoRegularizer)

    cfg_gp = {"defense": {"name": "grad_penalty", "penalty_coeff": 0.1, "norm": "l2"}}
    reg_gp = get_regularizer(cfg_gp)
    assert isinstance(reg_gp, GradientPenaltyRegularizer)
    assert reg_gp.penalty_coeff == 0.1
    assert reg_gp.norm == "l2"


def test_gradient_penalty_finite_scalar_and_backprop_cpu() -> None:
    """Assert penalty returns a finite scalar tensor and backprops cleanly on CPU (GEMINI.md §9)."""
    set_seed(42)
    critic = CentralizedCriticNetwork(state_dim=54, num_agents=3)
    states = torch.randn(16, 54)

    reg = GradientPenaltyRegularizer(penalty_coeff=0.1, norm="l2")
    penalty = reg.penalty(critic, states)

    assert torch.is_tensor(penalty)
    assert penalty.dim() == 0  # Scalar tensor
    assert torch.isfinite(penalty)
    assert penalty.item() > 0.0

    # Backprop through second-order graph
    critic.zero_grad()
    penalty.backward()

    # Verify gradients computed for critic parameters
    has_grad = False
    for param in critic.parameters():
        if param.grad is not None and param.grad.abs().sum().item() > 0.0:
            has_grad = True
            break
    assert has_grad


def test_noop_equivalence() -> None:
    """Assert NoRegularizer returns zero and produces identical update to unregularized baseline."""
    set_seed(100)
    critic = CentralizedCriticNetwork(state_dim=54, num_agents=3)
    states = torch.randn(16, 54)

    no_reg = NoRegularizer()
    p_zero = no_reg.penalty(critic, states)
    assert p_zero.item() == 0.0

    # Compare MAPPO update with NoRegularizer vs regularizer=None
    set_seed(200)
    mappo1 = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3, device="cpu")
    set_seed(200)
    mappo2 = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3, device="cpu")

    obs = torch.randn(16, 3, 18)
    state = torch.randn(16, 54)
    actions = torch.randint(0, 5, (16, 3))
    old_log_probs = torch.randn(16, 3)
    returns = torch.randn(16, 3)
    advantages = torch.randn(16, 3)

    metrics1 = mappo1.update(
        obs, state, actions, old_log_probs, returns, advantages, regularizer=None
    )
    metrics2 = mappo2.update(
        obs, state, actions, old_log_probs, returns, advantages, regularizer=no_reg
    )

    assert metrics1["critic_loss"] == pytest.approx(metrics2["critic_loss"])
    assert metrics1["actor_loss"] == pytest.approx(metrics2["actor_loss"])


class NonFiniteCritic(nn.Module):
    """Mock critic returning NaN for testing non-finite guards."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(54, 3)
        with torch.no_grad():
            self.fc.weight.fill_(float("nan"))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc(state)


def test_non_finite_guard_fires() -> None:
    """Assert RuntimeError is raised when regularizer encounters NaN/Inf (GEMINI.md §7)."""
    reg = GradientPenaltyRegularizer(penalty_coeff=0.1)
    bad_critic = NonFiniteCritic()
    states = torch.randn(16, 54)

    with pytest.raises(RuntimeError, match="Non-finite regularizer penalty"):
        reg.penalty(bad_critic, states)


def test_defense_enabled_smoke_training() -> None:
    """End-to-end smoke test running MAPPO training with gradient penalty defense enabled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config = load_config("configs/base.yaml")
        config["hardware"]["device"] = "cpu"
        config["env"]["n_parallel_envs"] = 2
        config["train"]["total_steps"] = 40
        config["train"]["n_steps"] = 10
        config["train"]["batch_size"] = 16
        config["train"]["n_epochs"] = 1
        config["train"]["log_interval_steps"] = 20

        # Enable defense
        config["defense"]["name"] = "grad_penalty"
        config["defense"]["penalty_coeff"] = 0.1

        exp_logger = ExperimentLogger(config=config, base_output_dir=tmp_path)
        trainer = Trainer(config=config, exp_logger=exp_logger)
        trainer.train()

        assert trainer.current_step >= 40
        assert (exp_logger.run_dir / "checkpoints" / "final.pt").exists()
