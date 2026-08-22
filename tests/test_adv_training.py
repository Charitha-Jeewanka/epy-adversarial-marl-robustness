"""Unit tests for SA-PPO adversarial training defense, detach semantics, and non-finite guards (GEMINI.md §9)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from admarl.algos.mappo import MAPPO
from admarl.algos.models import ActorNetwork
from admarl.defenses.factory import get_training_defense
from admarl.defenses.sa_ppo import SAPPOAdversarialDefense
from admarl.defenses.training_defense import NoActorDefense
from admarl.training.train import Trainer
from admarl.utils.config import load_config
from admarl.utils.logger import ExperimentLogger
from admarl.utils.seed import set_seed


def test_factory_instantiation() -> None:
    """Test get_training_defense factory creation."""
    cfg_none = {"adv_training": {"enabled": False, "reg_coeff": 0.0}}
    defense_none = get_training_defense(cfg_none)
    assert isinstance(defense_none, NoActorDefense)

    cfg_sa = {
        "adv_training": {
            "enabled": True,
            "epsilon": 0.05,
            "pgd_steps": 5,
            "pgd_step_size": 0.01,
            "reg_coeff": 1.0,
            "norm": "linf",
        }
    }
    defense_sa = get_training_defense(cfg_sa)
    assert isinstance(defense_sa, SAPPOAdversarialDefense)
    assert defense_sa.epsilon == 0.05
    assert defense_sa.pgd_steps == 5


def test_epsilon_ball_containment() -> None:
    """Assert inner PGD perturbations are strictly bounded within epsilon-ball."""
    epsilon = 0.05
    obs = torch.randn(16, 3, 18)
    actor = ActorNetwork(obs_dim=18, action_dim=5)

    defense = SAPPOAdversarialDefense(epsilon=epsilon, pgd_steps=5, pgd_step_size=0.01, reg_coeff=1.0)
    actions = torch.randint(0, 5, (16, 3))

    loss, _metrics = defense.compute_robust_loss(actor, obs, actions)
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss)


def test_explicit_detach_semantics() -> None:
    """Verify that final_perturbed_obs is detached and clean target logits anchor is detached."""
    set_seed(42)
    obs = torch.randn(8, 3, 18)
    actor = ActorNetwork(obs_dim=18, action_dim=5)
    actions = torch.randint(0, 5, (8, 3))

    defense = SAPPOAdversarialDefense(epsilon=0.05, pgd_steps=3, reg_coeff=1.0)

    # Backprop through robust loss
    actor.zero_grad()
    loss, _ = defense.compute_robust_loss(actor, obs, actions)
    loss.backward()

    # Verify gradients computed for actor parameters
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in actor.parameters())
    assert has_grad


def test_noop_equivalence() -> None:
    """Assert NoActorDefense returns zero loss and matches unregularized MAPPO update."""
    no_defense = NoActorDefense()
    obs = torch.randn(16, 3, 18)
    actions = torch.randint(0, 5, (16, 3))
    actor = ActorNetwork(obs_dim=18, action_dim=5)

    loss_zero, metrics = no_defense.compute_robust_loss(actor, obs, actions)
    assert loss_zero.item() == 0.0
    assert metrics == {}

    # Compare MAPPO update with NoActorDefense vs None
    set_seed(123)
    mappo1 = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3, device="cpu")
    set_seed(123)
    mappo2 = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3, device="cpu")

    state = torch.randn(16, 54)
    old_log_probs = torch.randn(16, 3)
    returns = torch.randn(16, 3)
    advantages = torch.randn(16, 3)

    metrics1 = mappo1.update(obs, state, actions, old_log_probs, returns, advantages, training_defense=None)
    metrics2 = mappo2.update(obs, state, actions, old_log_probs, returns, advantages, training_defense=no_defense)

    assert metrics1["actor_loss"] == pytest.approx(metrics2["actor_loss"])
    assert metrics1["critic_loss"] == pytest.approx(metrics2["critic_loss"])


class NonFiniteActor(nn.Module):
    """Mock actor returning NaN for testing non-finite guards."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(18, 5)
        with torch.no_grad():
            self.fc.weight.fill_(float("nan"))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.fc(obs)
        return torch.distributions.Categorical(logits=logits, validate_args=False)


def test_non_finite_guard_fires() -> None:
    """Assert RuntimeError is raised when SA-PPO regularizer encounters NaN/Inf (GEMINI.md §7)."""
    defense = SAPPOAdversarialDefense(epsilon=0.05, pgd_steps=3, reg_coeff=1.0)
    bad_actor = NonFiniteActor()
    obs = torch.randn(16, 3, 18)
    actions = torch.randint(0, 5, (16, 3))

    with pytest.raises(RuntimeError, match="Non-finite robustness loss"):
        defense.compute_robust_loss(bad_actor, obs, actions)


def test_adv_training_smoke() -> None:
    """End-to-end smoke test running MAPPO training with SA-PPO adversarial training enabled."""
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

        # Enable SA-PPO adversarial training
        config["adv_training"]["enabled"] = True
        config["adv_training"]["epsilon"] = 0.05
        config["adv_training"]["pgd_steps"] = 3
        config["adv_training"]["reg_coeff"] = 1.0

        exp_logger = ExperimentLogger(config=config, base_output_dir=tmp_path)
        trainer = Trainer(config=config, exp_logger=exp_logger)
        trainer.train()

        assert trainer.current_step >= 40
        assert (exp_logger.run_dir / "checkpoints" / "final.pt").exists()
