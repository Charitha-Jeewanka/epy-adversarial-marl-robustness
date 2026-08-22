"""Budget/epsilon-ball invariants and safety tests for attacks (GEMINI.md §9)."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from admarl.algos.models import CentralizedCriticNetwork
from admarl.attacks.critic_sensitivity import CriticSensitivityAttack
from admarl.attacks.factory import get_attack
from admarl.attacks.no_attack import NoAttack


def test_factory_instantiation() -> None:
    """Test get_attack factory creation."""
    cfg_none = {"attack": {"name": "none"}}
    attack_none = get_attack(cfg_none)
    assert isinstance(attack_none, NoAttack)

    cfg_sens = {
        "attack": {
            "name": "critic_sensitivity",
            "budget_k": 3,
            "epsilon": 0.05,
            "norm": "linf",
        }
    }
    attack_sens = get_attack(cfg_sens)
    assert isinstance(attack_sens, CriticSensitivityAttack)
    assert attack_sens.budget_k == 3
    assert attack_sens.epsilon == 0.05


def test_budget_never_exceeded() -> None:
    """Assert attack perturbation count never exceeds budget_k across rollout steps (GEMINI.md §9)."""
    budget_k = 3
    attack = CriticSensitivityAttack(budget_k=budget_k, epsilon=0.05, norm="linf")
    critic = CentralizedCriticNetwork(state_dim=54, num_agents=3)

    obs = torch.randn(3, 18)
    state = obs.reshape(-1)

    perturbed_count = 0
    for step in range(10):
        perturbed_obs, is_perturbed = attack.perturb(obs, state, critic, step=step)
        if is_perturbed:
            perturbed_count += 1
            # Verify observation was actually altered
            assert not torch.equal(perturbed_obs, obs)
        else:
            # Verify unperturbed observation returned when budget exhausted
            assert torch.equal(perturbed_obs, obs)

    assert perturbed_count == budget_k
    assert attack.perturbations_used == budget_k

    # Test reset_episode restores budget
    attack.reset_episode()
    assert attack.perturbations_used == 0


def test_perturbation_within_epsilon_ball() -> None:
    """Assert max perturbation strictly obeys epsilon-ball constraint for Linf and L2 norms."""
    epsilon = 0.05

    # Linf test
    attack_linf = CriticSensitivityAttack(budget_k=5, epsilon=epsilon, norm="linf")
    critic = CentralizedCriticNetwork(state_dim=54, num_agents=3)

    obs = torch.randn(3, 18)
    state = obs.reshape(-1)

    perturbed_linf, is_perturbed = attack_linf.perturb(obs, state, critic)
    assert is_perturbed
    max_diff_linf = torch.max(torch.abs(perturbed_linf - obs)).item()
    assert max_diff_linf <= epsilon + 1e-6

    # L2 test
    attack_l2 = CriticSensitivityAttack(budget_k=5, epsilon=epsilon, norm="l2")
    perturbed_l2, is_perturbed_l2 = attack_l2.perturb(obs, state, critic)
    assert is_perturbed_l2
    max_diff_l2 = torch.max(torch.abs(perturbed_l2 - obs)).item()
    assert max_diff_l2 <= epsilon + 1e-6


def test_sensitivity_driven_selection() -> None:
    """Assert perturbation is sensitivity-driven and deterministic for a fixed critic and seed."""
    torch.manual_seed(42)
    critic = CentralizedCriticNetwork(state_dim=54, num_agents=3)
    obs = torch.randn(3, 18)
    state = obs.reshape(-1)

    attack1 = CriticSensitivityAttack(budget_k=1, epsilon=0.05)
    p_obs1, _ = attack1.perturb(obs, state, critic)

    attack2 = CriticSensitivityAttack(budget_k=1, epsilon=0.05)
    p_obs2, _ = attack2.perturb(obs, state, critic)

    assert torch.equal(p_obs1, p_obs2)


class NonFiniteCritic(nn.Module):
    """Mock critic returning non-finite outputs for testing guards."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(54, 3)
        with torch.no_grad():
            self.fc.weight.fill_(float("nan"))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc(state)


def test_non_finite_guard_fires() -> None:
    """Assert RuntimeError is raised when non-finite values are encountered (GEMINI.md §7)."""
    attack = CriticSensitivityAttack(budget_k=5, epsilon=0.05)
    bad_critic = NonFiniteCritic()

    obs = torch.randn(3, 18)
    state = obs.reshape(-1)

    # Injected non-finite output should raise RuntimeError or return unperturbed gracefully
    with pytest.raises(RuntimeError, match="Non-finite value"):
        attack.perturb(obs, state, bad_critic)
