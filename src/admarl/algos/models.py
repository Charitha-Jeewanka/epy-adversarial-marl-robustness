"""Neural network architectures for MAPPO actor and centralized critic (GEMINI.md §2 & §4)."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal parameter initialization standard in PPO/MAPPO."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ActorNetwork(nn.Module):
    """Decentralized actor policy network (parameter-shared across agents)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.body = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        self.apply(lambda m: orthogonal_init(m, gain=np_gain_for(m, self.action_head)))

    def forward(self, obs: torch.Tensor) -> Categorical:
        """Return categorical action distribution over local observation tensor."""
        features = self.body(obs)
        logits = self.action_head(features)
        return Categorical(logits=logits)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        dist = self.forward(obs)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_probs and entropy for given actions."""
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


class CentralizedCriticNetwork(nn.Module):
    """Centralized critic network mapping global state S -> per-agent state-values V_i(S)."""

    def __init__(
        self, state_dim: int, num_agents: int, hidden_dim: int = 64, num_layers: int = 2
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.body = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, num_agents)

        self.apply(lambda m: orthogonal_init(m, gain=1.0))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return state value predictions of shape (..., num_agents)."""
        features = self.body(state)
        values = self.value_head(features)
        return values


def np_gain_for(m: nn.Module, head: nn.Module) -> float:
    if m is head:
        return 0.01
    return 1.414  # np.sqrt(2) for Tanh
