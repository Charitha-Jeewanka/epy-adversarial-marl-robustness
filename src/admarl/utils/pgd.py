"""Shared Projected Gradient Descent (PGD) and epsilon-ball projection primitives (GEMINI.md §4 & DRY principle)."""
from __future__ import annotations

import torch


def project_epsilon_ball(
    orig_obs: torch.Tensor,
    perturbed_obs: torch.Tensor,
    epsilon: float,
    norm: str = "linf",
) -> torch.Tensor:
    """Project perturbed observations back into the epsilon-ball around orig_obs.

    Args:
        orig_obs: Tensor of original clean observations
        perturbed_obs: Tensor of perturbed observations
        epsilon: Epsilon-ball perturbation radius
        norm: Perturbation norm ('linf' or 'l2')

    Returns:
        Tensor of projected observations bounded strictly within epsilon-ball
    """
    if epsilon <= 0.0:
        return orig_obs.clone()

    if norm.lower() == "linf":
        return torch.clamp(perturbed_obs, orig_obs - epsilon, orig_obs + epsilon)
    elif norm.lower() == "l2":
        delta = perturbed_obs - orig_obs
        shape = delta.shape
        flat_delta = delta.reshape(shape[0], -1)
        norms = torch.norm(flat_delta, p=2, dim=-1, keepdim=True)
        scale = torch.clamp(epsilon / (norms + 1e-8), max=1.0)
        scale = scale.reshape([-1] + [1] * (delta.dim() - 1))
        return orig_obs + delta * scale
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Supported norms: 'linf', 'l2'")


def pgd_step(
    obs: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    norm: str = "linf",
    maximize: bool = True,
) -> torch.Tensor:
    """Perform a single gradient-based perturbation step.

    Args:
        obs: Observation tensor to perturb
        grad: Gradient tensor with respect to obs
        step_size: Step size magnitude
        norm: Norm type ('linf' or 'l2')
        maximize: If True, steps in direction of gradient (ascent); otherwise descent.

    Returns:
        Updated observation tensor post PGD step
    """
    direction = 1.0 if maximize else -1.0

    if norm.lower() == "linf":
        delta = direction * step_size * torch.sign(grad)
    elif norm.lower() == "l2":
        flat_grad = grad.reshape(grad.shape[0], -1)
        norms = torch.norm(flat_grad, p=2, dim=-1, keepdim=True)
        scale = norms.reshape([-1] + [1] * (grad.dim() - 1)) + 1e-8
        delta = direction * step_size * (grad / scale)
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Supported norms: 'linf', 'l2'")

    return obs + delta
