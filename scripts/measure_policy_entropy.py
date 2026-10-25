"""Mean per-step action entropy of trained actors on clean rollouts.

Answers ICAC 2026 Reviewer 3: "State the entropy or temperature of the trained policies."
Fills the two [TBD] values in Section III-A of the camera-ready.

Mirrors the clean (k=0) rollout protocol of run_unified_8seed_eval.py:
same eval seeds (3000..3029), 25 steps, greedy (argmax) execution, no attack.
Entropy is computed on the full softmax distribution at every visited state,
averaged per seed, then reported as mean +/- SEM over seeds. Units: nats
(natural log), so the maximum for 5 actions is ln 5 = 1.609.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from admarl.algos.mappo import MAPPO
from admarl.envs.mpe import MPEEnv
from scripts.run_unified_8seed_eval import EVAL_SEEDS, load_mappo_model

logger = logging.getLogger(__name__)

MAX_STEPS = 25

# simple_tag_v3 eval env kwargs, copied from evaluate_tag_cell() in run_verify2_tag_gap.py
# (the Table III script). Prey speed/accel are left at MPEEnv defaults (1.05/3.1), as in Table III.
TAG_ENV_KWARGS = {"num_obstacles": 3, "shaping_coeff": 0.0}


def load_tag_model(seed: int) -> MAPPO:
    """Load a simple_tag_v3 predator checkpoint; copied from main() in run_verify2_tag_gap.py."""
    ckpt_path = Path("runs/spike_simple_tag/models") / f"none_seed{seed}" / "checkpoints" / "final.pt"
    ckpt_data = torch.load(ckpt_path, weights_only=False)
    # simple_tag_v3 predator dims: obs_dim=18, state_dim=70, action_dim=5, num_agents=3
    mappo = MAPPO(obs_dim=18, state_dim=70, action_dim=5, num_agents=3)
    mappo.load_state_dict(ckpt_data["model_state"])
    return mappo


def mean_clean_entropy(
    mappo, env_name: str, eval_seeds: list[int] = EVAL_SEEDS, env_kwargs: dict | None = None
) -> float:
    """Mean action entropy over all agents and steps of clean greedy rollouts.

    For simple_tag the actor only covers the 3 predators; the prey is driven by
    MPEEnv's flee heuristic, so it never enters the average.
    """
    env = MPEEnv(env_name=env_name, max_cycles=MAX_STEPS, **(env_kwargs or {}))
    mappo.actor.eval()
    entropies: list[float] = []

    for seed in eval_seeds:
        obs_np, _state_np = env.reset(seed=seed)
        for _ in range(MAX_STEPS):
            obs = torch.tensor(obs_np, dtype=torch.float32, device=mappo.device)
            with torch.no_grad():
                logits = mappo.actor(obs).logits
            entropies.extend(Categorical(logits=logits).entropy().cpu().tolist())
            actions = torch.argmax(logits, dim=-1).cpu().numpy()
            obs_np, _state_np, _r, term, trunc, _info = env.step(actions)
            if term.all() or trunc.all():
                break

    env.close()
    return float(np.mean(entropies))


def summarize(seed_values: list[float]) -> dict[str, float | list[float]]:
    arr = np.array(seed_values)
    return {
        "mean": float(arr.mean()),
        "sem": float(arr.std(ddof=1) / np.sqrt(len(arr))),
        "seed_values": seed_values,
    }


def main() -> None:
    results: dict[str, dict] = {"units": "nats", "max_entropy_ln5": float(np.log(5))}

    # simple_spread_v3: undefended arm, seeds 0..7 (same checkpoints as Tables I and IV)
    spread = [mean_clean_entropy(load_mappo_model("none", s), "simple_spread") for s in range(8)]
    results["simple_spread_v3"] = summarize(spread)

    # simple_tag_v3: predator policies, seeds 0..2 (same checkpoints and eval env as Table III)
    tag = [mean_clean_entropy(load_tag_model(s), "simple_tag", env_kwargs=TAG_ENV_KWARGS) for s in range(3)]
    results["simple_tag_v3"] = summarize(tag)

    out = Path("policy_entropy.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved %s: %s", out, json.dumps(results, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()