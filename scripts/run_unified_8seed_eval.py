"""Unified 8-Seed Evaluation & Significance Harness (GEMINI.md §8 & §11).

Single-source evaluation pass producing ALL spread paper results from ONE execution (n=8 seeds).

Covers:
1. Table 1 (Accessibility gap: PGD vs Oracle k-sweep for none arm)
2. Table 4A / 4B (Logit margins + per-state distributional analysis)
3. Table 5 (Three-arm defense comparison: none vs lipschitz vs sa_ppo at eps=0.15)
4. VERIFY 1 (PGD step budget convergence plateau)
5. Statistical Significance Tests (n=8): Welch's t-test and Bootstrap p-values for none-vs-lipschitz and none-vs-sa_ppo
6. Export for robustness curves plotting
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats
import torch

from admarl.algos.mappo import MAPPO
from admarl.attacks.critic_sensitivity import CriticSensitivityAttack
from admarl.envs.mpe import MPEEnv

logger = logging.getLogger(__name__)

SEEDS = list(range(8))  # 8 seeds: 0..7
NUM_EPISODES = 30
EVAL_SEEDS = [3000 + i for i in range(NUM_EPISODES)]
MODELS_DIR = Path("runs/sweep_phase6/models")


def evaluate_spread_cell(
    mappo: MAPPO,
    budget_k: int = 5,
    epsilon: float = 0.05,
    pgd_steps: int = 5,
    attack_mode: str = "pgd",  # "pgd" or "oracle"
    num_episodes: int = NUM_EPISODES,
    eval_seeds: list[int] | None = None,
) -> dict[str, float]:
    if eval_seeds is None:
        eval_seeds = EVAL_SEEDS

    env = MPEEnv(env_name="simple_spread", max_cycles=25)
    ep_returns: list[float] = []

    attack = (
        CriticSensitivityAttack(budget_k=budget_k, epsilon=epsilon, pgd_steps=pgd_steps)
        if (budget_k > 0 and attack_mode == "pgd")
        else None
    )

    mappo.actor.eval()
    if mappo.critic is not None:
        mappo.critic.eval()

    device = mappo.device

    for ep in range(min(num_episodes, len(eval_seeds))):
        seed = eval_seeds[ep]
        obs_np, state_np = env.reset(seed=seed)
        if attack is not None:
            attack.reset_episode()
        attack_budget_remaining = budget_k
        ep_return = 0.0

        for step in range(25):
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                clean_dist = mappo.actor(obs_tensor)
                clean_actions = torch.argmax(clean_dist.logits, dim=-1)
            clean_actions_np = clean_actions.cpu().numpy()

            executed_actions_np = clean_actions_np.copy()
            should_attack = (budget_k >= 25) or (
                attack_budget_remaining > 0 and (step % (25 // max(1, budget_k)) == 0)
            )

            if should_attack and budget_k > 0:
                attack_budget_remaining -= 1

                if attack_mode == "pgd" and attack is not None:
                    perturbed_obs, _ = attack.perturb(
                        obs=obs_tensor, state=state_tensor, critic=mappo.critic, actor=mappo.actor, step=step
                    )
                    with torch.no_grad():
                        adv_actions, _ = mappo.actor.get_action(perturbed_obs, deterministic=True)
                    executed_actions_np = adv_actions.cpu().numpy()

                elif attack_mode == "oracle":
                    with torch.no_grad():
                        worst_actions = torch.argmin(clean_dist.logits, dim=-1)
                    executed_actions_np = worst_actions.cpu().numpy()

            next_obs_np, next_state_np, rewards_np, term_np, trunc_np, _info = env.step(executed_actions_np)

            ep_return += float(rewards_np.mean())
            obs_np = next_obs_np
            state_np = next_state_np

            if term_np.all() or trunc_np.all():
                break

        ep_returns.append(ep_return)

    env.close()

    act_frac = 0.0
    worst_hit_frac = 0.0
    if attack is not None and attack.total_attacked_count > 0:
        act_frac = float(attack.action_changed_count / attack.total_attacked_count)
        worst_hit_frac = float(attack.worst_case_hit_count / attack.total_attacked_count)

    return {
        "return_mean": float(np.mean(ep_returns)),
        "action_changed_frac": act_frac,
        "worst_hit_frac": worst_hit_frac,
    }


def measure_logit_margins(
    mappo: MAPPO, num_episodes: int = NUM_EPISODES, eval_seeds: list[int] | None = None
) -> dict[str, Any]:
    if eval_seeds is None:
        eval_seeds = EVAL_SEEDS

    env = MPEEnv(env_name="simple_spread", max_cycles=25)
    mappo.actor.eval()
    device = mappo.device

    delta_z_adj_list = []
    delta_z_worst_list = []
    delta_z_achieved_list = []

    attack = CriticSensitivityAttack(budget_k=25, epsilon=0.15, pgd_steps=5)

    for ep in range(min(num_episodes, len(eval_seeds))):
        seed = eval_seeds[ep]
        obs_np, state_np = env.reset(seed=seed)
        attack.reset_episode()

        for step in range(25):
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
            state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                dist = mappo.actor(obs_tensor)
                logits = dist.logits

            clean_actions = torch.argmax(logits, dim=-1)
            worst_actions = torch.argmin(logits, dim=-1)

            for i in range(len(clean_actions)):
                c_act = clean_actions[i].item()
                w_act = worst_actions[i].item()
                z_clean = logits[i, c_act].item()
                z_worst = logits[i, w_act].item()

                other_logits = [logits[i, a].item() for a in range(5) if a != c_act]
                z_adj = max(other_logits)

                delta_z_adj_list.append(z_clean - z_adj)
                delta_z_worst_list.append(z_clean - z_worst)

            perturbed_obs, _ = attack.perturb(
                obs=obs_tensor, state=state_tensor, critic=mappo.critic, actor=mappo.actor, step=step
            )
            with torch.no_grad():
                pert_dist = mappo.actor(perturbed_obs)
                pert_logits = pert_dist.logits

            for i in range(len(clean_actions)):
                c_act = clean_actions[i].item()
                shift = abs(pert_logits[i, c_act].item() - logits[i, c_act].item())
                delta_z_achieved_list.append(shift)

            next_obs_np, next_state_np, _rewards_np, term_np, trunc_np, _info = env.step(
                clean_actions.cpu().numpy()
            )
            obs_np = next_obs_np
            state_np = next_state_np

            if term_np.all() or trunc_np.all():
                break

    env.close()

    shifts_np = np.array(delta_z_achieved_list)
    adj_np = np.array(delta_z_adj_list)
    worst_np = np.array(delta_z_worst_list)

    return {
        "margin_adjacent_mean": float(np.mean(adj_np)),
        "margin_worst_mean": float(np.mean(worst_np)),
        "shift_achieved_mean": float(np.mean(shifts_np)),
        "pct_exceed_adj_margin": float(np.mean(shifts_np >= adj_np)),
        "pct_exceed_worst_margin": float(np.mean(shifts_np >= worst_np)),
    }


def run_welch_and_bootstrap_tests(a: list[float], b: list[float]) -> dict[str, float]:
    """Perform Welch's t-test and 10,000-sample bootstrap test between two seed-level metric distributions."""
    arr_a = np.array(a)
    arr_b = np.array(b)

    welch_res = stats.ttest_ind(arr_a, arr_b, equal_var=False)

    diff_mean = float(np.mean(arr_b) - np.mean(arr_a))

    # 10,000 bootstrap resamples
    rng = np.random.default_rng(42)
    n_boot = 10000
    boot_diffs = []
    for _ in range(n_boot):
        sample_a = rng.choice(arr_a, size=len(arr_a), replace=True)
        sample_b = rng.choice(arr_b, size=len(arr_b), replace=True)
        boot_diffs.append(np.mean(sample_b) - np.mean(sample_a))

    boot_diffs_np = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs_np, 2.5))
    ci_upper = float(np.percentile(boot_diffs_np, 97.5))

    # Two-sided bootstrap p-value under null hypothesis (mean diff = 0)
    null_diffs = boot_diffs_np - np.mean(boot_diffs_np)
    p_boot = float(np.mean(np.abs(null_diffs) >= np.abs(diff_mean)))

    return {
        "mean_a": float(np.mean(arr_a)),
        "sem_a": float(np.std(arr_a, ddof=1) / np.sqrt(len(arr_a))),
        "mean_b": float(np.mean(arr_b)),
        "sem_b": float(np.std(arr_b, ddof=1) / np.sqrt(len(arr_b))),
        "mean_diff": diff_mean,
        "welch_t_stat": float(welch_res.statistic),
        "welch_p_value": float(welch_res.pvalue),
        "bootstrap_p_value": p_boot,
        "bootstrap_ci_95_lower": ci_lower,
        "bootstrap_ci_95_upper": ci_upper,
    }


def load_mappo_model(arm: str, seed: int) -> MAPPO:
    ckpt_path = MODELS_DIR / f"{arm}_seed{seed}" / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint for {arm}_seed{seed} not found at {ckpt_path}")
    ckpt_data = torch.load(ckpt_path, weights_only=False)
    mappo = MAPPO(obs_dim=18, state_dim=54, action_dim=5, num_agents=3)
    mappo.load_state_dict(ckpt_data["model_state"])
    return mappo


def run_full_8seed_eval() -> dict[str, Any]:
    logger.info("Starting Full 8-Seed Evaluation Pass (seeds 0..7)")
    results: dict[str, Any] = {"seeds": SEEDS, "num_episodes": NUM_EPISODES}

    # =========================================================================
    # 1. TABLE 1: Accessibility Gap (PGD vs Oracle k-sweep for none arm)
    # =========================================================================
    logger.info("=== 1. Table 1: Accessibility Gap (PGD vs Oracle) ===")
    budgets = [0, 5, 10, 15, 20, 25]
    epsilons = [0.01, 0.05, 0.10, 0.15]
    table1_res: dict[str, Any] = {}

    for eps in epsilons:
        eps_grid: dict[str, Any] = {}
        for k in budgets:
            rets, flips, worsts = [], [], []
            for seed in SEEDS:
                mappo = load_mappo_model("none", seed)
                res = evaluate_spread_cell(
                    mappo, budget_k=k, epsilon=eps, pgd_steps=5, attack_mode="pgd"
                )
                rets.append(res["return_mean"])
                flips.append(res["action_changed_frac"])
                worsts.append(res["worst_hit_frac"])

            eps_grid[str(k)] = {
                "return_mean": float(np.mean(rets)),
                "return_sem": float(np.std(rets, ddof=1) / np.sqrt(len(SEEDS))),
                "return_seed_means": rets,
                "flip_frac": float(np.mean(flips)),
                "worst_hit_frac": float(np.mean(worsts)),
            }
        table1_res[str(eps)] = eps_grid

    # Oracle k-sweep
    oracle_grid: dict[str, Any] = {}
    for k in budgets:
        rets = []
        for seed in SEEDS:
            mappo = load_mappo_model("none", seed)
            res = evaluate_spread_cell(
                mappo, budget_k=k, epsilon=0.0, pgd_steps=0, attack_mode="oracle"
            )
            rets.append(res["return_mean"])

        oracle_grid[str(k)] = {
            "return_mean": float(np.mean(rets)),
            "return_sem": float(np.std(rets, ddof=1) / np.sqrt(len(SEEDS))),
            "return_seed_means": rets,
        }
    table1_res["oracle"] = oracle_grid
    results["table1_accessibility"] = table1_res

    # =========================================================================
    # 2. TABLE 4A / 4B: Logit Margins & Distributional Analysis
    # =========================================================================
    logger.info("=== 2. Table 4A/4B: Logit Margins & Distributional Analysis ===")
    m_adj_means, m_worst_means, shift_means = [], [], []
    pct_exceed_adj_list, pct_exceed_worst_list = [], []

    for seed in SEEDS:
        mappo = load_mappo_model("none", seed)
        m_res = measure_logit_margins(mappo)
        m_adj_means.append(m_res["margin_adjacent_mean"])
        m_worst_means.append(m_res["margin_worst_mean"])
        shift_means.append(m_res["shift_achieved_mean"])
        pct_exceed_adj_list.append(m_res["pct_exceed_adj_margin"])
        pct_exceed_worst_list.append(m_res["pct_exceed_worst_margin"])

    results["table4_logit_margins"] = {
        "margin_adjacent_mean": float(np.mean(m_adj_means)),
        "margin_adjacent_sem": float(np.std(m_adj_means, ddof=1) / np.sqrt(len(SEEDS))),
        "margin_adjacent_seed_means": m_adj_means,
        "margin_worst_mean": float(np.mean(m_worst_means)),
        "margin_worst_sem": float(np.std(m_worst_means, ddof=1) / np.sqrt(len(SEEDS))),
        "margin_worst_seed_means": m_worst_means,
        "shift_achieved_mean": float(np.mean(shift_means)),
        "shift_achieved_sem": float(np.std(shift_means, ddof=1) / np.sqrt(len(SEEDS))),
        "shift_achieved_seed_means": shift_means,
        "pct_exceed_adj_margin": float(np.mean(pct_exceed_adj_list)),
        "pct_exceed_adj_margin_sem": float(np.std(pct_exceed_adj_list, ddof=1) / np.sqrt(len(SEEDS))),
        "pct_exceed_worst_margin": float(np.mean(pct_exceed_worst_list)),
        "pct_exceed_worst_margin_sem": float(np.std(pct_exceed_worst_list, ddof=1) / np.sqrt(len(SEEDS))),
    }

    # =========================================================================
    # 3. TABLE 5: Three-Arm Defense Comparison (eps=0.15 across budget k)
    # =========================================================================
    logger.info("=== 3. Table 5: Three-Arm Defense Comparison (eps=0.15) ===")
    arms = ["none", "lipschitz", "sa_ppo"]
    eps_defense = 0.15
    table5_res: dict[str, Any] = {}

    for arm in arms:
        arm_k_grid: dict[str, Any] = {}
        for k in budgets:
            rets, flips, worsts = [], [], []
            for seed in SEEDS:
                mappo = load_mappo_model(arm, seed)
                res = evaluate_spread_cell(
                    mappo, budget_k=k, epsilon=eps_defense, pgd_steps=5, attack_mode="pgd"
                )
                rets.append(res["return_mean"])
                flips.append(res["action_changed_frac"])
                worsts.append(res["worst_hit_frac"])

            arm_k_grid[str(k)] = {
                "return_mean": float(np.mean(rets)),
                "return_sem": float(np.std(rets, ddof=1) / np.sqrt(len(SEEDS))),
                "return_seed_means": rets,
                "flip_frac": float(np.mean(flips)),
                "worst_hit_frac": float(np.mean(worsts)),
            }
        table5_res[arm] = arm_k_grid
    results["table5_three_arm"] = table5_res

    # =========================================================================
    # 4. VERIFY 1: PGD Step Budget Sweep (eps=0.15, k=25)
    # =========================================================================
    logger.info("=== 4. VERIFY 1: PGD Step Budget Plateau ===")
    step_counts = [1, 5, 10, 20, 40, 80]
    verify1_res: dict[str, Any] = {}

    for steps in step_counts:
        rets, flips, worsts = [], [], []
        for seed in SEEDS:
            mappo = load_mappo_model("none", seed)
            res = evaluate_spread_cell(
                mappo, budget_k=25, epsilon=0.15, pgd_steps=steps, attack_mode="pgd"
            )
            rets.append(res["return_mean"])
            flips.append(res["action_changed_frac"])
            worsts.append(res["worst_hit_frac"])

        verify1_res[str(steps)] = {
            "return_mean": float(np.mean(rets)),
            "return_sem": float(np.std(rets, ddof=1) / np.sqrt(len(SEEDS))),
            "return_seed_means": rets,
            "flip_frac": float(np.mean(flips)),
            "worst_hit_frac": float(np.mean(worsts)),
        }
    results["verify1_pgd_steps"] = verify1_res

    # =========================================================================
    # 5. STATISTICAL SIGNIFICANCE TESTS (n=8 seeds)
    # =========================================================================
    logger.info("=== 5. Statistical Significance Tests (n=8) ===")
    none_clean_seeds = table5_res["none"]["0"]["return_seed_means"]
    none_k25_seeds = table5_res["none"]["25"]["return_seed_means"]

    lipschitz_clean_seeds = table5_res["lipschitz"]["0"]["return_seed_means"]
    lipschitz_k25_seeds = table5_res["lipschitz"]["25"]["return_seed_means"]

    sa_ppo_clean_seeds = table5_res["sa_ppo"]["0"]["return_seed_means"]
    sa_ppo_k25_seeds = table5_res["sa_ppo"]["25"]["return_seed_means"]

    sig_tests = {
        "none_vs_lipschitz_clean": run_welch_and_bootstrap_tests(none_clean_seeds, lipschitz_clean_seeds),
        "none_vs_lipschitz_k25": run_welch_and_bootstrap_tests(none_k25_seeds, lipschitz_k25_seeds),
        "none_vs_sa_ppo_clean": run_welch_and_bootstrap_tests(none_clean_seeds, sa_ppo_clean_seeds),
        "none_vs_sa_ppo_k25": run_welch_and_bootstrap_tests(none_k25_seeds, sa_ppo_k25_seeds),
    }
    results["significance_tests"] = sig_tests

    # Save to JSON file
    out_json = Path("eval_results_8seed_unified.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved 8-seed unified evaluation results to %s", out_json)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_8seed_eval()
