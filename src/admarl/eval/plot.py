"""Plotting entry point for generating robustness curves and paper figures (GEMINI.md §8 & §11).

Usage:
    python -m admarl.eval.plot --sweep-dir runs/sweep_phase6 --output-dir figures/
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from admarl.eval.aggregate import aggregate_sweep_results

logger = logging.getLogger(__name__)

# Distinct color palette per arm
ARM_COLORS = {
    "none": "#d95f02",       # Orange/Red for baseline
    "lipschitz": "#1b9e77",  # Green/Teal for our contribution
    "sa_ppo": "#7570b3",     # Purple for SA-PPO baseline
}

ARM_LABELS = {
    "none": "Undefended MAPPO (Baseline)",
    "lipschitz": "Lipschitz Centralized Critic (Ours)",
    "sa_ppo": "SA-PPO Adversarial Training (Reproduced Baseline)",
}


def generate_robustness_plots(sweep_dir: Path | str, output_dir: Path | str) -> list[Path]:
    """Generate robustness curves and paper figures from aggregated metrics.csv data.

    Args:
        sweep_dir: Directory containing sweep run subdirectories
        output_dir: Directory to save generated plot images

    Returns:
        List of generated figure file paths
    """
    sweep_path = Path(sweep_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = aggregate_sweep_results(sweep_path)
    summary = data.get("summary", {})

    if not summary:
        logger.warning("No summary data found in %s to plot.", sweep_dir)
        return []

    generated_files: list[Path] = []

    # Set matplotlib style
    plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14})

    # Group data by arm, epsilon, budget
    arms = sorted({v["arm"] for v in summary.values()})
    all_epsilons = sorted({v["epsilon"] for v in summary.values()})
    all_budgets = sorted({v["budget_k"] for v in summary.values()})

    # Target fixed values for 2D slices
    target_eps = 0.05 if 0.05 in all_epsilons else (all_epsilons[0] if all_epsilons else 0.0)
    target_k = 5 if 5 in all_budgets else (all_budgets[-1] if all_budgets else 0)

    # 1. Figure 1: Return vs. Attack Budget (k) at fixed epsilon
    _fig, ax = plt.subplots(figsize=(7, 5))
    for arm in arms:
        k_vals = []
        means = []
        cis = []

        for k in all_budgets:
            key = f"{arm}_k{k}_eps{target_eps}"
            if key in summary:
                k_vals.append(k)
                means.append(summary[key]["mean"])
                cis.append(summary[key]["ci95"])

        if k_vals:
            color = ARM_COLORS.get(arm, "#333333")
            label = ARM_LABELS.get(arm, arm)
            ax.plot(k_vals, means, marker="o", linewidth=2.0, color=color, label=label)
            ax.fill_between(
                k_vals,
                [m - c for m, c in zip(means, cis)],
                [m + c for m, c in zip(means, cis)],
                color=color,
                alpha=0.2,
            )

    ax.set_xlabel("Attack Budget $k$ (Perturbations per Episode)")
    ax.set_ylabel("Post-Attack Mean Return")
    ax.set_title(rf"Robustness vs. Attack Budget ($\epsilon = {target_eps}$)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()

    fig_path1_png = out_path / "return_vs_budget.png"
    fig_path1_pdf = out_path / "return_vs_budget.pdf"
    plt.savefig(fig_path1_png, dpi=300)
    plt.savefig(fig_path1_pdf)
    plt.close()
    generated_files.extend([fig_path1_png, fig_path1_pdf])

    # 2. Figure 2: Return vs. Perturbation Radius (epsilon) at fixed budget
    _fig, ax = plt.subplots(figsize=(7, 5))
    for arm in arms:
        eps_vals = []
        means = []
        cis = []

        for eps in all_epsilons:
            key = f"{arm}_k{target_k}_eps{eps}"
            if key in summary:
                eps_vals.append(eps)
                means.append(summary[key]["mean"])
                cis.append(summary[key]["ci95"])

        if eps_vals:
            color = ARM_COLORS.get(arm, "#333333")
            label = ARM_LABELS.get(arm, arm)
            ax.plot(eps_vals, means, marker="s", linewidth=2.0, color=color, label=label)
            ax.fill_between(
                eps_vals,
                [m - c for m, c in zip(means, cis)],
                [m + c for m, c in zip(means, cis)],
                color=color,
                alpha=0.2,
            )

    ax.set_xlabel(r"Perturbation Radius $\epsilon$ ($L_\infty$)")
    ax.set_ylabel("Post-Attack Mean Return")
    ax.set_title(f"Robustness vs. Perturbation Radius ($k = {target_k}$)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()

    fig_path2_png = out_path / "return_vs_epsilon.png"
    fig_path2_pdf = out_path / "return_vs_epsilon.pdf"
    plt.savefig(fig_path2_png, dpi=300)
    plt.savefig(fig_path2_pdf)
    plt.close()
    generated_files.extend([fig_path2_png, fig_path2_pdf])

    # 3. Figure 3: Clean Anchor (k=0) Performance Comparison
    _fig, ax = plt.subplots(figsize=(6, 4))
    clean_arms = []
    clean_means = []
    clean_stds = []

    for arm in arms:
        # Find k=0 entry for this arm
        clean_keys = [k for k in summary if summary[k]["arm"] == arm and summary[k]["budget_k"] == 0]
        if clean_keys:
            key = clean_keys[0]
            clean_arms.append(ARM_LABELS.get(arm, arm))
            clean_means.append(summary[key]["mean"])
            clean_stds.append(summary[key]["std"])

    if clean_arms:
        bars = ax.bar(clean_arms, clean_means, yerr=clean_stds, capsize=5, alpha=0.85)
        for bar, arm_key in zip(bars, arms):
            bar.set_color(ARM_COLORS.get(arm_key, "#333333"))

        ax.set_ylabel("Clean Episode Return ($k = 0$)")
        ax.set_title("Nominal Performance Anchor ($k=0$)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        fig_path3_png = out_path / "clean_anchor_comparison.png"
        fig_path3_pdf = out_path / "clean_anchor_comparison.pdf"
        plt.savefig(fig_path3_png, dpi=300)
        plt.savefig(fig_path3_pdf)
        plt.close()
        generated_files.extend([fig_path3_png, fig_path3_pdf])

    logger.info("Generated %d figure artifacts in %s", len(generated_files), out_path)
    return generated_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate robustness figures from aggregated experiment metrics.")
    parser.add_argument("--sweep-dir", type=str, default="runs/sweep_phase6", help="Path to sweep results directory")
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory to save generated figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_robustness_plots(sweep_dir=args.sweep_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
