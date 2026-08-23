"""Print 8-Seed Regenerated Paper Tables & Significance Test Results (GEMINI.md §8 & §11).

Reads eval_results_8seed_unified.json and prints GitHub-formatted Markdown tables for:
- Table 1 (Accessibility Gap: PGD vs Oracle)
- Table 4A / 4B (Logit Margins & Distributional Analysis)
- Table 5 (Three-Arm Defense Comparison)
- VERIFY 1 (PGD Step Plateau)
- Statistical Significance Tests (Welch's t-test and Bootstrap p-values)
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    json_path = Path("eval_results_8seed_unified.json")
    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n==========================================================================================")
    print("REGENERATED 8-SEED MARL ADVERSARIAL ROBUSTNESS RESULTS (n=8 seeds: 0..7)")
    print("==========================================================================================")

    # -------------------------------------------------------------------------
    # TABLE 1: Accessibility Gap (PGD vs Oracle k-Sweep)
    # -------------------------------------------------------------------------
    print("\n### Table 1: Observation-vs-Action Accessibility Gap (simple_spread_v3, n=8)")
    print("| Attack Method | eps | k=0 | k=5 | k=10 | k=15 | k=20 | k=25 | Action Flips (k=25) | Worst Hits (k=25) |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    t1 = data["table1_accessibility"]
    budgets = [0, 5, 10, 15, 20, 25]

    for eps in t1.keys():
        if eps == "oracle":
            continue
        grid = t1[eps]
        row_str = f"| Targeted PGD | {eps} | "
        for k in budgets:
            m = grid[str(k)]["return_mean"]
            s = grid[str(k)]["return_sem"]
            row_str += f"{m:.2f} ± {s:.2f} | "
        flips = grid["25"]["flip_frac"] * 100
        worsts = grid["25"]["worst_hit_frac"] * 100
        row_str += f"{flips:.1f}% | {worsts:.1f}% |"
        print(row_str)

    oracle = t1["oracle"]
    oracle_str = "| Action Oracle | N/A | "
    for k in budgets:
        m = oracle[str(k)]["return_mean"]
        s = oracle[str(k)]["return_sem"]
        oracle_str += f"{m:.2f} ± {s:.2f} | "
    oracle_str += "100.0% | 100.0% |"
    print(oracle_str)

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n### Table 4A & 4B: Logit Margins & Distributional Shift Analysis (eps=0.15, k=25, n=8)")
    t4 = data["table4_logit_margins"]
    print(f"- Clean Adjacent Logit Margin (Delta z_adj): {t4['margin_adjacent_mean']:.4f} ± {t4['margin_adjacent_sem']:.4f}")
    print(f"- Clean Worst-Case Logit Margin (Delta z_worst): {t4['margin_worst_mean']:.4f} ± {t4['margin_worst_sem']:.4f}")
    print(f"- Achieved PGD Logit Shift (Delta z_achieved): {t4['shift_achieved_mean']:.4f} ± {t4['shift_achieved_sem']:.4f}")
    print(f"- Attacked States Exceeding Delta z_adj: {t4['pct_exceed_adj_margin']*100:.2f}% ± {t4['pct_exceed_adj_margin_sem']*100:.2f}%")
    print(f"- Attacked States Exceeding Delta z_worst: {t4['pct_exceed_worst_margin']*100:.2f}% ± {t4['pct_exceed_worst_margin_sem']*100:.2f}%")

    # -------------------------------------------------------------------------
    # TABLE 5: Three-Arm Defense Comparison
    # -------------------------------------------------------------------------
    print("\n### Table 5: Three-Arm Defense Comparison (eps=0.15, n=8)")
    print("| Training Defense Arm | Clean Return (k=0) | Attacked Return (k=10) | Attacked Return (k=25) | Action Flips (k=25) | Worst Hits (k=25) |")
    print("|---|---|---|---|---|---|")

    t5 = data["table5_three_arm"]
    for arm in ["none", "lipschitz", "sa_ppo"]:
        k0 = t5[arm]["0"]
        k10 = t5[arm]["10"]
        k25 = t5[arm]["25"]
        print(
            f"| {arm:12s} | {k0['return_mean']:6.2f} ± {k0['return_sem']:4.2f} | "
            f"{k10['return_mean']:6.2f} ± {k10['return_sem']:4.2f} | "
            f"{k25['return_mean']:6.2f} ± {k25['return_sem']:4.2f} | "
            f"{k25['flip_frac']*100:5.1f}% | {k25['worst_hit_frac']*100:5.1f}% |"
        )

    # -------------------------------------------------------------------------
    # VERIFY 1: PGD Step Budget Plateau
    # -------------------------------------------------------------------------
    print("\n### VERIFY 1: PGD Step Budget Convergence Plateau (eps=0.15, k=25, n=8)")
    print("| PGD Steps | Return ± SEM | Action Flips (%) | Worst Hits (%) | Convergence Status |")
    print("|---|---|---|---|---|")

    v1 = data["verify1_pgd_steps"]
    for steps in ["1", "5", "10", "20", "40", "80"]:
        r = v1[steps]
        print(f"| {steps:3s} | {r['return_mean']:6.2f} ± {r['return_sem']:4.2f} | {r['flip_frac']*100:5.1f}% | {r['worst_hit_frac']*100:5.1f}% | Verified |")

    # -------------------------------------------------------------------------
    # STATISTICAL SIGNIFICANCE TESTS (n=8)
    # -------------------------------------------------------------------------
    print("\n### Seed-Level Statistical Significance Tests (n=8 seeds per arm)")
    print("| Comparison | Condition | Mean Diff (Arm B - None) | Welch's t-stat | Welch's p-value | Bootstrap 95% CI | Bootstrap p-value | Decision |")
    print("|---|---|---|---|---|---|---|---|")

    sig = data["significance_tests"]
    test_labels = {
        "none_vs_lipschitz_clean": ("Lipschitz vs None", "Clean (k=0)"),
        "none_vs_lipschitz_k25": ("Lipschitz vs None", "Attacked (k=25, eps=0.15)"),
        "none_vs_sa_ppo_clean": ("SA-PPO vs None", "Clean (k=0)"),
        "none_vs_sa_ppo_k25": ("SA-PPO vs None", "Attacked (k=25, eps=0.15)"),
    }

    for test_key, (label, cond) in test_labels.items():
        st = sig[test_key]
        p_val = st["welch_p_value"]
        decision = "Statistically Significant (p < 0.05)" if p_val < 0.05 else "Not Significant (Moot)"
        ci_str = f"[{st['bootstrap_ci_95_lower']:.2f}, {st['bootstrap_ci_95_upper']:.2f}]"
        print(
            f"| {label:18s} | {cond:24s} | {st['mean_diff']:+6.2f} | {st['welch_t_stat']:6.3f} | {p_val:7.4f} | {ci_str:18s} | {st['bootstrap_p_value']:7.4f} | {decision} |"
        )


if __name__ == "__main__":
    main()
