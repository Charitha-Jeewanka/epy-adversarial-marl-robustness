"""Results aggregation utilities for Phase 6 sweep harness (GEMINI.md §8)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def aggregate_sweep_results(sweep_dir: Path | str) -> dict[str, Any]:
    """Scan all completed cell subdirectories in sweep_dir and aggregate summary statistics.

    Args:
        sweep_dir: Directory containing individual sweep cell run subdirectories

    Returns:
        Dictionary containing grouped statistics (mean, std, CI) per (arm, k, epsilon)
    """
    sweep_path = Path(sweep_dir)
    if (sweep_path / "eval").exists():
        sweep_path = sweep_path / "eval"

    if not sweep_path.exists():
        return {"records": [], "summary": {}}

    records: list[dict[str, Any]] = []

    for run_dir in sweep_path.iterdir():
        if not run_dir.is_dir():
            continue

        config_file = run_dir / "resolved_config.yaml"
        eval_file = run_dir / "eval_results.json"

        # Require both resolved_config.yaml and eval_results.json in the cell directory
        if not (config_file.exists() and eval_file.exists()):
            continue

        try:
            with open(config_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            with open(eval_file, encoding="utf-8") as f:
                eval_data = json.load(f)

            arm = config.get("sweep_meta", {}).get("arm", "unknown")
            k = int(config.get("sweep_meta", {}).get("budget_k", config.get("attack", {}).get("budget_k", 0)))
            eps = float(config.get("sweep_meta", {}).get("epsilon", config.get("attack", {}).get("epsilon", 0.0)))
            seed = int(config.get("seed", 0))

            post_attack_return = float(eval_data.get("post_attack_return_mean", 0.0))

            records.append({
                "arm": arm,
                "budget_k": k,
                "epsilon": eps,
                "seed": seed,
                "post_attack_return": post_attack_return,
                "run_dir": str(run_dir),
            })
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as err:
            logger.warning("Skipping invalid run directory %s: %s", run_dir, err)
            continue

    if not records:
        return {"records": [], "summary": {}}

    df_records = pd.DataFrame(records)

    # Group by (arm, budget_k, epsilon) to compute mean, std, and 95% CI
    summary: dict[str, Any] = {}
    grouped = df_records.groupby(["arm", "budget_k", "epsilon"])

    for (arm, k, eps), group in grouped:
        key = f"{arm}_k{k}_eps{eps}"
        returns = group["post_attack_return"].to_numpy()
        n = len(returns)
        mean = float(np.mean(returns))
        std = float(np.std(returns)) if n > 1 else 0.0
        ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0

        summary[key] = {
            "arm": arm,
            "budget_k": k,
            "epsilon": eps,
            "num_seeds": n,
            "mean": mean,
            "std": std,
            "ci95": ci95,
            "seeds": group["seed"].tolist(),
        }

    return {
        "records": records,
        "summary": summary,
        "dataframe": df_records,
    }
