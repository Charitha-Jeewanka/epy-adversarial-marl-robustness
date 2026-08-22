"""Sweep results aggregation utility (GEMINI.md §8)."""
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
    """Scan all run subdirectories in sweep_dir and aggregate summary statistics.

    Args:
        sweep_dir: Directory containing individual sweep run subdirectories

    Returns:
        Dictionary containing grouped statistics (mean, std, CI) per (arm, k, epsilon)
    """
    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        return {"records": [], "summary": {}}

    records: list[dict[str, Any]] = []

    for run_dir in sweep_path.iterdir():
        if not run_dir.is_dir():
            continue

        config_file = run_dir / "resolved_config.yaml"
        metrics_file = run_dir / "metrics.csv"
        meta_file = run_dir / "meta.json"

        if not (config_file.exists() and metrics_file.exists()):
            continue

        try:
            with open(config_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            meta = {}
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)

            df = pd.read_csv(metrics_file)
            if df.empty:
                continue

            last_row = df.iloc[-1].to_dict()

            arm = config.get("sweep_meta", {}).get("arm", "unknown")
            k = config.get("attack", {}).get("budget_k", 0)
            eps = config.get("attack", {}).get("epsilon", 0.0)
            seed = config.get("seed", 0)

            post_attack_return = float(last_row.get("post_attack_return", last_row.get("episode_return_mean", 0.0)))
            eval_file = run_dir / "eval_results.json"
            if eval_file.exists():
                with open(eval_file, encoding="utf-8") as f:
                    eval_data = json.load(f)
                    post_attack_return = float(eval_data.get("post_attack_return_mean", post_attack_return))

            records.append({
                "arm": arm,
                "budget_k": k,
                "epsilon": eps,
                "seed": seed,
                "post_attack_return": post_attack_return,
                "episode_return_mean": float(last_row.get("episode_return_mean", 0.0)),
                "critic_loss": float(last_row.get("critic_loss", 0.0)),
                "policy_loss": float(last_row.get("policy_loss", 0.0)),
                "git_commit": meta.get("git", {}).get("short_commit", "unknown"),
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
