"""eval package for evaluation, results aggregation, and plotting (GEMINI.md §4 & §8)."""
from admarl.eval.aggregate import aggregate_sweep_results
from admarl.eval.evaluator import evaluate_policy
from admarl.eval.plot import generate_robustness_plots

__all__ = [
    "aggregate_sweep_results",
    "evaluate_policy",
    "generate_robustness_plots",
]
