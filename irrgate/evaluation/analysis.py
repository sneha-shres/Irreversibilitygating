from __future__ import annotations

import pandas as pd

from irrgate.config import Config
from irrgate.data.loader import Trajectory
from irrgate.evaluation.metrics import compute_false_positive_rate, compute_recall
from irrgate.evaluation.runner import TrajectoryResult, evaluate_trajectory


def produce_results_table(
    positive_results: list[TrajectoryResult],
    negative_results: list[TrajectoryResult],
) -> pd.DataFrame:
    """Output a table with per-benchmark recall, FPR, MVC (mean LLM calls per trajectory)."""
    # For now, MVC is 0 since we're using stub rubric
    # In real implementation, count actual LLM calls

    # Group by benchmark
    benchmark_data = {}
    all_results = positive_results + negative_results

    for result in all_results:
        benchmark = result.trajectory.benchmark
        if benchmark not in benchmark_data:
            benchmark_data[benchmark] = {"positives": [], "negatives": []}

        if result.trajectory.side_effect_label.lower() == "yes":
            benchmark_data[benchmark]["positives"].append(result)
        else:
            benchmark_data[benchmark]["negatives"].append(result)

    rows = []
    for benchmark, data in benchmark_data.items():
        recall = compute_recall(data["positives"])
        fpr = compute_false_positive_rate(data["negatives"])
        n_pos = len(data["positives"])
        n_neg = len(data["negatives"])
        # MVC: mean LLM calls per trajectory (0 for stub)
        mvc = 0.0

        rows.append({
            "benchmark": benchmark,
            "recall": recall,
            "fpr": fpr,
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "mvc": mvc,
        })

    return pd.DataFrame(rows)


def produce_ablation_table(
    eval_set: tuple[list[Trajectory], list[Trajectory]],
    configs: list[Config],
) -> pd.DataFrame:
    """
    Run the full evaluation under several configs and report recall and FPR for each.

    Configs should include:
      - Full IrrGate (f, d, pi)
      - f only (collapse d, pi to constants)
      - f + d only (no pi)
      - f + pi only (no d)
    """
    positives, negatives = eval_set

    rows = []
    for config in configs:
        positive_results = [evaluate_trajectory(traj, config) for traj in positives]
        negative_results = [evaluate_trajectory(traj, config) for traj in negatives]

        recall = compute_recall(positive_results)
        fpr = compute_false_positive_rate(negative_results)

        rows.append({
            "config": config_to_name(config),
            "recall": recall,
            "fpr": fpr,
        })

    return pd.DataFrame(rows)


def config_to_name(config: Config) -> str:
    """Convert config to a readable name for ablation table."""
    if config.tau_d == 0.0 and config.tau_pi == 0.0:
        return "f only"
    elif config.tau_pi == 0.0:
        return "f + d only"
    elif config.tau_d == 0.0:
        return "f + pi only"
    else:
        return "Full IrrGate"