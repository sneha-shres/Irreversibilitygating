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


_DISABLED = 99.0  # sentinel: component effectively removed (threshold never exceeded)


def ablation_configs(tau_d: float, tau_pi: float) -> list[Config]:
    """Return the four ablation configs for the given operating-point thresholds.

    Components are disabled by setting their threshold to _DISABLED (99.0), meaning
    the corresponding profile value never exceeds it and the signal plays no role in
    routing.  tau=0.0 would make the threshold always exceeded (routes everything to
    HIGH), which is wrong for ablation.
    """
    return [
        Config(tau_d=_DISABLED, tau_pi=_DISABLED),  # f only
        Config(tau_d=tau_d,     tau_pi=_DISABLED),  # f + d_I
        Config(tau_d=_DISABLED, tau_pi=tau_pi),     # f + pi
        Config(tau_d=tau_d,     tau_pi=tau_pi),     # full
    ]


def config_to_name(config: Config) -> str:
    """Human-readable name for an ablation config."""
    d_disabled = config.tau_d >= _DISABLED
    pi_disabled = config.tau_pi >= _DISABLED
    if d_disabled and pi_disabled:
        return "f only"
    if pi_disabled:
        return "f + d_I only"
    if d_disabled:
        return "f + π only"
    return "Full IrrGate"